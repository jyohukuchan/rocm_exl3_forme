"""Flat LSA decode, captured as a whole attention block on ROCm/CUDA.

Clustering and page/version checks stay outside the graph. Shapes grow in
powers of two; one resident graph per attention module bounds graph memory.
"""
import os
import torch
from ...cache import CacheLayer
from ...cache.lsa import CacheLayer_lsa, CacheLayer_lsa_quant
from ...constants import PAGE_SIZE
from ...util.tensor import get_for_device


def try_lsa_decode(module, x, params):
    indexer = module.lsa_indexer
    if (os.environ.get('EXL3_LSA_DECODE_GRAPH', '1') == '0'
            or params.get('_lsa_graph_state') is not None
            or indexer.settings.decode != 'flat' or indexer.tp_mode
            or x.shape[1] != 1 or not params.get('causal', True)
            or params.get('non_causal_spans') is not None or params.get('sim_kvq')
            or module.has_lora()):
        return None
    cache = params['cache']
    if module.has_split_cache:
        cache = module.tp_cache_lookup[cache]
    layer = cache if isinstance(cache, CacheLayer) else cache.layers[module.layer_idx, params.get('layer_instance') or 0]
    if not (type(layer) is CacheLayer_lsa or type(layer) is CacheLayer_lsa_quant and not layer.compand_a):
        return None
    past = get_for_device(params, 'cache_seqlens', 'cpu')
    if any(indexer.can_read_dense(int(p) + 1) for p in past):
        return None
    bt = get_for_device(params, 'block_table', module.device)
    seq = get_for_device(params, 'cache_seqlens', module.device)
    # A replay writes this page's projected key. Version changes remain visible
    # to paging/overwrite checks, including windows smaller than a cache page.
    layer.mark_index_pages(bt, seq, 1)
    assembled = [layer.buckets(indexer, bt[b], int(p) // indexer.settings.window_size)
                 for b, p in enumerate(past)]
    buckets = max(1, max(c.shape[0] for c, _ in assembled))
    capacity = 1 << (buckets - 1).bit_length()
    position_ids = get_for_device(params, 'position_ids', module.device, None)
    positions = get_for_device(params, 'positions', module.device, None)
    inv_freq = get_for_device(params, 'inv_freq', module.device, None)
    key = (id(layer), tuple(x.shape), x.dtype, capacity,
           tuple(position_ids.shape) if position_ids is not None else None,
           inv_freq.data_ptr() if inv_freq is not None else None,
           tuple(t.data_ptr() for t in layer.get_tensors()))
    state = getattr(indexer, 'decode_graph', None)
    if state is None or state.key != key:
        # Release the old graph before reserving a larger context workspace.
        indexer.decode_graph = None
        state = None
        state = LSADecodeGraph(key, module, layer, x, params, bt, seq, assembled,
                               positions, position_ids, inv_freq)
        indexer.decode_graph = state
    else:
        state.stage(x, bt, seq, assembled, positions, position_ids, params.get('position', 0))
    state.graph.replay()
    state.replays += 1
    return state.output


class LSADecodeGraph:
    def __init__(self, key, module, layer, x, params, bt, seq, assembled, positions, position_ids, inv_freq):
        self.key = key
        self.layer = layer
        self.replays = 0
        self.x = torch.empty_like(x)
        self.past = torch.empty_like(seq)
        self.bt = torch.zeros((len(seq), layer.index_shape[0]), device=x.device, dtype=bt.dtype)
        self.positions = torch.empty_like(seq) if position_ids is None else None
        self.position_ids = torch.empty_like(position_ids) if position_ids is not None else None
        cap = key[3]
        s = module.lsa_indexer.settings
        self.buckets = [(torch.zeros((cap, module.lsa_indexer.index_dim), device=x.device),
                         torch.full((cap, s.bucket_cap), -1, dtype=torch.int32, device=x.device))
                        for _ in seq]
        self.sources = None
        self.stage(x, bt, seq, assembled, positions, position_ids, params.get('position', 0))
        p = {k: v for k, v in params.items() if k not in ('dev_cache', 'positions', 'position_ids')}
        p.update(cache=layer, cache_seqlens=self.past, block_table=self.bt,
                 positions=self.positions, position_ids=self.position_ids,
                 inv_freq=inv_freq, position=0, _lsa_graph_state=self)
        # Cache resolution has already happened; the recursive forward must not
        # treat this concrete layer as a TP cache id.
        assert not module.has_split_cache
        with torch.cuda.device(x.device):
            stream = torch.cuda.Stream(device=x.device)
            stream.wait_stream(torch.cuda.current_stream(x.device))
            with torch.cuda.stream(stream):
                for _ in range(2):
                    module.decode_flash_attn(self.x, x.shape[0], 1, p)
            torch.cuda.current_stream(x.device).wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self.output = module.decode_flash_attn(self.x, x.shape[0], 1, p)

    def stage(self, x, bt, seq, assembled, positions, position_ids, position):
        self.x.copy_(x)
        self.past.copy_(seq)
        self.bt[:, :bt.shape[1]].copy_(bt)
        if self.position_ids is not None:
            self.position_ids.copy_(position_ids)
        elif positions is not None:
            self.positions.copy_(positions)
        else:
            self.positions.fill_(position)
        sources = tuple((c.data_ptr(), m.data_ptr()) for c, m in assembled)
        if sources != self.sources:
            for (dst_c, dst_m), (c, m) in zip(self.buckets, assembled):
                dst_c.zero_(); dst_m.fill_(-1)
                dst_c[:len(c)].copy_(c); dst_m[:len(m)].copy_(m)
            # Retain the tensors too: allocator pointer reuse must not be mistaken
            # for an unchanged bucket table after eviction or prefix replacement.
            self.sources = sources
            self.source_tensors = assembled
