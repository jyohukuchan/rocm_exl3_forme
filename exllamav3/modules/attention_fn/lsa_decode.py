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


class _TPReadTables:
    """A bucket table shared by the short-query graph slots of one TP owner."""
    def __init__(self, centroids, members, capacity):
        self.centroids = torch.zeros((capacity, centroids.shape[1]), dtype=centroids.dtype, device=centroids.device)
        self.members = torch.full((capacity, members.shape[1]), -1, dtype=members.dtype, device=members.device)
        self.source = None

    def update(self, centroids, members):
        source = (centroids.data_ptr(), members.data_ptr())
        if source != self.source:
            self.centroids.zero_(); self.members.fill_(-1)
            self.centroids[:len(centroids)].copy_(centroids)
            self.members[:len(members)].copy_(members)
            self.source = source
            self.keepalive = (centroids, members)


def tp_select(indexer, query, positions, centroids, members, batch):
    if (os.environ.get('EXL3_LSA_DECODE_GRAPH', '1') == '0' or not query.is_cuda
            or len(query) > 16 or indexer.settings.decode != 'flat'):
        return indexer.select(query, positions, centroids, members)
    capacity = 1 << (max(1, len(centroids)) - 1).bit_length()
    tables = getattr(indexer, '_tp_read_tables', {})
    graphs = getattr(indexer, '_tp_read_graphs', {})
    current = tables.get(batch)
    if current is None or len(current.centroids) != capacity:
        # A new range drops graphs that point into the old tables.
        graphs = {k: v for k, v in graphs.items() if k[0] != batch}
        current = tables[batch] = _TPReadTables(centroids, members, capacity)
    current.update(centroids, members)
    key = (batch, len(query))
    state = graphs.get(key)
    if state is None:
        state = _TPSelectionGraph(indexer, query, positions, current)
        graphs[key] = state
    else:
        state.query.copy_(query); state.positions.copy_(positions)
    indexer._tp_read_tables, indexer._tp_read_graphs = tables, graphs
    state.graph.replay(); state.replays += 1
    return state.output


class _TPSelectionGraph:
    def __init__(self, indexer, query, positions, tables):
        self.query, self.positions = torch.empty_like(query), torch.empty_like(positions)
        self.query.copy_(query); self.positions.copy_(positions)
        self.tables = tables
        self.replays = 0
        with torch.cuda.device(query.device):
            stream = torch.cuda.Stream(device=query.device)
            stream.wait_stream(torch.cuda.current_stream())
            def run():
                return indexer.select(self.query, self.positions, tables.centroids, tables.members)
            with torch.cuda.stream(stream):
                for _ in range(2): run()
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self.output = run()


def tp_attend(indexer, layer, q, k, v, indices, scale, bt, qc, kv_heads, batch):
    from .qsa_triton import qsa_sparse_attend_rows
    def eager():
        return qsa_sparse_attend_rows(q, k, v, indices, scale, bt, PAGE_SIZE, qc, kv_heads)
    if os.environ.get('EXL3_LSA_DECODE_GRAPH', '1') == '0' or len(q) > 16:
        return eager()
    graphs = getattr(indexer, '_tp_attention_graphs', {})
    key = (id(layer), batch, len(q), tuple(q.shape), k.data_ptr(), v.data_ptr())
    state = graphs.get(key)
    if state is None:
        state = _TPAttentionGraph(layer, q, k, v, indices, scale, bt, qc, kv_heads)
        graphs[key] = state
    else:
        state.stage(q, indices, bt)
    indexer._tp_attention_graphs = graphs
    state.graph.replay(); state.replays += 1
    return state.output


class _TPAttentionGraph:
    def __init__(self, layer, q, k, v, indices, scale, bt, qc, kv_heads):
        from .qsa_triton import qsa_sparse_attend_rows
        self.layer = layer
        self.query, self.indices = torch.empty_like(q), torch.empty_like(indices)
        self.bt = torch.zeros((len(q), layer.max_num_tokens // PAGE_SIZE), dtype=bt.dtype, device=bt.device)
        self.stage(q, indices, bt)
        self.replays = 0
        with torch.cuda.device(q.device):
            stream = torch.cuda.Stream(device=q.device)
            stream.wait_stream(torch.cuda.current_stream())
            def run():
                return qsa_sparse_attend_rows(self.query, k, v, self.indices, scale, self.bt, PAGE_SIZE, qc, kv_heads)
            with torch.cuda.stream(stream):
                for _ in range(2): run()
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self.output = run()

    def stage(self, q, indices, bt):
        self.query.copy_(q); self.indices.copy_(indices)
        self.bt[:, :bt.shape[1]].copy_(bt)
