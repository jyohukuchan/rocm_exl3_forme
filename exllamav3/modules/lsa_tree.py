"""Incremental D-LSA hierarchy and beam search, following the official Qwen backend.

Node groups are cached by their children's page versions. A view can contain
future nodes, but a query only reads the levels grown at its index-page start.
"""
from collections import OrderedDict
import itertools
import math
import torch
import torch.nn.functional as F

NODE_SCALE = 16.0
_node_versions = itertools.count(1)


def pair_nodes(sums, size):
    groups, count, dim = sums.shape
    labels = torch.arange(count, device=sums.device).expand(groups, count)
    for _ in range(size.bit_length() - 1):
        n = sums.shape[1]
        if sums.is_cuda and 32 <= n <= 4096 and not n & (n - 1) and not dim % 16:
            from ..vendor.lsa.ward import cluster_node_buckets
            pairs = cluster_node_buckets(sums, precision="ieee")
        else:
            from ..vendor.lsa.ward_reference import ward_rounds
            pairs = ward_rounds(sums, torch.ones(sums.shape[:2], device=sums.device), n // 2, 2)[0]
        labels = pairs.gather(1, labels)
        order = pairs.argsort(dim=1, stable=True)
        sums = sums.gather(1, order[..., None].expand_as(sums)).reshape(groups, n // 2, 2, dim).sum(2)
    return labels


class NodeGroups:
    def __init__(self, capacity, block, size, dim, precision, device):
        self.capacity, self.block, self.size = capacity, block, size
        self.precision = precision
        nodes = block // size
        dtype = torch.uint8 if precision == "fp8" else torch.bfloat16
        self.centroids = torch.empty((capacity, nodes, dim), dtype=dtype, device=device)
        self.sums = torch.empty((capacity, nodes, dim), dtype=torch.float32, device=device)
        self.children = torch.empty((capacity, nodes, size), dtype=torch.int32, device=device)
        self.cached = OrderedDict()
        self.versions = [0] * capacity

    def read(self, sums, identities):
        count = len(sums) // self.block
        signatures = [tuple(identities[i * self.block:(i + 1) * self.block]) for i in range(count)]
        missing = list(dict.fromkeys(sig for sig in signatures if sig not in self.cached))
        if missing:
            rows = [signatures.index(sig) for sig in missing]
            grouped = sums[:count * self.block].reshape(count, self.block, -1)[rows]
            labels = pair_nodes(grouped, self.size)
            children = labels.argsort(dim=1, stable=True).reshape(len(rows), -1, self.size)
            ordered = grouped.gather(1, children.flatten(1)[..., None].expand_as(grouped))
            node_sums = ordered.reshape(len(rows), -1, self.size, sums.shape[1]).sum(2)
            centroids = F.normalize(node_sums, dim=-1)
            if self.precision == "fp8":
                centroids = (centroids * NODE_SCALE).to(torch.float8_e4m3fn).view(torch.uint8)
            else:
                centroids = centroids.bfloat16()
            for sig in signatures:
                if sig in self.cached:
                    self.cached.move_to_end(sig)
            for row, sig in enumerate(missing):
                slot = len(self.cached) if len(self.cached) < self.capacity else self.cached.popitem(last=False)[1]
                self.sums[slot].copy_(node_sums[row])
                self.centroids[slot].copy_(centroids[row])
                self.children[slot].copy_(children[row].int())
                self.versions[slot] = next(_node_versions)
                self.cached[sig] = slot
        slot_indices = [self.cached[sig] for sig in signatures]
        slots = torch.tensor(slot_indices, device=sums.device)
        for sig in signatures:
            self.cached.move_to_end(sig)
        centroids = self.centroids[slots].flatten(0, 1)
        if self.precision == "fp8":
            centroids = centroids.view(torch.float8_e4m3fn).float() / NODE_SCALE
        children = self.children[slots].long()
        offsets = torch.arange(count, device=sums.device)[:, None, None] * self.block
        children = (children + offsets).flatten(0, 1)
        # A parent must not recursively hash all descendants for every child.
        # A slot incarnation is unique even after eviction/reuse, while its
        # identity costs O(1) to hash at any depth of the hierarchy.
        identities = [(self.versions[slot], child)
                      for slot in slot_indices for child in range(self.block // self.size)]
        return centroids.float(), children, self.sums[slots].flatten(0, 1), identities


class TreeStore:
    def __init__(self, settings, capacity, dim, device, page_size=None, growth_buckets=4096):
        self.settings = settings
        self.page_size = page_size or math.lcm(8192, settings.window_size)
        self.growth_buckets = growth_buckets
        self.groups = []
        self.count_cache = {}
        for level in range(1, settings.top_level + 1):
            block = settings.at("node_cluster_block_size", level)
            size = settings.at("node_bucket_size", level)
            groups = capacity // block
            if not groups:
                break
            self.groups.append(NodeGroups(groups, block, size, dim, settings.node_centroid_precision, device))
            capacity = groups * (block // size)

    @staticmethod
    def storage_size(settings, capacity, dim):
        total = 0
        for level in range(1, settings.top_level + 1):
            block = settings.at("node_cluster_block_size", level)
            size = settings.at("node_bucket_size", level)
            capacity = capacity // block * (block // size)
            if not capacity:
                break
            total += capacity * (dim * (5 if settings.node_centroid_precision == "fp8" else 6) + size * 4)
        return total

    def view(self, sums, identities):
        levels = []
        for group in self.groups:
            if len(sums) < group.block:
                break
            centroids, children, sums, identities = group.read(sums, identities)
            levels.append((centroids, children))
        return TreeView(self, levels)

    def counts(self, page):
        if page not in self.count_cache:
            s = self.settings
            per_page = self.page_size // s.bucket_size
            covered, made, closed = 0, [0] * s.top_level, [0] * s.top_level
            for end in range(per_page, page * per_page + 1, per_page):
                block = s.at("node_cluster_block_size", 1)
                blocks = (end - covered) // block
                if blocks < max(1, self.growth_buckets // block):
                    continue
                covered += blocks * block
                made[0] += blocks * block // s.at("node_bucket_size", 1)
                for level in range(1, s.top_level):
                    block = s.at("node_cluster_block_size", level + 1)
                    taken = (made[level - 1] - closed[level - 1]) // block * block
                    if not taken:
                        break
                    closed[level - 1] += taken
                    made[level] += taken // s.at("node_bucket_size", level + 1)
            self.count_cache[page] = covered, made, closed
        return self.count_cache[page]


class TreeView:
    def __init__(self, store, levels):
        self.store, self.levels = store, levels

    @staticmethod
    def best(queries, vectors, candidates, count):
        # Candidate ids are ascending before the stable sort: ties choose lower ids.
        if not candidates.shape[1]:
            return candidates
        outputs = []
        chunk = max(1, (1 << 19) // candidates.shape[1])
        for first in range(0, len(queries), chunk):
            ids = candidates[first:first + chunk].sort(dim=1).values
            rows = vectors[ids.clamp_min(0)].float()
            scores = (rows * queries[first:first + chunk, None].float()).sum(-1).masked_fill(ids < 0, -torch.inf)
            keep = scores.argsort(dim=1, descending=True, stable=True)[:, :count]
            outputs.append(ids.gather(1, keep))
        return torch.cat(outputs)

    def pick(self, queries, positions, centroids):
        s = self.store.settings
        result = torch.full((len(queries), s.top_p), -1, dtype=torch.int32, device=queries.device)
        # Prefill pages have one frozen hierarchy; only the uncovered leaf frontier varies per query.
        pages = positions.long() // self.store.page_size
        for page in pages.unique().cpu().tolist():
            rows = (pages == page).nonzero().flatten()
            query = queries[rows]
            covered, made, closed = self.store.counts(page)
            levels = sum(bool(n) for n in made)
            start = levels
            if s.start_nodes is not None:
                start = next((i for i in range(1, levels + 1) if made[i - 1] <= s.start_nodes), levels)
            carried = torch.empty((len(rows), 0), dtype=torch.long, device=queries.device)
            for level in range(start, 0, -1):
                first = 0 if level == start and s.start_nodes is not None else closed[level - 1]
                opened = torch.arange(first, made[level - 1], device=queries.device).expand(len(rows), -1)
                vectors, children = self.levels[level - 1]
                kept = self.best(query, vectors, torch.cat((carried, opened), 1), s.at("beam_width", level))
                carried = children[kept.clamp_min(0)]
                carried = torch.where(kept[..., None] >= 0, carried, -1).flatten(1)
            visible = positions[rows].long() // s.window_size * (s.window_size // s.bucket_size)
            frontier_end = min(len(centroids), int(visible.max()))
            opened = torch.arange(covered, max(covered, frontier_end), device=queries.device).expand(len(rows), -1)
            opened = torch.where(opened < visible[:, None], opened, -1)
            candidates = torch.cat((carried, opened), 1)
            kept = self.best(query, centroids, candidates, s.top_p)
            result[rows, :kept.shape[1]] = kept.int()
        return result.contiguous()
