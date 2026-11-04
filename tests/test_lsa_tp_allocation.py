from exllamav3.model.model_tp_alloc import TPAllocation, TPAllocator


def test_index_owner_follows_active_attention_rank_and_memory_budget():
    components=[]
    for i in range(8):
        key=f'attn{i}'
        components.extend([
            TPAllocation(key,channel_width=1,channels_to_split=4,storage_to_split=400),
            TPAllocation(key+'.index',channel_width=1,channels_to_split=1,storage_to_split=200,
                         max_devices=1,affinity_key=key)])
    allocator=TPAllocator(components,1,1)
    allocator.initial_split([2000,3000])
    plan=allocator.compile_tp_plan()
    owners=[]
    for i in range(8):
        owner=next(d for d,p in enumerate(plan) if p[f'attn{i}.index'][1]>p[f'attn{i}.index'][0])
        assert plan[owner][f'attn{i}'][1]>plan[owner][f'attn{i}'][0]
        owners.append(owner)
    assert owners.count(1)>=owners.count(0)
