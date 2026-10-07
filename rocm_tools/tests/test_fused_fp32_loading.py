from types import SimpleNamespace

import pytest
import torch
from exllamav3.modules.linear import Linear
from exllamav3.ext import exllamav3_ext as ext


@pytest.mark.parametrize('batched', [False, True])
def test_fp32_fused_checkpoint_projection_is_loaded_as_fp16(monkeypatch, batched):
    weight = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 17
    tensors = {'packed': weight.unsqueeze(0)} if batched else {'packed.weight': weight}
    def get_tensor(key, device, **kwargs):
        value = tensors.get(key)
        if value is None:
            if kwargs.get('optional'):
                return None
            raise KeyError(key)
        if kwargs.get('fidx') is not None:
            value = value[kwargs['fidx']]
        if kwargs.get('float2half'):
            value = value.half()
        if kwargs.get('transpose'):
            value = value.T
        return value
    storage = SimpleNamespace(
        has_tensor=lambda key: key in tensors,
        has_tensor_group=lambda key, groups: key+'.weight' in tensors,
        get_tensor=get_tensor)
    # This wrapper holds GPU-kernel metadata; the numerical CPU forward below
    # takes the ordinary matmul path and needs no GPU allocation.
    monkeypatch.setattr(ext, 'BC_LinearFP16', lambda weight, bias: None)
    linear = Linear(SimpleNamespace(stc=storage), 'query', 4, 2,
                    fkey='packed', frange=(1, 3), fidx=0 if batched else None,
                    transpose_fused_weights=False, pad_to=1)
    linear.device = torch.device('cpu')
    assert linear.load_fp16('query')
    assert linear.inner.weight.dtype == torch.float16
    x = torch.tensor([[1, 2, 3, 4]], dtype=torch.float16)
    assert torch.equal(linear.inner.forward(x, {}), x @ weight[1:3].half().T)
