from types import SimpleNamespace
import pytest
import torch
from exllamav3.modules.attn import Attention
from exllamav3.modules.sliding_attn import SlidingAttention
from exllamav3.modules.multilinear import native_projection_rows_supported
from exllamav3.ext import exllamav3_ext as ext


@pytest.mark.parametrize('kind',[Attention,SlidingAttention])
def test_unsupported_multirow_bundle_uses_the_same_individual_projections(monkeypatch,kind):
    calls=[]
    def linear(name,value):
        return SimpleNamespace(inner=SimpleNamespace(cooperative_gemm_supported=False),
            forward=lambda x,p:(calls.append(name) or torch.full((*x.shape[:-1],3),value)))
    m=SimpleNamespace(multi_qkv=None,multi_qg=None,multi_kv=object(),has_lora=lambda:False,interleaved_gate=False,use_k_as_v=False,
        q_proj=linear('q',1),g_proj=None,k_proj=linear('k',2),v_proj=linear('v',3),
        finish_qkv=lambda q,k,v,g,*a:(q,k,v,g))
    monkeypatch.setattr(ext,'exl3_mgemm',lambda *a:(_ for _ in ()).throw(AssertionError('unsupported cooperative path')))
    q,k,v,g=kind.project_qkv(m,torch.zeros(1,16,4),{})
    assert calls==['q','k','v'] and torch.equal(k,torch.full((1,16,3),2))
    assert torch.equal(v,torch.full((1,16,3),3)) and g is None


def test_single_row_gemv_is_preserved_but_sliced_cooperative_decode_is_not():
    projection=SimpleNamespace(inner=SimpleNamespace(cooperative_gemm_supported=False))
    assert native_projection_rows_supported([projection],1)
    assert not native_projection_rows_supported([projection],16)
    assert not native_projection_rows_supported([projection],1,sliced=True)
    assert native_projection_rows_supported([SimpleNamespace(inner=SimpleNamespace())],16)
