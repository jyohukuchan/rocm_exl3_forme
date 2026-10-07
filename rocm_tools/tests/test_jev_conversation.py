from types import SimpleNamespace
import pytest
import torch
from rocm_tools.jev_conversation import JEVConversation,ConversationConflict


class FakeTokenizer:
    actual_vocab_size=100
    def encode(self,text,**kwargs):
        return torch.tensor([[99,10]] if text=='<|im_end|>\n' else [[1,2,3]])
    def decode(self,ids,**kwargs):return 'OK'


def make_runtime(monkeypatch):
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    state=SimpleNamespace(position=0,freed=False)
    state.free=lambda:setattr(state,'freed',True)
    calls=[]
    def run(ids,params):
        assert params['past_len']==state.position
        assert params['loras']==()
        state.position+=ids.shape[-1]
        calls.append((ids.numel(),params['past_len']))
    def forward(ids,params):
        run(ids,params)
        logits=torch.zeros((1,1,100))
        logits[0,0,99 if ids[0,0]==7 else 7]=10
        return logits
    r=SimpleNamespace(torch=torch,cache=SimpleNamespace(get_new_state=lambda:state),
        tokenizer=FakeTokenizer(),hf_tokenizer=SimpleNamespace(apply_chat_template=lambda *a,**kw:'fragment'),
        context=1024,chunk_size=1024,eos_ids=[99],state_parts=lambda content:(content,[]),
        model=SimpleNamespace(prefill=run,forward=forward))
    return r,state,calls


def test_append_reuses_kv_and_recurrent_position_without_prefilling_old_tokens(monkeypatch):
    r,state,calls=make_runtime(monkeypatch)
    s=JEVConversation(r,'initial controls')
    first=s.append('frame 1',1)
    second=s.append('frame 2',2)
    assert first['text']=='OK' and second['text']=='OK'
    assert first['usage']['prompt_tokens_details']['cached_tokens']==0
    assert second['usage']['prompt_tokens_details']['cached_tokens']==6
    assert second['usage']['prefilled_tokens']==3
    assert state.position==12 and s.ids.shape[-1]==12
    assert calls==[(3,0),(1,3),(2,4),(3,6),(1,9),(2,10)]
    s.close();assert state.freed


def test_duplicate_turn_is_rejected_without_reversing_or_advancing_state(monkeypatch):
    r,state,calls=make_runtime(monkeypatch);s=JEVConversation(r,'controls')
    s.append('frame',1);previous=list(calls)
    with pytest.raises(ConversationConflict):s.append('duplicate',1)
    assert calls==previous and not state.freed


def test_partial_generation_failure_invalidates_the_session(monkeypatch):
    r,state,calls=make_runtime(monkeypatch);s=JEVConversation(r,'controls')
    r.model.forward=lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('GPU error'))
    with pytest.raises(RuntimeError):s.append('frame',1)
    assert s.closed and state.freed


def test_thinking_mode_is_fixed_at_setup_and_used_for_later_turns(monkeypatch):
    r,state,calls=make_runtime(monkeypatch);flags=[]
    r.hf_tokenizer.apply_chat_template=lambda *a,**kw:(flags.append(kw['enable_thinking']) or 'fragment')
    s=JEVConversation(r,'controls',enable_thinking=True)
    s.append('frame 1',1);s.append('frame 2',2)
    assert flags==[True,True] and s.info()['enable_thinking'] is True
    assert state.position==12
    with pytest.raises(ValueError,match='boolean'):JEVConversation(r,'controls',enable_thinking='on')
