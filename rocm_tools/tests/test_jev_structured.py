import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

from rocm_tools.jev_conversation import JEVConversation
from rocm_tools.jev_structured import SessionJSONConstraint
from test_jev_conversation import make_runtime


def test_http_session_forwards_format_and_reports_native_validation():
    from fastapi.testclient import TestClient
    from rocm_tools.jev_server import create_app
    received=[]
    def append(*args,**kwargs):
        received.append(kwargs['response_format'])
        return {'text':'{"approved":false}','finish_reason':'stop','usage':{},'session':{},
                'input_images_added':0,'structured_output':{'validated':True,'backend':'native_llguidance'}}
    runtime=SimpleNamespace(close=lambda:None,decision_enabled=False,append_conversation=append)
    format={'type':'json_object'}
    with TestClient(create_app(runtime,model_name='small')) as client:
        result=client.post('/v1/chat/sessions/s1',json={'model':'small','turn':1,'content':'review','response_format':format})
    assert result.status_code==200 and received==[format]
    assert result.json()['structured_output']['validated'] is True


def test_final_json_mask_does_not_constrain_forced_native_thought_boundary(monkeypatch):
    runtime,state,calls=make_runtime(monkeypatch)
    runtime.reasoning_end_id=88;runtime.reasoning_close_ids=torch.tensor([[88,10]])
    accepted=[]
    def mask(logits):
        result=torch.full_like(logits,float('-inf'));result[8]=0;return result
    constraint=SimpleNamespace(mask=mask,accept=lambda token:(accepted.append(token) or True),
                               finish=lambda:{'validated':True,'constrained_tokens':len(accepted)})
    monkeypatch.setattr('rocm_tools.jev_structured.build_session_constraint',lambda *a:constraint)
    session=JEVConversation(runtime,'controls',enable_thinking=True)
    out=session.append('image',1,reasoning_budget=0,response_format={'type':'json_object'})
    assert accepted==[8]
    assert session.ids.tolist()==[[1,2,3,88,10,8,99,10]]
    assert state.position==8 and out['structured_output']['validated']


def test_native_matcher_masks_early_eos_and_enforces_enum_fields():
    pytest.importorskip('llguidance')
    directory=Path(os.environ.get('EXL3_JSON_TEST_MODEL','/models/safetensors/gemma-4-26B-A4B-it-exl3-4.10bpw'))
    if not directory.is_dir():pytest.skip('Native tokenizer fixture is not mounted')
    from exllamav3 import Config,Tokenizer
    tokenizer=Tokenizer.from_config(Config.from_directory(str(directory)))
    runtime=SimpleNamespace(torch=torch,tokenizer=tokenizer,reasoning_end_id=88)
    schema={'type':'json_schema','json_schema':{'name':'input','strict':True,'schema':{
        'type':'object','properties':{'op':{'enum':['tap_key']},'key':{'enum':['e']}},
        'required':['op','key'],'additionalProperties':False}}}
    constraint=SessionJSONConstraint(runtime,schema)
    for _ in range(64):
        logits=torch.zeros(tokenizer.actual_vocab_size);logits[tokenizer.eos_token_id]=100
        filtered=constraint.mask(logits);token=int(filtered.argmax())
        if token==tokenizer.eos_token_id:break
        if constraint.accept(token):break
    else:raise AssertionError('Grammar did not finish')
    result=constraint.finish()
    assert result['validated'] and result['constrained_tokens']>0
    assert json.loads(tokenizer.decode(torch.tensor(constraint.tokens)))=={'op':'tap_key','key':'e'}
