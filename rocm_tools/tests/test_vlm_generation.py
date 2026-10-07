from types import SimpleNamespace
import pytest
import torch
from fastapi.testclient import TestClient
from rocm_tools.jev_runtime import JEVRuntime, apply_processor_chat_template
from rocm_tools.jev_server import create_app


class Template:
    def __init__(self,bos,closing):self.bos=bos;self.closing=closing
    def apply_chat_template(self,messages,add_generation_prompt=False,**kwargs):
        result=self.bos+''.join('<'+m['role']+'>'+m['content']+self.closing for m in messages)
        return result+('<assistant>' if add_generation_prompt else '')


def test_processor_template_removes_the_tokenizers_untrained_default_system_message(tmp_path):
    import json
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    native = "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({'[UNK]': 0}, unk_token='[UNK]')),
        chat_template='<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'+native)
    messages = [{'role':'user','content':'Locate the crafting cell'}]
    assert 'helpful assistant' in tokenizer.apply_chat_template(messages,tokenize=False)
    (tmp_path/'chat_template.json').write_text(json.dumps({'chat_template': native}))
    assert apply_processor_chat_template(tokenizer,tmp_path)=='processor:chat_template.json'
    assert tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)==(
        '<|im_start|>user\nLocate the crafting cell<|im_end|>\n<|im_start|>assistant\n')


def test_absent_processor_template_preserves_tokenizer_template(tmp_path):
    tokenizer=SimpleNamespace(chat_template='existing native template')
    assert apply_processor_chat_template(tokenizer,tmp_path)=='tokenizer'
    assert tokenizer.chat_template=='existing native template'


@pytest.mark.parametrize('bos,closing',[('', '<|im_end|>\n'),('<bos>','<end_of_turn>\n')])
def test_template_boundary_and_delta_do_not_duplicate_bos(bos,closing):
    r=JEVRuntime.__new__(JEVRuntime);r.hf_tokenizer=Template(bos,closing)
    assert r.assistant_closing()==closing
    assert r.render_user_delta('new image')=='<user>new image'+closing+'<assistant>'


def test_bidirectional_image_span_is_not_cut_by_prefill_chunk():
    r=JEVRuntime.__new__(JEVRuntime);r.chunk_size=4
    r.model=SimpleNamespace(caps={'atomic_mm_prefill':True})
    ids=torch.tensor([[1,2,500,501,502,503,504,3]])
    assert list(r.prefill_chunks(ids,[SimpleNamespace(first_index=500,last_index=505)]))==[(0,7),(7,8)]


def test_generation_only_info_and_decision_routes_fail_explicitly():
    r=SimpleNamespace(decision_enabled=False,context=32768,close=lambda:None,
        generate=lambda *a,**kw:{'text':'OK','finish_reason':'stop','usage':{}})
    with TestClient(create_app(r,model_name='small-vlm')) as c:
        info=c.get('/v1/decide/info').json()
        assert info['protocol']=='exl3-vl-chat-v1' and info['decision_enabled'] is False
        assert info['strategies']==[] and info['thinking']==['off','on']
        assert c.post('/v1/decide',json={'model':'small-vlm'}).status_code==400
        assert c.post('/v1/systemone',json={'model':'small-vlm'}).status_code==400
        assert c.post('/v1/chat/completions',json={'model':'small-vlm','messages':[{'role':'user','content':'hi'}]}).json()['choices'][0]['message']['content']=='OK'


def test_non_mrope_architecture_does_not_attempt_qwen_rotary_positions():
    r=JEVRuntime.__new__(JEVRuntime);r.model=SimpleNamespace(g_rope=None)
    assert r.rope_frequencies(torch.tensor([[1]]),[object()],100) is None


def test_historical_think_prefix_changes_do_not_rewrite_cached_tokens():
    class StripsHistoricalThinking(Template):
        def apply_chat_template(self,messages,**kwargs):
            text=super().apply_chat_template(messages,**kwargs)
            if messages[-1]['role']=='assistant':
                text=text.replace('<assistant>','<assistant><think></think>')
            return text
    r=JEVRuntime.__new__(JEVRuntime);r.hf_tokenizer=StripsHistoricalThinking('','<|im_end|>\n')
    r.conversation_closing=r.assistant_closing()
    assert r.render_user_delta('second')=='<user>second<|im_end|>\n<assistant>'


def test_vision_budget_preserves_model_bounds_and_rejects_unsupported_override():
    r=JEVRuntime.__new__(JEVRuntime)
    r.config=SimpleNamespace(vision_pp=SimpleNamespace(min_pixels=4096,max_pixels=1048576,size={'shortest_edge':4096,'longest_edge':1048576}))
    r.configure_vision_budget(524288)
    assert r.config.vision_pp.max_pixels==524288
    assert r.config.vision_pp.size=={'shortest_edge':4096,'longest_edge':524288}
    with pytest.raises(ValueError,match='minimum'):r.configure_vision_budget(1024)
    r.configure_vision_budget(2097152)
    assert r.config.vision_pp.max_pixels==524288  # Never expand a model's configured limit.
    r.config=SimpleNamespace(vision_pp=SimpleNamespace(max_soft_tokens=280))
    r.configure_vision_budget(262144)
    with pytest.raises(ValueError,match='pixel-based'):r.configure_vision_budget(524288)
