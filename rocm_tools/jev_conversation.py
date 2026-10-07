"""Exclusive append-only System 2 chat using live KV and GDN recurrent state."""

import time
import uuid


class ConversationConflict(ValueError):
    pass


class JEVConversation:
    def __init__(self, runtime, system):
        if not isinstance(system, str) or not system.strip():
            raise ValueError('A non-empty initial system instruction is required')
        self.runtime=runtime
        self.id=uuid.uuid4().hex
        self.system=system
        self.state=runtime.cache.get_new_state()
        self.ids=runtime.torch.empty((1,0),dtype=runtime.torch.long)
        self.embeddings=[]
        self.turn=0
        self.last_activity=time.monotonic()
        self.closed=False

    def info(self):
        return {'session_id':self.id,'turn':self.turn,'cached_tokens':self.state.position,
                'images_retained':len(self.embeddings),'closed':self.closed,
                'state_reuse':'KV and GDN recurrent state; System 2 only',
                'exclusive':True}

    def close(self):
        if not self.closed:
            self.runtime.torch.cuda.synchronize()
            self.state.free()
            self.embeddings.clear()
            self.closed=True

    def append(self, content, turn, max_tokens=384, temperature=0.0):
        r=self.runtime;t=r.torch
        if self.closed or type(turn) is not int or turn!=self.turn+1:
            raise ConversationConflict('Session turn is closed, repeated or out of order')
        if type(max_tokens) is not int or not 1<=max_tokens<r.context:
            raise ValueError('Invalid completion budget')
        if not isinstance(temperature,(int,float)) or not 0<=temperature<=2:
            raise ValueError('Invalid temperature')
        before=self.state.position
        try:
            text,new_embeddings=r.state_parts(content)
            messages=([{'role':'system','content':self.system}] if self.turn==0 else [])
            messages.append({'role':'user','content':text})
            fragment=r.hf_tokenizer.apply_chat_template(messages,tokenize=False,
                add_generation_prompt=True,enable_thinking=False)
            # Preserve the exact previously generated token IDs. Re-tokenizing a
            # decoded assistant answer could change BPE boundaries and invalidate KV.
            new_ids=r.tokenizer.encode(fragment,encode_special_tokens=True,embeddings=new_embeddings)
            closing=r.tokenizer.encode('<|im_end|>\n',encode_special_tokens=True)
            input_end=before+new_ids.shape[-1]
            if input_end+max_tokens+closing.numel()>r.context:
                raise ValueError('Conversation exceeds configured context; start a new session')
            self.embeddings.extend(new_embeddings)
            self.ids=t.cat((self.ids,new_ids),dim=1)
            freqs=(r.model.g_rope.get_mrope_freqs(self.ids,self.embeddings,
                    input_end+max_tokens+closing.numel())[0] if self.embeddings else None)
            def params():
                return {'attn_mode':'flash_attn','cache':r.cache,'past_len':self.state.position,
                        'batch_shape':(1,r.context),'recurrent_states':[self.state],
                        'loras':(),'last_tokens_only':1,'indexed_embeddings':self.embeddings,
                        'inv_freq':freqs}
            logits=None
            for start in range(0,new_ids.shape[-1],r.chunk_size):
                end=min(new_ids.shape[-1],start+r.chunk_size)
                if end<new_ids.shape[-1]:r.model.prefill(new_ids[:,start:end],params())
                else:logits=r.model.forward(new_ids[:,start:end],params())[0,-1].float()
            tokens=[];reason='length'
            for _ in range(max_tokens):
                logits=logits[:r.tokenizer.actual_vocab_size]
                if not bool(t.isfinite(logits).all()):raise RuntimeError('Non-finite session logits')
                token=(int(logits.argmax()) if temperature==0 else int(t.multinomial(t.softmax(logits/temperature,dim=-1),1)))
                if token in r.eos_ids:
                    reason='stop';break
                tokens.append(token)
                ids=t.tensor([[token]],dtype=t.long)
                logits=r.model.forward(ids,params())[0,-1].float()
            if reason=='length':
                raise ValueError('Truncated session answer; conversation invalidated')
            # Commit the canonical assistant closing delimiter, including newline,
            # so the next user fragment appends at an exact turn boundary.
            r.model.prefill(closing,params())
            generated=t.tensor([tokens],dtype=t.long)
            self.ids=t.cat((self.ids,generated,closing),dim=1)
            if self.ids.shape[-1]!=self.state.position:
                raise RuntimeError('Cached token and recurrent-state positions diverged')
            self.turn=turn;self.last_activity=time.monotonic()
            answer=r.tokenizer.decode(generated[0],decode_special_tokens=True) if tokens else ''
            return {'text':answer,'finish_reason':reason,'session':self.info(),
                    'usage':{'prompt_tokens':input_end,'completion_tokens':len(tokens),
                             'total_tokens':input_end+len(tokens),
                             'prompt_tokens_details':{'cached_tokens':before},
                             'prefilled_tokens':new_ids.shape[-1]},
                    'input_images_added':len(new_embeddings)}
        except BaseException:
            self.close()
            raise
