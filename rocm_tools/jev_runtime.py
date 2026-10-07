"""Serialized native EXL3 runtime for VL chat and optional JEV decisions.

Single GPU (or single-process layer split) only: per-request adapter selection,
separate fresh recurrent state, exact decision head, bounded cached prefill.
"""
from __future__ import annotations
from contextlib import contextmanager
import json
from pathlib import Path
import time
import resource
import os


def parse_cache_quant(value):
    """Return independent K/V widths; None preserves the FP16 baseline."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = tuple(int(part.strip()) for part in value.split(','))
        except ValueError as exc:
            raise ValueError('cache_quant must be k_bits,v_bits in 2..8') from exc
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(type(bits) is not int or not 2 <= bits <= 8 for bits in value)):
        raise ValueError('cache_quant must be k_bits,v_bits in 2..8')
    return tuple(value)


class JEVRuntime:
    def __init__(self, directory, *, context=16384, chunk_size=1024, vision=True, gpu_split=None,
                 load_no_forward=False, cache_quant=None, decision=True, vision_max_pixels=262144):
        self.cache_quant = parse_cache_quant(cache_quant)
        self.conversation = None
        self.decision_enabled = decision
        self.profile = self.lora = self.head_weights = None
        if resource.getrlimit(resource.RLIMIT_NOFILE)[0] < 4096:
            raise ValueError('JEV requires at least 4096 open files; launch with ulimit -n 65536')
        import torch
        from exllamav3 import Config, Model, Cache, Tokenizer
        from exllamav3.model.lora import LoRA
        from exllamav3.model.decision import DecisionProfile
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
        self.torch = torch
        self.directory = Path(directory)
        self.context, self.chunk_size = context, chunk_size
        self.config = Config.from_directory(directory)
        self.configure_vision_budget(vision_max_pixels)
        if not 256 <= context <= self.config.max_position_embeddings or context % 256:
            raise ValueError('Context must be a multiple of 256 within the model limit')
        self.tokenizer = Tokenizer.from_config(self.config)
        self.hf_tokenizer = AutoTokenizer.from_pretrained(directory)
        generation_path=self.directory/'generation_config.json'
        generation = json.loads(generation_path.read_text()) if generation_path.exists() else {}
        eos = generation.get('eos_token_id',self.hf_tokenizer.eos_token_id or self.tokenizer.eos_token_id)
        self.eos_ids = eos if isinstance(eos,list) else [eos]
        if decision:
            self.profile = DecisionProfile.from_directory(directory,
                lambda s: self.tokenizer.encode(s).flatten().tolist())
        self.model = Model.from_config(self.config)
        cache_kwargs = {}
        if self.cache_quant is not None:
            from exllamav3.cache import CacheLayer_quant
            cache_kwargs = {'layer_type': CacheLayer_quant,
                            'k_bits': self.cache_quant[0], 'v_bits': self.cache_quant[1]}
        self.cache = Cache(self.model, max_num_tokens=context, max_batch_size=1, **cache_kwargs)
        load_args = {'use_per_device':gpu_split} if gpu_split else {'device':'cuda:0'}
        span_reserve=512 if self.model.caps.get('atomic_mm_prefill',False) else 0
        self.model.load(**load_args, max_chunk_size=chunk_size+span_reserve, max_output_size=1,
                        autosplit_no_forward=load_no_forward)
        devices = {torch.device(m.device) for m in self.model if m.device is not None}
        self.device_indices = sorted({d.index if d.index is not None else torch.cuda.current_device()
                                     for d in devices if d.type=='cuda'})
        if decision:
            self.lora = LoRA.from_directory(self.model, str(self.directory/'adapter_vllm'), strict=True,
                                           dtype=torch.float32)
            self.lora.enabled = False
            sidecar = load_file(str(self.directory/'decision_rows.safetensors'))
            if sidecar['token_ids'].tolist() != self.profile.all_ids:
                raise ValueError('Decision rows and tokenizer differ')
            head = self.model.modules[-1]
            a, b = head.lora_a_tensors[self.lora], head.lora_b_tensors[self.lora]
            row_ids = sidecar['token_ids'].to(head.device)
            self.head_weights = (sidecar['base_rows'].float().T.to(head.device)
                                 + a.float() @ b.float()[:,row_ids]).contiguous()
            if not bool(torch.isfinite(self.head_weights).all()):
                raise ValueError('Non-finite decision head')
            self.head_lookup = {v:i for i,v in enumerate(self.profile.all_ids)}
        self.vision_model = None
        if vision:
            self.vision_model = Model.from_config(self.config, component='vision')
            pp=self.config.vision_pp
            patch_size=getattr(pp,'patch_size',None)
            patch_budget=(pp.max_pixels+patch_size**2-1)//patch_size**2 if patch_size and hasattr(pp,'max_pixels') else 0
            vision_chunk=max(1024,((patch_budget+255)//256)*256) if patch_budget else 1024
            self.vision_model.load(device='cuda:0', max_chunk_size=vision_chunk)
        self.conversation_closing = self.assistant_closing()
        if os.getenv('EXL3_VLM_UNFUSED_PROJECTIONS')=='1':
            # Diagnostic/reference path: the same individual Linear projections
            # without pointer-batched or graph-captured projection bundles.
            for module in self.model:
                for attribute in ['multi_qkv','multi_kv','multi_qg']:
                    if hasattr(module,attribute):setattr(module,attribute,None)
                if hasattr(module,'multi_gu'):
                    module.multi_gu=[None]*len(module.multi_gu)
                    module.bc=None
        if os.getenv('EXL3_VLM_REFERENCE_MOE')=='1':
            # Isolate expert batching from the same quantized Linear weights.
            for module in self.model:
                if type(module).__name__=='BlockSparseMLP':
                    module.bc=None
                    module.support_quant_paths=False
                    module.fused_mode_buffers=None
        if os.getenv('EXL3_VLM_TRACE')=='1':
            trace_modules=[]
            for top in self.model.modules[:2]:trace_modules.extend(list(top))
            for index,module in enumerate(trace_modules):
                original=module.forward
                label=f'{index}:{type(module).__name__}:{getattr(module,"key","")}'
                def traced(*args,_original=original,_label=label,**kwargs):
                    print('VLM module start',_label,flush=True)
                    result=_original(*args,**kwargs)
                    torch.cuda.synchronize()
                    print('VLM module done',_label,flush=True)
                    return result
                module.forward=traced

    def configure_vision_budget(self, max_pixels):
        if type(max_pixels) is not int or max_pixels<=0:
            raise ValueError('vision_max_pixels must be a positive integer')
        self.vision_max_pixels=max_pixels
        pp=getattr(self.config,'vision_pp',None)
        if pp is None or not hasattr(pp,'max_pixels'):
            if max_pixels!=262144:
                raise ValueError('This architecture does not expose a pixel-based vision budget')
            return
        if max_pixels<pp.min_pixels:
            raise ValueError('Vision pixel budget must not be below the preprocessor minimum')
        pp.max_pixels=min(pp.max_pixels,max_pixels)
        if hasattr(pp,'size'):pp.size=dict(pp.size,longest_edge=pp.max_pixels)

    def vision_info(self):
        pp=getattr(self.config,'vision_pp',None)
        return {'enabled':self.vision_model is not None,'requested_max_pixels':self.vision_max_pixels,
                'configured_max_pixels':getattr(pp,'max_pixels',None),
                'min_pixels':getattr(pp,'min_pixels',None),
                'max_soft_tokens':getattr(pp,'max_soft_tokens',None)}

    def image_info(self, embeddings):
        result=[]
        for embedding in embeddings:
            record={'embedding_tokens':int(embedding.embeddings.shape[0])}
            metadata=getattr(embedding,'metadata',{})
            for key in ['original_size','preprocessed_size']:
                if key in metadata:record[key]=[int(v) for v in metadata[key]]
            result.append(record)
        return result

    def assistant_closing(self):
        marker='EXL3_ASSISTANT_BOUNDARY_749c81'
        rendered=self.hf_tokenizer.apply_chat_template([
            {'role':'user','content':'boundary probe'}, {'role':'assistant','content':marker}],
            tokenize=False,add_generation_prompt=False,enable_thinking=False)
        if rendered.count(marker)!=1:raise ValueError('Cannot derive assistant boundary from chat template')
        closing=rendered.split(marker,1)[1]
        if not closing:raise ValueError('Chat template has no assistant closing delimiter')
        return closing

    def render_user_delta(self, content, enable_thinking=False):
        """Serialize a later user turn without repeating BOS or setup tokens."""
        marker='EXL3_PREVIOUS_ASSISTANT_51e2d7'
        prior=[{'role':'user','content':'EXL3_DELTA_PROBE'},{'role':'assistant','content':marker}]
        kwargs={'tokenize':False,'enable_thinking':enable_thinking}
        after=self.hf_tokenizer.apply_chat_template(prior+[{'role':'user','content':content}],
            add_generation_prompt=True,**kwargs)
        if after.count(marker)!=1:raise ValueError('Cannot identify previous assistant boundary')
        tail=after.split(marker,1)[1]
        closing=getattr(self,'conversation_closing',None) or self.assistant_closing()
        if not tail.startswith(closing):raise ValueError('Historical assistant closing differs from cached boundary')
        # Some Qwen templates strip prior empty thinking tags when re-rendered.
        # Keep the real cached assistant tokens and only take the next user tail.
        return tail[len(closing):]

    def rope_frequencies(self, ids, embeddings, length):
        rope=getattr(self.model,'g_rope',None)
        if not embeddings or rope is None or getattr(rope,'mrope_section',None) is None:return None
        return rope.get_mrope_freqs(ids,list(embeddings),length)[0]

    def prefill_chunks(self, ids, embeddings):
        """Keep bidirectional vision spans atomic, including at chunk boundaries."""
        n=ids.shape[-1];start=0
        atomic=bool(embeddings) and self.model.caps.get('atomic_mm_prefill',False)
        spans=[(e.first_index,e.last_index) for e in embeddings] if atomic else []
        values=ids[0].tolist() if atomic else None
        while start<n:
            end=min(start+self.chunk_size,n)
            if atomic:
                for first,last in spans:
                    if end<n and first<=values[end-1]<last and first<=values[end]<last:
                        while end<n and first<=values[end]<last:end+=1
                        break
            yield start,end
            start=end

    def cache_info(self):
        """Report actual resident KV tensors, not just the requested cache flags."""
        classes, widths, byte_count = {}, set(), 0
        sample = None
        for layer in self.cache.layers.values():
            name = type(layer).__name__
            classes[name] = classes.get(name, 0) + 1
            bits = (getattr(layer, 'k_bits', None), getattr(layer, 'v_bits', None))
            widths.add(bits)
            tensors = [tensor for tensor in layer.get_tensors() if tensor is not None]
            byte_count += sum(tensor.numel()*tensor.element_size() for tensor in tensors)
            if sample is None:
                sample = {'class': name, 'k_bits': bits[0], 'v_bits': bits[1],
                          'tensors': [{'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                                       'device': str(tensor.device),
                                       'bytes': tensor.numel()*tensor.element_size()} for tensor in tensors]}
        return {'requested': list(self.cache_quant) if self.cache_quant else 'fp16',
                'context': self.context, 'layer_classes': classes,
                'observed_bits': [list(bits) for bits in sorted(widths, key=str)],
                'kv_tensor_bytes': byte_count, 'sample': sample,
                'recurrent_layers': len(self.cache.recurrent_layers),
                'recurrent_quantization': 'unchanged; KV widths do not quantize model recurrent state'}

    def state_parts(self, value):
        """Preserve image/text order using the same MM embeddings as ordinary VL."""
        if isinstance(value,str): return value,[]
        if isinstance(value,dict): return json.dumps(value,ensure_ascii=False),[]
        if not isinstance(value,list): raise ValueError('State must be text, JSON object or a list')
        from rocm_tools.exl3_server.vision import image_bytes,decode_image
        text,embeddings = [],[]
        for part in value:
            if isinstance(part,str): text.append(part)
            elif isinstance(part,dict) and (part.get('type')=='image_url' or 'image' in part):
                if self.vision_model is None: raise ValueError('Vision is disabled')
                if len(embeddings)>=16: raise ValueError('At most 16 images are supported')
                image_part = part if part.get('type')=='image_url' else {'image_url':{'url':part['image']}}
                data,_ = image_bytes(image_part,20*1024**2,False)
                image = decode_image(data,16777216)
                try:
                    if os.getenv('EXL3_VLM_TRACE')=='1':print('VLM trace: vision forward start',flush=True)
                    emb = self.vision_model.get_image_embeddings(self.tokenizer,image)
                    if os.getenv('EXL3_VLM_TRACE')=='1':print('VLM trace: vision forward done',tuple(emb.embeddings.shape),flush=True)
                finally: image.close()
                if not bool(self.torch.isfinite(emb.embeddings).all()):
                    raise RuntimeError('Non-finite image embeddings')
                embeddings.append(emb)
                text.append(emb.text_alias)
            elif isinstance(part,dict) and part.get('type')=='text': text.append(part['text'])
            else: text.append(json.dumps(part,ensure_ascii=False))
        return ''.join(text),embeddings

    @contextmanager
    def session(self,prompt,embeddings=(),*,adapter=False,decision=False,reserve=0):
        self.expire_conversation()
        if getattr(self,'conversation',None) is not None:
            from rocm_tools.jev_conversation import ConversationConflict
            raise ConversationConflict('An exclusive chat session is active; close it before stateless/System 1 requests')
        # Cache tensors are allocated by Model.load in inference mode. State
        # clearing and the entire request must use the same thread-local mode.
        with self.torch.inference_mode():
            with self._session(prompt,embeddings,adapter=adapter,decision=decision,reserve=reserve) as session:
                yield session

    def expire_conversation(self):
        conversation=getattr(self,'conversation',None)
        if conversation is not None and (conversation.closed or time.monotonic()-conversation.last_activity>180):
            with self.torch.inference_mode():conversation.close()
            self.conversation=None

    def open_conversation(self, system, enable_thinking=False):
        from rocm_tools.jev_conversation import JEVConversation,ConversationConflict
        self.expire_conversation()
        if self.conversation is not None:raise ConversationConflict('A chat session is already active')
        with self.torch.inference_mode():
            self.conversation=JEVConversation(self,system,enable_thinking=enable_thinking)
        return self.conversation.info()

    def append_conversation(self, session_id, content, turn, max_tokens=384, temperature=0.0):
        from rocm_tools.jev_conversation import ConversationConflict
        self.expire_conversation()
        if self.conversation is None or self.conversation.id!=session_id:
            raise ConversationConflict('Unknown or expired chat session')
        with self.torch.inference_mode():
            return self.conversation.append(content,turn,max_tokens,temperature)

    def close_conversation(self, session_id):
        from rocm_tools.jev_conversation import ConversationConflict
        if self.conversation is None:return {'closed':True}
        if self.conversation.id!=session_id:raise ConversationConflict('Session id does not match')
        with self.torch.inference_mode():self.conversation.close()
        self.conversation=None
        return {'closed':True}

    @contextmanager
    def _session(self,prompt,embeddings=(),*,adapter=False,decision=False,reserve=0):
        if (adapter or decision) and not self.decision_enabled:
            raise ValueError('This checkpoint has no trained decision head; use chat generation')
        ids = self.tokenizer.encode(prompt,encode_special_tokens=True,embeddings=list(embeddings))
        n = ids.shape[-1]
        if n<1 or n+reserve>self.context: raise ValueError('Request exceeds the configured context')
        freqs = self.rope_frequencies(ids,embeddings,n+reserve)
        state = self.cache.get_new_state()
        def params(s):
            return {'attn_mode':'flash_attn','cache':self.cache,'past_len':s,
                    'batch_shape':(1,self.context),'recurrent_states':[state],
                    'loras':(self.lora,) if adapter else (), 'last_tokens_only':1,
                    'head_override':self.head_weights if decision else None,
                    'indexed_embeddings':list(embeddings),'inv_freq':freqs}
        try:
            logits = None
            for s,e in self.prefill_chunks(ids,embeddings):
                if os.getenv('EXL3_VLM_TRACE')=='1':print('VLM trace: text prefill',s,e,flush=True)
                if e<n: self.model.prefill(ids[:,s:e],params(s))
                else: logits = self.model.forward(ids[:,s:e],params(s))[0,-1].float()
            if not bool(self.torch.isfinite(logits).all()): raise RuntimeError('Non-finite logits')
            yield logits,n,params
        finally:
            for device in self.device_indices:self.torch.cuda.synchronize(device)
            state.free()

    def decide(self,kind,state,question,options=None,*,strategy='single'):
        if not self.decision_enabled:raise ValueError('This checkpoint has no trained decision head; use chat generation')
        start = time.monotonic()
        text,embeddings = self.state_parts(state)
        prepared = self.profile.question(kind,question,options)
        if strategy not in ('single','tournament','permute'):raise ValueError('Invalid choice strategy')
        requests,prompt_tokens = 0,0
        def one(opts):
            nonlocal requests,prompt_tokens
            q = self.profile.question(kind,question,opts)
            with self.session('[kind] '+kind+'\n[state] '+text+q['suffix'],embeddings,
                              adapter=True,decision=True) as (logits,n,_):
                raw = [logits[self.head_lookup[i]].item() for i in q['ids']]
                prompt_tokens+=n;requests+=1
                return self.profile.probabilities(raw,q)
        opts = prepared['options']
        if kind!='choice' or len(opts)<=16 or strategy=='single':p=one(opts)
        elif strategy=='permute':
            import random,zlib
            rng=random.Random(zlib.crc32(question.encode()))
            orders=[list(range(len(opts)))]+[rng.sample(range(len(opts)),len(opts)) for _ in range(3)]
            p=[0.0]*len(opts)
            for order in orders:
                for i,v in zip(order,one([opts[j] for j in order])):p[i]+=v/4
        else:
            import math
            groups=math.ceil(len(opts)/16);width,extra=divmod(len(opts),groups)
            grouped=[];in_group={};start_i=0
            for j in range(groups):
                end=start_i+width+(j<extra)
                indices=list(range(start_i,end));grouped.append(indices)
                in_group.update(zip(indices,one(opts[start_i:end])))
                start_i=end
            chosen=[max(g,key=lambda i:(in_group[i],-i)) for g in grouped]
            rest=sorted((i for i in range(len(opts)) if i not in chosen),key=lambda i:(-in_group[i],i))
            finalists=sorted(chosen+rest[:max(0,16-len(chosen))])
            final=dict(zip(finalists,one([opts[i] for i in finalists])))
            group_of={i:j for j,g in enumerate(grouped) for i in g}
            share=[0.0]*groups;cap=[0.0]*groups
            for i in finalists:share[group_of[i]]+=final[i];cap[group_of[i]]+=in_group[i]
            among=sum(a*b for a,b in zip(share,cap))
            p=[final[i]*among if i in final else share[group_of[i]]*in_group[i] for i in range(len(opts))]
            total=sum(p);p=[v/total for v in p]
        best=max(range(len(p)),key=p.__getitem__)
        out={'kind':kind,'effective_kind':kind,'options':opts,'probabilities':p,'choice_index':best,
             'choice':opts[best],'protocol':'jev27-bare-v1','model':str(self.directory.name),
             'adaptation':'native' if kind!='choice' or len(opts)<=16 else strategy,
             'num_model_requests':requests,'elapsed_seconds':time.monotonic()-start,
             'usage':{'prompt_tokens':prompt_tokens,'completion_tokens':0,'total_tokens':prompt_tokens}}
        if kind=='score':out['score']=sum(i*v for i,v in enumerate(p))
        return out

    def generate(self,messages,*,max_tokens=128,temperature=0.0,enable_thinking=False,reasoning_effort=None):
        if not isinstance(max_tokens,int) or not 1<=max_tokens<=self.context:
            raise ValueError('max_tokens is outside the configured context')
        if not isinstance(temperature,(int,float)) or not 0<=temperature<=2:
            raise ValueError('temperature must be between 0 and 2')
        plain,embeddings=[],[]
        for message in messages:
            content,media=self.state_parts(message['content'])
            plain.append({'role':message['role'],'content':content});embeddings.extend(media)
        kwargs={'enable_thinking':enable_thinking}
        if reasoning_effort is not None:kwargs['reasoning_effort']=reasoning_effort
        prompt=self.hf_tokenizer.apply_chat_template(plain,tokenize=False,add_generation_prompt=True,**kwargs)
        out=self.generate_prompt(prompt,embeddings,max_tokens,temperature)
        out['input_images']=len(embeddings)
        out['image_embedding_tokens']=[int(e.embeddings.shape[0]) for e in embeddings]
        out['image_preprocessing']=self.image_info(embeddings)
        return out

    def generate_prompt(self,prompt,embeddings,max_tokens,temperature=0.0,stop_ids=()):
        tokens=[];reason='length'
        with self.session(prompt,embeddings,reserve=max_tokens) as (logits,n,params):
            for i in range(max_tokens):
                logits=logits[:self.tokenizer.actual_vocab_size]
                if not bool(self.torch.isfinite(logits).all()):raise RuntimeError('Non-finite generation logits')
                token=int(logits.argmax()) if temperature==0 else int(self.torch.multinomial(
                    self.torch.softmax(logits/temperature,dim=-1),1))
                if token in self.eos_ids or token in stop_ids:reason='stop';break
                tokens.append(token)
                if i+1<max_tokens:
                    ids=self.torch.tensor([[token]],dtype=self.torch.long)
                    logits=self.model.forward(ids,params(n+i))[0,-1].float()
        text=self.tokenizer.decode(self.torch.tensor(tokens,dtype=self.torch.long),decode_special_tokens=True) if tokens else ''
        return {'text':text,'finish_reason':reason,'usage':{'prompt_tokens':n,'completion_tokens':len(tokens),
                                                          'total_tokens':n+len(tokens)}}

    def adaptive_decide(self,kind,state,question,options=None,*,strategy='single',thinking='off',
                        threshold=0.8,think_budget=1024,return_reasoning=False,debug=False,
                        reasoning_effort=None):
        """Confidence-gated System 2, using the official 50:50 distribution mix."""
        import math
        start=time.monotonic()
        if thinking=='default':thinking='off'
        if thinking not in ('off','auto','on'):raise ValueError('Invalid thinking mode')
        if kind=='score' and thinking!='off':raise ValueError('Thinking supports noul and choice')
        if not isinstance(threshold,(int,float)) or not math.isfinite(threshold) or not 0<=threshold<=1:
            raise ValueError('threshold must be between 0 and 1')
        if not isinstance(think_budget,int) or not 1<=think_budget<self.context:
            raise ValueError('think_budget must be positive and smaller than context')
        if reasoning_effort not in (None,'low','medium','xhigh'):raise ValueError('Invalid reasoning effort')
        result=self.decide(kind,state,question,options,strategy=strategy)
        if thinking=='off':return result
        p1=result['probabilities']
        info={'mode':thinking,'threshold':threshold,'budget':think_budget,'used':False}
        if thinking=='on' or max(p1)<threshold:
            text,embeddings=self.state_parts(state)
            shown=['Yes (true)','No (false)'] if kind=='noul' else result['options']
            # Contextual answer labels differ from bare-v1 option-line labels.
            import string
            prefix=self.tokenizer.encode('Answer: (').flatten().tolist()
            labels=[]
            for lab in list(string.ascii_uppercase)+[a+b for a in string.ascii_uppercase for b in string.ascii_uppercase]:
                ids=self.tokenizer.encode(f'Answer: ({lab})').flatten().tolist()
                if ids[:len(prefix)]==prefix and len(ids)==len(prefix)+2:labels.append((lab,ids[len(prefix)]))
                if len(labels)==len(shown):break
            if len(labels)!=len(shown):raise ValueError('Not enough contextual System 2 answer labels')
            body='\n'.join(f'({lab}) {opt}' for (lab,_),opt in zip(labels,shown))
            user=text+f'\n\nQuestion: {question}\n\nOptions:\n{body}\n\n'+(
                'Think it through carefully, then give your final answer on the last line in the form: Answer: (X)')
            kwargs={'enable_thinking':True}
            if reasoning_effort is not None:kwargs['reasoning_effort']=reasoning_effort
            prompt=self.hf_tokenizer.apply_chat_template([{'role':'user','content':user}],tokenize=False,
                                                        add_generation_prompt=True,**kwargs)
            end=self.tokenizer.encode('</think>',encode_special_tokens=True).flatten().tolist()
            if len(end)!=1:raise ValueError('Model has no single-token thinking delimiter')
            thought=self.generate_prompt(prompt,embeddings,think_budget,stop_ids=end)
            read_prompt=prompt+thought['text']+'</think>\n\nAnswer: ('
            with self.session(read_prompt,embeddings) as (logits,n,_):
                z=[logits[i].item() for _,i in labels];mx=max(z);e=[math.exp(v-mx) for v in z]
                total=sum(e);p2=[v/total for v in e]
            if kind=='noul':p2=[p2[1],p2[0]]
            mixed=[0.5*a+0.5*b for a,b in zip(p1,p2)]
            best=max(range(len(mixed)),key=mixed.__getitem__)
            result.update(probabilities=mixed,choice_index=best,choice=result['options'][best])
            usage=result['usage'];usage['prompt_tokens']+=thought['usage']['prompt_tokens']+n
            usage['completion_tokens']+=thought['usage']['completion_tokens']
            usage['total_tokens']=usage['prompt_tokens']+usage['completion_tokens']
            result['num_model_requests']+=2
            info.update(used=True,think_tokens=thought['usage']['completion_tokens'],
                        finished_within_budget=thought['finish_reason']=='stop',mix=0.5)
            if return_reasoning:info['reasoning']=thought['text'].strip()
            if debug:info.update(system1=p1,system2=p2)
        elif debug:info['system1']=p1
        result['thinking']=info
        result['elapsed_seconds']=time.monotonic()-start
        return result

    def close(self):
        if self.conversation is not None:self.close_conversation(self.conversation.id)
        if self.lora is not None:self.lora.unload()
        if self.vision_model is not None:self.vision_model.unload()
        self.model.unload()
