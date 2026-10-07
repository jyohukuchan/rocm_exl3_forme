"""Measure a real near-limit JEV vision prompt with an audited quantized KV cache."""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import resource
import time


R9700_UUID = 'GPU-a8e9ddefa2d60f55'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def near_limit_prompt(runtime, target_tokens, image_part):
    # Embed the current image once; repeated sizing does not recompute its vision tower.
    image_alias, embeddings = runtime.state_parts([image_part])
    marker = 'CAPACITY_FILLER_INSERT_731B'
    messages = [{'role':'user','content':
        'This is a context-capacity test. The following text is irrelevant padding.\n'
        +marker+'\nThe game screenshot follows:\n'+image_alias+
        '\nIgnore the padding. Reply with exactly OK.'}]
    template = runtime.hf_tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    if template.count(marker) != 1:
        raise ValueError('Capacity filler marker must appear once')
    prefix, suffix = template.split(marker)
    unit = next((word for word in [' x',' a',' the']
                 if runtime.tokenizer.encode(word).numel()==1),None)
    if unit is None:
        raise ValueError('No stable one-token padding unit')
    initial = runtime.tokenizer.encode(prefix+suffix,encode_special_tokens=True,embeddings=embeddings).numel()
    repeats = target_tokens-initial
    if repeats <= 0:
        raise ValueError('Target prompt is too short for the image and instructions')
    for _ in range(4):
        prompt = prefix+unit*repeats+suffix
        ids = runtime.tokenizer.encode(prompt,encode_special_tokens=True,embeddings=embeddings)
        count = ids.numel()
        if count == target_tokens:
            return prompt,embeddings,{'tokens':count,'padding_unit':unit,'padding_repeats':repeats,
                'text_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                'token_ids_sha256':hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()}
        repeats += target_tokens-count
    raise ValueError('Could not construct an exact-length context probe')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default='/models/safetensors/JEV-27B-VL-exl3-4bpw')
    parser.add_argument('--context',type=int,required=True)
    parser.add_argument('--cache-quant',default='5,4')
    parser.add_argument('--chunk-size',type=int,default=1024)
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if os.environ.get('ROCR_VISIBLE_DEVICES') != R9700_UUID:
        raise SystemExit('Run only in the existing R9700-only container')
    if args.output.exists():
        raise SystemExit('Use a new output directory to preserve failed attempts')
    args.output.mkdir(parents=True)
    soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    target=65536 if hard==resource.RLIM_INFINITY else min(65536,hard)
    resource.setrlimit(resource.RLIMIT_NOFILE,(max(soft,target),hard))
    from rocm_tools.jev_runtime import JEVRuntime,parse_cache_quant
    import torch
    if torch.cuda.device_count()!=1:
        raise SystemExit('Exactly one GPU must be visible')
    props=torch.cuda.get_device_properties(0)
    if 'R9700' not in props.name:
        raise SystemExit('Visible GPU must be the R9700')
    def memory():
        torch.cuda.synchronize()
        free,total=torch.cuda.mem_get_info()
        return {'allocated_bytes':torch.cuda.memory_allocated(),
                'reserved_bytes':torch.cuda.memory_reserved(),
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                'device_free_bytes':free,'device_total_bytes':total}
    sources=[Path(__file__),Path(__file__).with_name('jev_runtime.py')]
    result={'context':args.context,'cache_quant':list(parse_cache_quant(args.cache_quant)),
        'gpu':props.name,'gpu_uuid':R9700_UUID,'visible_devices':1,
        'torch_version':torch.__version__,'rocm_version':torch.version.hip,
        'chunk_size':args.chunk_size,'image_count':1,
        'open_file_limit':resource.getrlimit(resource.RLIMIT_NOFILE)[0],
        'image_sha256':hashlib.sha256(args.image.read_bytes()).hexdigest(),
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'status':'loading','memory_before_load':memory(),
        'notes':'One real game screenshot with an exactly sized synthetic text prefix. This checks execution/capacity, not long-context game-playing quality.'}
    started=time.monotonic();runtime=None
    def update(**values):
        result.update(values,elapsed_seconds=time.monotonic()-started)
        write_json(args.output/'result.json',result)
    update()
    try:
        runtime=JEVRuntime(args.model,context=args.context,chunk_size=args.chunk_size,
                           cache_quant=args.cache_quant,vision=True)
        cache=runtime.cache_info()
        if cache['observed_bits'] != [list(parse_cache_quant(args.cache_quant))]:
            raise ValueError('Actual cache widths do not match requested K/V bits')
        if set(cache['layer_classes']) != {'CacheLayer_quant'}:
            raise ValueError('The full-attention KV layers are not all quantized')
        update(status='short_smoke',cache=cache,model_max_context=runtime.config.max_position_embeddings,
               load_seconds=time.monotonic()-started,memory_after_load=memory())
        print(json.dumps({'phase':'loaded','context':args.context,'cache':cache,
                          'memory':result['memory_after_load']}),flush=True)
        image_part={'type':'image_url','image_url':{'url':'data:image/png;base64,'+
                                                    base64.b64encode(args.image.read_bytes()).decode()}}
        smoke=runtime.generate([{'role':'user','content':[
            {'type':'text','text':'Reply with exactly OK.'},image_part]}],max_tokens=16)
        decision=runtime.decide('noul','Two plus two equals four.','Is this statement correct?')
        update(short_generation=smoke,short_decision=decision,memory_after_smoke=memory())
        if not smoke['text'].strip() or decision['choice_index']!=1:
            raise ValueError('Short generation/native decision smoke check failed')
        target=args.context-64
        prompt,embeddings,prompt_info=near_limit_prompt(runtime,target,image_part)
        update(status='long_prefill',prompt=prompt_info,output_budget=32)
        original_prefill=runtime.model.prefill
        progress_started=time.monotonic();last_notice=0
        def observed_prefill(ids,params,*a,**kw):
            nonlocal last_notice
            output=original_prefill(ids,params,*a,**kw)
            count=params['past_len']+ids.shape[-1]
            if count-last_notice>=16384:
                last_notice=count
                print(json.dumps({'phase':'prefill','context':args.context,
                    'processed_tokens':count,'target_tokens':target,
                    'seconds':time.monotonic()-progress_started,'memory':memory()}),flush=True)
            return output
        runtime.model.prefill=observed_prefill
        torch.cuda.reset_peak_memory_stats()
        stage_started=time.monotonic()
        output=runtime.generate_prompt(prompt,embeddings,max_tokens=32)
        seconds=time.monotonic()-stage_started
        if output['usage']['prompt_tokens']!=target or not output['text'].strip():
            raise ValueError('Near-limit generation did not produce a non-empty answer')
        update(status='passed',long_generation=output,long_seconds=seconds,
               exact_ok=output['text'].strip()=='OK',memory_after_long=memory())
        print(json.dumps({'phase':'passed','context':args.context,'prompt_tokens':target,
                          'seconds':seconds,'output':output,'memory':result['memory_after_long']}),flush=True)
    except Exception as error:
        update(status='failed',error={'type':type(error).__name__,'message':str(error)})
        print(json.dumps({'phase':'failed','context':args.context,'error':result['error']}),flush=True)
    finally:
        if runtime is not None:
            runtime.close()
        update()
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
