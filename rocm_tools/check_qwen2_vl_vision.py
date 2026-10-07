"""Compare Qwen2-VL vision embeddings in separate EXL3/HF GPU processes.

Stop the owned game VLM server first. Each mode loads only a vision tower.
HF receives the saved EXL3 pixels and also checks its independent processor.
"""
import argparse
import json
import os
from pathlib import Path
import time

import torch
from PIL import Image


def assert_gpu_ownership():
    if os.environ.get('ROCR_VISIBLE_DEVICES') != 'GPU-a8e9ddefa2d60f55':
        raise RuntimeError('This probe requires the allocated R9700 only')
    for path in Path('/proc').iterdir():
        if not path.name.isdigit() or int(path.name) == os.getpid():
            continue
        try:
            argv = (path / 'cmdline').read_bytes().split(b'\0')
            state = (path / 'stat').read_text().split()[2]
        except OSError:
            continue
        if state != 'Z' and b'rocm_tools.jev_server' in argv:
            raise RuntimeError('Stop the owned VLM server before this reference probe')


def compare(actual, reference):
    # FP32 reduction over millions of entries can produce cosine > 1 on CPU.
    a, b = actual.double().flatten(), reference.double().flatten()
    if actual.shape != reference.shape:
        raise ValueError(f'Shape mismatch: {actual.shape}, {reference.shape}')
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Non-finite reference result')
    return {
        'shape': list(actual.shape),
        'mean_abs': (a-b).abs().mean().item(),
        'max_abs': (a-b).abs().max().item(),
        'relative_l2': ((a-b).norm()/b.norm().clamp_min(1e-12)).item(),
        'cosine': torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['exl3', 'hf'], required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert_gpu_ownership()
    args.output.mkdir(parents=True, exist_ok=True)
    image = Image.open(args.image).convert('RGB')
    cases = [('small', 50176), ('native', 524288)]
    outputs = {}
    if args.mode == 'exl3':
        from exllamav3 import Config, Model, Tokenizer
        config = Config.from_directory(args.checkpoint)
        tokenizer = Tokenizer.from_config(config)
        vision = Model.from_config(config, component='vision')
        vision.load(device='cuda:0', max_chunk_size=3072)
        stages = {}
        for index in (1, 8, 32):
            module = vision.modules[index]
            original = module.forward
            def capture(*a, _original=original, _key=index, **kw):
                result = _original(*a, **kw)
                stages[_key] = result.squeeze(0).cpu()
                return result
            module.forward = capture
        for name, pixels in cases:
            config.vision_pp.min_pixels = 3136
            config.vision_pp.max_pixels = pixels
            input_pixels, size, grid = vision.preprocess(image)
            stages.clear()
            start = time.monotonic()
            embedding = vision.get_image_embeddings(tokenizer, image)
            torch.cuda.synchronize()
            result = {'pixels': input_pixels, 'grid': grid, 'size': size,
                      'embedding': embedding.embeddings, 'stages': dict(stages)}
            torch.save(result, args.output / (name + '-exl3.pt'))
            outputs[name] = {'grid': grid, 'size': size,
                             'tokens': embedding.embeddings.shape[0],
                             'seconds': time.monotonic() - start}
    else:
        from safetensors import safe_open
        from transformers import AutoProcessor
        from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
        from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel
        config = Qwen2VLConfig.from_pretrained(args.checkpoint).vision_config
        config._attn_implementation = 'sdpa'
        # Rotary-frequency buffers are derived rather than stored in the checkpoint.
        # Construct on CPU so those buffers are real before moving the tower to GPU.
        vision = Qwen2VisionTransformerPretrainedModel(config)
        state = {}
        with safe_open(str(Path(args.checkpoint)/'model.safetensors'), framework='pt', device='cpu') as f:
            for key in f.keys():
                if key.startswith('visual.'):
                    state[key.removeprefix('visual.')] = f.get_tensor(key)
        vision.load_state_dict(state, strict=True, assign=True)
        del state
        vision = vision.eval().half().to('cuda')
        stages = {}
        for index in (0, 7, 31):
            vision.blocks[index].register_forward_hook(
                lambda module, inputs, result, key=index+1: stages.__setitem__(key, result.cpu()))
        for name, pixels in cases:
            saved = torch.load(args.output / (name + '-exl3.pt'), weights_only=True)
            processor = AutoProcessor.from_pretrained(args.checkpoint, min_pixels=3136, max_pixels=pixels)
            preprocessed = processor.image_processor(images=image, return_tensors='pt')
            grid = preprocessed['image_grid_thw']
            if tuple(grid[0].tolist()) != tuple(saved['grid']):
                raise ValueError('Independent preprocessing produced a different patch grid')
            pixel_metrics = compare(saved['pixels'], preprocessed['pixel_values'].half())
            stages.clear()
            start = time.monotonic()
            result = vision(saved['pixels'].to('cuda'), grid_thw=grid.to('cuda'))
            output = result.pooler_output.cpu()
            torch.cuda.synchronize()
            outputs[name] = {'preprocessing': pixel_metrics,
                             'embedding': compare(saved['embedding'], output),
                             'stages': {str(k): compare(saved['stages'][k], v) for k, v in stages.items()},
                             'seconds': time.monotonic() - start}
            torch.save({'embedding': output, 'stages': stages}, args.output / (name + '-hf.pt'))
    report = {'mode': args.mode, 'checkpoint': args.checkpoint, 'image': str(args.image),
              'weight_dtype': 'float16', 'device': torch.cuda.get_device_name(0),
              'cases': outputs}
    (args.output / (args.mode + '.json')).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
