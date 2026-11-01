from types import SimpleNamespace

import pytest

from exllamav3.generator.generator import Generator
from exllamav3.model.model import Model
import exllamav3.model.model as model_module


@pytest.mark.parametrize('capacity,requested,expected', [
    (8192, None, 4096), (2048, None, 2048), (2048, 4096, 4096),
])
def test_load_chunk_default_fits_attached_cache_without_changing_explicit_request(monkeypatch, capacity, requested, expected):
    model = Model.__new__(Model)
    cache = SimpleNamespace(max_num_tokens=capacity, initialized=False)
    # SimpleNamespace cannot be weak-referenced, so a callable models a live weakref.
    model.cache_weakrefs = {1: lambda: cache}
    model.config = SimpleNamespace(infer_params=SimpleNamespace())
    model.modules = [SimpleNamespace(device='cpu')]
    monkeypatch.setattr(model_module, 'free_mem', lambda: None)
    monkeypatch.setattr(model_module.g_tensor_cache, 'drop_all', lambda: None)
    monkeypatch.setattr(model_module.torch.cuda, 'device_count', lambda: 1)
    captured = []
    def load(*args):
        captured.append(args[4])
        return iter(())
    monkeypatch.setattr(model, '_load_autosplit', load)
    list(model.load_gen(max_chunk_size=requested))
    assert captured == [expected]
    assert cache.initialized


def test_generator_chunk_default():
    import inspect
    assert inspect.signature(Generator).parameters['max_chunk_size'].default == 4096
