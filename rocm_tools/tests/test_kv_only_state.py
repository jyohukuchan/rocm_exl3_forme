from collections import deque
import pytest
import torch
from exllamav3.cache.cache import Cache
from exllamav3.cache.recurrent_util import advance_recurrent_states


def empty_cache(recurrent_layers=None):
    cache=object.__new__(Cache)
    cache.free_list=deque([0]);cache.recurrent_state_cls=None
    cache.recurrent_layers=recurrent_layers or {}
    return cache


def test_kv_only_session_advances_position_and_releases_its_slot_once():
    cache=empty_cache()
    state=Cache.get_new_state(cache)
    advance_recurrent_states(torch.zeros(1,5,dtype=torch.long),{'recurrent_states':[state]},None)
    assert state.position==5 and not cache.free_list
    state.free();state.free()
    assert list(cache.free_list)==[0]


def test_missing_recurrent_state_class_does_not_silently_use_kv_only_state():
    cache=empty_cache({0:object()})
    with pytest.raises(ValueError,match='state class'):Cache.get_new_state(cache)
    assert list(cache.free_list)==[0]


def test_kv_only_capacity_probe_can_allocate_state_at_a_nonzero_position():
    cache=empty_cache()
    state=cache.get_test_state(13)
    assert state.position==13 and not cache.free_list
    state.free()
    assert list(cache.free_list)==[0]
