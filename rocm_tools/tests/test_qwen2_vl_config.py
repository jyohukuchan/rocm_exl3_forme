from types import SimpleNamespace

from exllamav3.architecture.qwen2_vl import Qwen2VLConfig, Qwen2VLVisionModel
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig


def test_sparse_qwen2_vl_checkpoint_uses_hf_tower_defaults_not_language_width():
    checkpoint = {'hidden_size': 1536, 'in_chans': 3, 'model_type': 'qwen2_vl'}
    config = object.__new__(Qwen2VLConfig)
    config.hidden_size = 1536
    actual = config.read_vision_config(checkpoint)
    reference = Qwen2VLVisionConfig(**checkpoint)
    assert actual.hidden_size == reference.embed_dim == 1280
    assert actual.out_hidden_size == reference.hidden_size == 1536
    assert actual.intermediate_size == int(reference.embed_dim * reference.mlp_ratio)
    assert actual.hidden_act == reference.hidden_act
    assert actual.depth == reference.depth


def test_full_vision_attention_is_segmented_per_frame_in_unmerged_patch_units():
    vision = object.__new__(Qwen2VLVisionModel)
    vision.config = SimpleNamespace(vision=SimpleNamespace(spatial_merge_size=2))
    order, offsets = vision.image_attention_layout((2, 4, 6))
    assert order.tolist() == list(range(12))
    assert offsets == [0, 24, 48]
