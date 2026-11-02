from sglang.srt.entrypoints.warmup import _warmup_registry
from sglang.srt.models.deepseek_v4 import dsv4_alloc_decode_batch_sizes


def test_alloc_decode_batch_sizes_cover_the_padded_upper_bound():
    assert dsv4_alloc_decode_batch_sizes(8) == [1, 2, 4, 8]
    assert dsv4_alloc_decode_batch_sizes(6) == [1, 2, 4, 8]
    assert dsv4_alloc_decode_batch_sizes(1) == [1]


def test_dsv4_device_kernels_is_a_named_warmup():
    assert "dsv4_device_kernels" in _warmup_registry
