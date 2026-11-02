"""Engram disk offload: checkpoint mapping and decode windows. CPU only."""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.kernels.ops.quantization.fp4_fake_quant import FP8_BLOCK_SIZE
from sglang.srt.layers.engram import _gather_table_rows, _unpack_fp4_rows
from sglang.srt.layers.engram_disk import (
    CheckpointTable,
    _fp4_to_fp8_lut,
    _safetensors_weight_map,
    cache_budget_bytes,
    reset_disk_cache,
)
from sglang.srt.layers.engram_staging import EngramRowStager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

LAYER_ID = 3
NUM_EMBEDDINGS = 64
DIM = 64
SCALE_DIM = DIM // FP8_BLOCK_SIZE


def _make_checkpoint(d: str, sharded: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """A fake checkpoint with layer 3's engram table; returns (weight, scale)."""
    weight = torch.randn(NUM_EMBEDDINGS, DIM).to(torch.float8_e4m3fn)
    scale = torch.randint(120, 140, (NUM_EMBEDDINGS, SCALE_DIM), dtype=torch.uint8)
    from safetensors.torch import save_file

    tensors = {
        f"layers.{LAYER_ID}.engram.embed.weight": weight,
        f"layers.{LAYER_ID}.engram.embed.scale": scale,
    }
    if sharded:
        # Two shards plus an index, the table split across them in name only.
        save_file({"other.weight": torch.zeros(4)}, os.path.join(d, "a.safetensors"))
        save_file(tensors, os.path.join(d, "b.safetensors"))
        index = {
            "weight_map": {
                "other.weight": "a.safetensors",
                **{k: "b.safetensors" for k in tensors},
            }
        }
        with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f)
    else:
        save_file(tensors, os.path.join(d, "model.safetensors"))
    return weight, scale


class TestCheckpointTable(CustomTestCase):
    def setUp(self):
        super().setUp()
        reset_disk_cache()

    def _check_mapping(self, d, weight, scale):
        table = CheckpointTable(LAYER_ID, NUM_EMBEDDINGS, DIM)
        self.assertFalse(table.ready)
        table.bind_shard(0, NUM_EMBEDDINGS, 1)
        table.map(d)
        self.assertTrue(table.ready)
        self.assertFalse(table.packed)
        ids = torch.tensor([0, 3, NUM_EMBEDDINGS - 1, 3])
        got_w, got_s = table.gather(ids)
        self.assertTrue(torch.equal(got_w, weight.view(torch.uint8)[ids]))
        self.assertTrue(torch.equal(got_s, scale[ids]))
        # The second gather is served from the hot-row cache, not the file.
        reads = {"n": 0}
        real_w, real_s = table.weight.read_sorted, table.scale.read_sorted

        def _count(real):
            def wrapped(*args, **kwargs):
                reads["n"] += 1
                return real(*args, **kwargs)

            return wrapped

        table.weight.read_sorted = _count(real_w)
        table.scale.read_sorted = _count(real_s)
        got_w2, got_s2 = table.gather(ids)
        self.assertEqual(reads["n"], 0)
        self.assertTrue(torch.equal(got_w2, got_w))
        self.assertTrue(torch.equal(got_s2, got_s))

    def test_unowned_rows_are_zero(self):
        with tempfile.TemporaryDirectory() as d:
            weight, scale = _make_checkpoint(d, sharded=False)
            table = CheckpointTable(LAYER_ID, NUM_EMBEDDINGS, DIM)
            table.bind_shard(0, 32, 2)
            table.map(d)
            got_w, got_s = table.gather(torch.tensor([1, 40]))
            self.assertTrue(torch.equal(got_w[0], weight.view(torch.uint8)[1]))
            self.assertTrue(torch.equal(got_s[0], scale[1]))
            self.assertTrue(torch.equal(got_w[1], torch.zeros_like(got_w[1])))
            self.assertTrue(torch.equal(got_s[1], torch.zeros_like(got_s[1])))

    def test_map_single_shard(self):
        with tempfile.TemporaryDirectory() as d:
            weight, scale = _make_checkpoint(d, sharded=False)
            self._check_mapping(d, weight, scale)

    def test_map_sharded_with_index(self):
        with tempfile.TemporaryDirectory() as d:
            weight, scale = _make_checkpoint(d, sharded=True)
            self._check_mapping(d, weight, scale)

    def test_missing_table_raises(self):
        with tempfile.TemporaryDirectory() as d:
            from safetensors.torch import save_file

            save_file({"other.weight": torch.zeros(4)}, os.path.join(d, "m.safetensors"))
            table = CheckpointTable(LAYER_ID, NUM_EMBEDDINGS, DIM)
            with self.assertRaises(ValueError):
                table.map(d)

    def test_no_safetensors_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                _safetensors_weight_map(d)


class TestPackedCheckpointTable(CustomTestCase):
    def test_map_packed_weight(self):
        with tempfile.TemporaryDirectory() as d:
            from safetensors.torch import save_file

            # MXFP4 layout: dim=64 packed into 32 U8 bytes, e8m0 scale per 32.
            weight = torch.randint(0, 256, (NUM_EMBEDDINGS, DIM // 2), dtype=torch.uint8)
            scale = torch.randint(120, 140, (NUM_EMBEDDINGS, SCALE_DIM), dtype=torch.uint8)
            save_file(
                {
                    f"layers.{LAYER_ID}.engram.embed.weight": weight,
                    f"layers.{LAYER_ID}.engram.embed.scale": scale,
                },
                os.path.join(d, "model.safetensors"),
            )
            table = CheckpointTable(LAYER_ID, NUM_EMBEDDINGS, DIM)
            table.map(d)
            self.assertTrue(table.packed)
            got_w, got_s = table.gather(torch.arange(NUM_EMBEDDINGS))
            self.assertTrue(torch.equal(got_w, weight))
            self.assertTrue(torch.equal(got_s, scale))

    def test_unpack_fp4_rows_is_lossless(self):
        # Every e2m1 code, both nibbles: the unpacked fp8 bytes must dequant to
        # the exact e2m1 values.
        e2m1 = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0]
            + [-0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
        )
        codes = torch.arange(16, dtype=torch.uint8)
        packed = (codes[0::2] & 0xF) | ((codes[1::2] & 0xF) << 4)
        packed = packed.unsqueeze(0)  # [1, 8] -> 16 values
        unpacked = _unpack_fp4_rows(packed)
        self.assertEqual(unpacked.shape, (1, 16))
        got = unpacked.view(torch.float8_e4m3fn).float().flatten()
        self.assertTrue(torch.equal(got, e2m1))
        # The LUT itself is the e4m3 encoding of the e2m1 value set.
        lut = _fp4_to_fp8_lut()
        self.assertTrue(torch.equal(lut.view(torch.float8_e4m3fn).float().abs()[1:8], e2m1[1:8]))


def _dequant_rows(weight_u8: torch.Tensor, scale_u8: torch.Tensor) -> torch.Tensor:
    """CPU twin of the in-memory Engram lookup: fp8 rows times e8m0 block scales."""
    weight = weight_u8.view(torch.float8_e4m3fn).float()
    scale = scale_u8.view(torch.float8_e8m0fnu).float()
    blocked = weight.unflatten(-1, (-1, FP8_BLOCK_SIZE))
    return (blocked * scale.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)


class TestDiskConfig(CustomTestCase):
    def test_disk_fields_and_peer_read_rejection(self):
        from sglang.srt.arg_groups.engram_config import EngramConfig

        config = EngramConfig.from_dict({"placement": "disk"})
        self.assertEqual(config.dir, "")
        self.assertEqual(config.cache_mib, 49152)
        self.assertEqual(config.lookup, "auto")
        explicit = EngramConfig.from_dict(
            {"placement": "disk", "dir": "/ckpt", "cache_mib": 0}
        )
        self.assertEqual(explicit.dir, "/ckpt")
        self.assertEqual(explicit.cache_mib, 0)
        with self.assertRaises(ValueError):
            EngramConfig.from_dict({"placement": "disk", "lookup": "peer_read"})
        with self.assertRaises(ValueError):
            EngramConfig.from_dict({"placement": "disk", "lookup": "all_to_all"})


class TestCacheBudget(CustomTestCase):
    def test_default_cap_is_48_gib_across_tp(self):
        from sglang.srt.arg_groups.engram_config import EngramConfig

        self.assertEqual(EngramConfig().cache_mib, 49152)
        self.assertEqual(cache_budget_bytes(4), (49152 << 20) // 4)

    def test_budget_splits_across_tp(self):
        self.assertIsNone(cache_budget_bytes(4, 0))
        self.assertEqual(cache_budget_bytes(4, 4096), (4096 << 20) // 4)
        self.assertEqual(cache_budget_bytes(1, 200_000), 200_000 << 20)
        self.assertEqual(cache_budget_bytes(4, 200_000) * 4, 200_000 << 20)


class TestDiskMatchesMemory(CustomTestCase):
    def test_gathered_rows_dequant_like_the_in_memory_table(self):
        with tempfile.TemporaryDirectory() as d:
            weight, scale = _make_checkpoint(d, sharded=False)
            table = CheckpointTable(LAYER_ID, NUM_EMBEDDINGS, DIM)
            table.bind_shard(0, NUM_EMBEDDINGS, 1)
            table.map(d)
            ids = torch.tensor([0, 5, 5, NUM_EMBEDDINGS - 1])
            got_w, got_s = table.gather(ids)
            disk = _dequant_rows(got_w, got_s)
            memory = _dequant_rows(weight.view(torch.uint8)[ids], scale[ids])
            self.assertTrue(torch.equal(disk, memory))


class TestDiskOffloadPolicy(CustomTestCase):
    def test_speculative_decoding_is_rejected(self):
        from sglang.srt.layers.engram import (
            engram_disk_supported_load_format,
            reject_engram_disk_speculative,
        )

        reject_engram_disk_speculative(None)
        with self.assertRaises(ValueError):
            reject_engram_disk_speculative("DSPARK")
        self.assertTrue(engram_disk_supported_load_format("auto"))
        self.assertTrue(engram_disk_supported_load_format("safetensors"))
        self.assertTrue(engram_disk_supported_load_format("fastsafetensors"))
        self.assertTrue(engram_disk_supported_load_format("LoadFormat.RUNAI_STREAMER"))
        self.assertFalse(engram_disk_supported_load_format("pt"))
        self.assertFalse(engram_disk_supported_load_format("dummy"))

    def test_overlap_is_disabled_and_memory_flags_stay(self):
        from sglang.srt.arg_groups.overrides import (
            _engram_disk_offload_disables_overlap,
        )

        view = SimpleNamespace(
            mem_fraction_static=0.8,
            max_running_requests=32,
            cuda_graph_max_bs_decode=64,
            engram_config=None,
        )
        self.assertEqual(_engram_disk_offload_disables_overlap(view), {})
        view.engram_config = {"placement": "disk", "cache_mib": 49152}
        out = _engram_disk_offload_disables_overlap(view)
        self.assertEqual(out, {"disable_overlap_schedule": True})
        view.engram_config = {"placement": "gpu", "lookup": "peer_read"}
        self.assertEqual(_engram_disk_offload_disables_overlap(view), {})
        self.assertEqual(view.mem_fraction_static, 0.8)
        self.assertEqual(view.max_running_requests, 32)
        self.assertEqual(view.cuda_graph_max_bs_decode, 64)

    def test_disable_mmap_does_not_materialize_engram_tensors(self):
        from safetensors.torch import save_file

        import sglang.srt.model_loader.weight_utils as weight_utils

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.safetensors")
            other = torch.arange(8, dtype=torch.float32)
            save_file(
                {
                    "other.weight": other,
                    f"layers.{LAYER_ID}.engram.embed.weight": torch.zeros(
                        4, dtype=torch.uint8
                    ),
                },
                path,
            )
            with patch(
                "sglang.srt.layers.engram_disk.engram_disk_enabled", return_value=True
            ):
                with patch.object(
                    weight_utils.safetensors.torch,
                    "load",
                    side_effect=AssertionError("full-shard load"),
                ):
                    got = dict(weight_utils._iter_safetensors_tensors(path, True))
            self.assertEqual(list(got), ["other.weight"])
            self.assertTrue(torch.equal(got["other.weight"], other))


class TestGatherTableRows(CustomTestCase):
    def test_gather_table_rows(self):
        table = torch.arange(64 * 8, dtype=torch.uint8).view(64, 8)
        ids = torch.tensor([3, 0, 63, 3])
        rows = _gather_table_rows(table, ids)
        if torch.cuda.is_available():
            # Without an accelerator the pinned allocation falls back to
            # pageable memory with a warning.
            self.assertTrue(rows.is_pinned())
        self.assertTrue(torch.equal(rows, table[ids]))
        # Both the single- and multi-chunk paths gather the same rows.
        many = torch.randint(0, 64, (9000,))
        self.assertTrue(torch.equal(_gather_table_rows(table, many), table[many]))


def _make_stager(max_ngram_size: int) -> EngramRowStager:
    """A stager shell with a trigram hasher, for host-window tests."""
    from sglang.srt.layers.engram import EngramHasher

    hasher = EngramHasher.__new__(EngramHasher)
    nn.Module.__init__(hasher)
    hasher.max_ngram_size = max_ngram_size
    stager = EngramRowStager.__new__(EngramRowStager)
    stager.hasher = hasher
    return stager


class TestDecodeWindows(CustomTestCase):
    def test_decode_windows_layout(self):
        stager = _make_stager(max_ngram_size=3)
        reqs = [
            # Column s must be the s-th predecessor of the last fill token.
            type("R", (), {"full_untruncated_fill_ids": [10, 11, 12, 13]})(),
            # A short request left-pads with zeros and blocks the look-back.
            type("R", (), {"full_untruncated_fill_ids": [20]})(),
        ]
        batch = type("B", (), {"reqs": reqs})()
        tokens, blocked = stager._decode_windows(batch)
        self.assertEqual(tokens.tolist(), [[13, 12, 11], [20, 0, 0]])
        self.assertEqual(blocked.tolist(), [[False, False, False], [False, True, True]])


if __name__ == "__main__":
    unittest.main()
