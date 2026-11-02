"""Scheduler-side staging of DeepSeek-V4.1 engram rows from a CPU-resident table
(the shared host table, or the disk table's checkpoint mapping).

Without ATS the forward's gather of random host rows is slow (_sorted_host_reads),
and a disk table cannot be gathered by the forward's kernels at all. The scheduler
knows a batch's tokens before its forward, so it hashes them on the CPU, gathers
each layer's rows into pinned memory, and copies them to the GPU while the
previous forward runs; the layer only dequantizes them. Extend batches stage into
fresh buffers on a side stream; decode batches stage into static buffers the
decode CUDA graphs captured, on the fenced schedule stream.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from sglang.kernels.ops.quantization.fp4_fake_quant import FP8_BLOCK_SIZE
from sglang.srt.environ import envs
from sglang.srt.layers.engram import (
    Engram,
    EngramEmbedding,
    EngramHasher,
    _unpack_fp4_rows,
)
from sglang.srt.model_executor.encoder_swa_replay import replay_len
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch

# Per engram layer: device fp8 rows, device e8m0 scales, and the event of their
# copy (None for the static decode buffers, which the schedule-stream fence
# has already ordered before the forward).
StagedRows = list[tuple[torch.Tensor, torch.Tensor, Optional[torch.cuda.Event]]]

_GATHER_THREADS = 16


class EngramRowStager:
    def __init__(
        self,
        hasher: EngramHasher,
        embeds: list[EngramEmbedding],
        device: torch.device,
        *,
        extend: bool = True,
        decode_bs: int = 0,
    ):
        self.hasher = hasher
        # fp8 and e8m0 have no CPU index_select; the gather moves bytes. A
        # packed (MXFP4) disk table gathers packed rows and the stage unpacks
        # them to fp8.
        self.tables = []
        self.disk = []
        for e in embeds:
            if e.disk_table is not None:
                self.tables.append(None)
                self.disk.append(e.disk_table)
            else:
                self.tables.append(
                    (
                        e.weight.data.view(torch.uint8),
                        e.scale.data.view(torch.uint8),
                        False,
                    )
                )
                self.disk.append(None)
        self.extend = extend
        self.device = device
        self.pool = ThreadPoolExecutor(_GATHER_THREADS)
        # The schedule stream waits on the running forward (the WAR barrier), so
        # copies there would not overlap it; each layer waits on its copy's event.
        self.copy_stream = torch.cuda.Stream(device)
        self.n_hash_cols = hasher.offsets.shape[1]
        self._decode_event: Optional[torch.cuda.Event] = None
        # Static decode buffers (disk tables): fixed addresses keep the staged
        # dequant capturable in the decode CUDA graphs, which read them baked.
        self._decode_buffers: Optional[
            list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = None
        if decode_bs:
            n = decode_bs * self.n_hash_cols
            self._decode_buffers = [
                (
                    torch.empty(n, e.dim, dtype=torch.uint8, device=device),
                    torch.empty(
                        n, e.dim // FP8_BLOCK_SIZE, dtype=torch.uint8, device=device
                    ),
                    torch.empty(n, e.dim, dtype=torch.uint8, pin_memory=True),
                    torch.empty(
                        n, e.dim // FP8_BLOCK_SIZE, dtype=torch.uint8, pin_memory=True
                    ),
                )
                for e in embeds
            ]

    @classmethod
    def from_model(
        cls,
        model: torch.nn.Module,
        hasher: EngramHasher,
        device: torch.device,
        max_running_requests: int = 0,
    ) -> Optional[EngramRowStager]:
        """A stager when every engram table is gatherable on the host."""
        engrams = sorted(
            (m for m in model.modules() if isinstance(m, Engram)),
            key=lambda m: m.layer_hash_index,
        )
        embeds = [m.embed for m in engrams]
        if not embeds or not all(e.cpu_stageable for e in embeds):
            return None
        any_disk = any(e.disk_table is not None for e in embeds)
        extend = (
            # Prefill CP hashes the whole prompt and keeps this rank's rows.
            get_parallel().attn_cp_size == 1
            # Disk tables always stage extends: the gather overlaps the running
            # forward instead of blocking it on storage reads.
            and (envs.SGLANG_OPT_DSV41_ENGRAM_HOST_STAGING.get() or any_disk)
        )
        # Disk tables also stage decodes into static buffers, which is what
        # keeps them out of the decode CUDA graphs' captured work.
        decode_bs = max_running_requests if any_disk else 0
        if not extend and not decode_bs:
            return None
        for e in embeds:
            e.enable_scheduler_staging()
        return cls(hasher, embeds, device, extend=extend, decode_bs=decode_bs)

    def stage(self, batch: ScheduleBatch) -> Optional[StagedRows]:
        """Stage every engram layer's rows for this batch, None to leave the
        lookup to the forward."""
        if batch.forward_mode.is_extend_without_speculative():
            if not self.extend:
                return None
            windows = self._windows(batch)
            if windows is None:
                return None
            ids, _ = self.hasher.hash_windows(*windows)
            staged = []
            for layer, table in enumerate(self.tables):
                layer_ids = ids[:, layer].reshape(-1)
                disk = self.disk[layer]
                if disk is not None:
                    rows, scales = disk.gather(layer_ids)
                    packed = disk.packed
                else:
                    weight, scale, packed = table
                    rows = self._gather(weight, layer_ids)
                    scales = self._gather(scale, layer_ids)
                if packed:
                    rows = _unpack_fp4_rows(rows)
                with torch.cuda.stream(self.copy_stream):
                    rows = rows.to(self.device, non_blocking=True)
                    scales = scales.to(self.device, non_blocking=True)
                    copied = torch.cuda.Event()
                    copied.record()
                staged.append((rows, scales, copied))
            return staged
        if batch.forward_mode.is_decode() and self._decode_buffers is not None:
            return self._stage_decode(batch)
        return None

    def static_staged_rows(self, bs: int) -> Optional[StagedRows]:
        """The decode staging buffers sliced to bs, for the decode CUDA graphs
        to capture; None when there are no disk tables."""
        if self._decode_buffers is None:
            return None
        n = bs * self.n_hash_cols
        assert n <= self._decode_buffers[0][0].shape[0], (
            f"decode capture bs {bs} exceeds the engram staging buffers; "
            "cap cuda_graph_max_bs at max_running_requests"
        )
        return [(w[:n], s[:n], None) for w, s, _, _ in self._decode_buffers]

    def _stage_decode(self, batch: ScheduleBatch) -> StagedRows:
        """Stage decode rows into the static buffers. Disk offload disables
        overlap scheduling, so the previous batch is fully processed: the
        sampled tokens are on the host, the GPU is between steps, and the H2D
        copy on the current stream is ordered before this batch's forward, so
        the dequant needs no event."""
        if self._decode_event is not None:
            # The pinned buffers are reused: the last H2D must have executed
            # before this gather overwrites them.
            self._decode_event.synchronize()
        ids, _ = self.hasher.hash_windows(*self._decode_windows(batch))
        staged = []
        for layer, (dev_w, dev_s, pin_w, pin_s) in enumerate(self._decode_buffers):
            layer_ids = ids[:, layer].reshape(-1)
            n = layer_ids.numel()
            disk = self.disk[layer]
            if disk is not None:
                packed_rows, scales = disk.gather(layer_ids)
                if disk.packed:
                    _unpack_fp4_rows(packed_rows, out=pin_w[:n])
                else:
                    pin_w[:n].copy_(packed_rows)
                pin_s[:n].copy_(scales)
            else:
                weight, scale, packed = self.tables[layer]
                if packed:
                    packed_rows = self._gather(weight, layer_ids)
                    _unpack_fp4_rows(packed_rows, out=pin_w[:n])
                else:
                    self._gather_into(weight, layer_ids, pin_w[:n])
                self._gather_into(scale, layer_ids, pin_s[:n])
            dev_w[:n].copy_(pin_w[:n], non_blocking=True)
            dev_s[:n].copy_(pin_s[:n], non_blocking=True)
            # Graph replay reads the whole captured buffer. Rows past this
            # batch must stay zero so the shard all-reduce does not sum stale
            # padding.
            if n < dev_w.shape[0]:
                dev_w[n:].zero_()
                dev_s[n:].zero_()
            staged.append((dev_w[:n], dev_s[:n], None))
        event = torch.cuda.Event()
        event.record()
        self._decode_event = event
        return staged

    def _decode_windows(
        self, batch: ScheduleBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One window per decode request: column s is the s-th predecessor of
        the last fill token, which this step feeds; blocked marks the look-back
        past the sequence start. The host twin of the windows EngramHasher
        builds on the device from its history."""
        n = self.hasher.max_ngram_size
        tokens, positions = [], []
        for req in batch.reqs:
            fill = req.full_untruncated_fill_ids
            assert fill, "decode request without fill ids"
            seg = np.asarray(fill[-n:], dtype=np.int64)
            if len(seg) < n:
                seg = np.concatenate([np.zeros(n - len(seg), np.int64), seg])
            tokens.append(seg[::-1])
            positions.append(len(fill) - 1)
        positions = torch.tensor(positions)
        blocked = positions.unsqueeze(-1) < torch.arange(n)
        return torch.from_numpy(np.stack(tokens)), blocked

    def _windows(
        self, batch: ScheduleBatch
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """The host twin of the windows EngramHasher builds on the device: tokens
        [T, n], column s the s-th predecessor, and blocked [T, n], the look-back
        past the sequence start. None unless every extend token is on the host."""
        max_ngram_size = self.hasher.max_ngram_size
        lookback = max_ngram_size - 1
        tokens, positions = [], []
        # The forward's rows: an encoder SWA replay hit extends from its replay
        # start (fold_encoder_swa_replay).
        resets = batch.encoder_swa_reset or []
        replay = [replay_len(p) if r else 0 for r, p in zip(resets, batch.prefix_lens)]
        replay += [0] * (len(batch.reqs) - len(replay))
        for req, prefix, extend, head in zip(
            batch.reqs, batch.prefix_lens, batch.extend_lens, replay, strict=True
        ):
            start, length = prefix - head, extend + head
            fill_ids = req.full_untruncated_fill_ids
            if start + length > len(fill_ids):
                # A decode row's newest token can still be on the device.
                return None
            if length == 0:
                continue
            lo = max(0, start - lookback)
            seg = np.asarray(fill_ids[lo : start + length], dtype=np.int64)
            seg = np.concatenate([np.zeros(lookback - (start - lo), np.int64), seg])
            tokens.append(
                np.lib.stride_tricks.sliding_window_view(seg, max_ngram_size)[:, ::-1]
            )
            positions.append(np.arange(start, start + length))
        if not tokens:
            return None
        positions = torch.from_numpy(np.concatenate(positions))
        blocked = positions.unsqueeze(-1) < torch.arange(max_ngram_size)
        return torch.from_numpy(np.concatenate(tokens)), blocked

    def _gather(self, table: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        out = torch.empty(
            (ids.numel(), table.shape[1]), dtype=table.dtype, pin_memory=True
        )
        self._gather_into(table, ids, out)
        return out

    def _gather_into(
        self, table: torch.Tensor, ids: torch.Tensor, out: torch.Tensor
    ) -> None:
        step = -(-ids.numel() // _GATHER_THREADS)
        chunks = [
            self.pool.submit(
                torch.index_select, table, 0, ids[i : i + step], out=out[i : i + step]
            )
            for i in range(0, ids.numel(), step)
        ]
        for chunk in chunks:
            chunk.result()
