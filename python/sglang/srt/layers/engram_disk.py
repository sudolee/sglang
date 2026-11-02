"""NVMe-backed DeepSeek-V4.1 Engram tables, read in-place from safetensors.

Unlike a host mmap of the ~97 GiB tables (which fills the page cache, one 4 KiB
page per random 136-byte row) this path:

* parses only the safetensors header for (file, offset, shape)
* gathers the current batch's rows with ``os.pread``
* coalesces consecutive ids so random hash lookups become sequential ranges
* keeps a compact hot-row cache (packed bytes, not 4 KiB pages).
  ``SGLANG_DSV41_ENGRAM_DISK_CACHE_MIB`` defaults to 49152 MiB across all TP
  ranks. Set it to 0 to keep every row this rank reads; that grows until
  allocation fails and does not stop at free RAM
* each rank reads and caches only its own row shard; the forward all-reduce
  reassembles the row
* issues ``posix_fadvise(DONTNEED)`` on the bytes just read so the page cache
  does not retain a second copy beside the compact cache

Enabled by ``SGLANG_ENABLE_DSV41_ENGRAM_DISK=1`` and/or by pointing
``SGLANG_DSV41_ENGRAM_DISK_DIR`` at a local copy of the checkpoint (empty DIR
falls back to ``--model-path``).
"""

from __future__ import annotations

import glob
import json
import logging
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import torch

from sglang.kernels.ops.quantization.fp4_fake_quant import FP8_BLOCK_SIZE
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def _fp4_to_fp8_lut() -> torch.Tensor:
    """Nibble value -> fp8 e4m3 byte. Every e2m1 value is exact in e4m3."""
    values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    signed = torch.cat([values, -values])
    return signed.to(torch.float8_e4m3fn).view(torch.uint8)


_FP4_LUT = None


def unpack_fp4_rows(packed: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """[n, dim//2] packed MXFP4 bytes -> [n, dim] fp8 e4m3 bytes."""
    global _FP4_LUT
    if _FP4_LUT is None:
        _FP4_LUT = _fp4_to_fp8_lut()
    lut = _FP4_LUT
    n, w = packed.shape
    if out is None:
        out = torch.empty(n, w * 2, dtype=torch.uint8, pin_memory=packed.is_pinned())
    out[:, 0::2] = lut[(packed & 0xF).long()]
    out[:, 1::2] = lut[(packed >> 4).long()]
    return out

_ENGRAM_WEIGHT_KEYS = (
    "layers.{}.engram.embed.weight",
    "model.layers.{}.engram.embed.weight",
)
_ENGRAM_SCALE_KEYS = (
    "layers.{}.engram.embed.scale",
    "layers.{}.engram.embed.weight_scale",
    "model.layers.{}.engram.embed.scale",
    "model.layers.{}.engram.embed.weight_scale",
)
_SAFETENSORS_BYTE_DTYPES = ("F8_E4M3", "F8_E8M0", "U8")
_GATHER_THREADS = 16
_FADV_RANDOM = getattr(os, "POSIX_FADV_RANDOM", 1)
_FADV_DONTNEED = getattr(os, "POSIX_FADV_DONTNEED", 4)
_FADV_NOREUSE = getattr(os, "POSIX_FADV_NOREUSE", 5)
_WAYS = 4

_pool: Optional[ThreadPoolExecutor] = None


def cache_budget_bytes(tp_size: int) -> Optional[int]:
    """Per-rank hot-row cache budget, or None when the cache is explicitly uncapped.

    Read only while disk offload is on. ``SGLANG_DSV41_ENGRAM_DISK_CACHE_MIB``
    is the total across TP ranks, in MiB. The default is 49152. ``0`` means no
    cap: the cache grows with every distinct row until allocation fails. A
    positive value is split across ranks and used as given.
    """
    mib = int(envs.SGLANG_DSV41_ENGRAM_DISK_CACHE_MIB.get() or 0)
    if mib <= 0:
        return None
    return (mib << 20) // max(int(tp_size), 1)


_cache_policy_logged = False


def _log_cache_policy(tp_size: int, budget: Optional[int]) -> None:
    """One startup line. The cap is not inferred from free RAM."""
    global _cache_policy_logged
    if _cache_policy_logged:
        return
    _cache_policy_logged = True
    ranks = max(int(tp_size), 1)
    if budget is None:
        logger.warning(
            "Engram disk cache is uncapped (SGLANG_DSV41_ENGRAM_DISK_CACHE_MIB=0). "
            "This rank keeps every distinct row it reads and grows until "
            "allocation fails. That is not a cap at free RAM. Set a positive "
            "MiB value to cap the total across TP ranks (default 49152)."
        )
        return
    logger.warning(
        "Engram disk cache cap: %d MiB total across TP ranks, %.2f GiB on this "
        "rank (tp=%d). Set SGLANG_DSV41_ENGRAM_DISK_CACHE_MIB=0 to disable the cap.",
        int(envs.SGLANG_DSV41_ENGRAM_DISK_CACHE_MIB.get() or 0),
        budget / 2**30,
        ranks,
    )


def _row_keys(salt: int, layer: int, rows: np.ndarray) -> np.ndarray:
    """Cache key. The low 32 bits hold the row, so 0 stays free for empty slots."""
    return (
        (np.uint64(salt) << np.uint64(48))
        | (np.uint64(layer + 1) << np.uint64(32))
        | (rows.astype(np.uint64) + np.uint64(1))
    )


def engram_disk_enabled() -> bool:
    """On/off switch: explicit flag, or a non-empty disk directory."""
    try:
        if envs.SGLANG_ENABLE_DSV41_ENGRAM_DISK.get():
            return True
    except AttributeError:
        pass
    return bool(envs.SGLANG_DSV41_ENGRAM_DISK_DIR.get())


def _pool_get() -> ThreadPoolExecutor:
    global _pool
    if _pool is None:
        _pool = ThreadPoolExecutor(_GATHER_THREADS, thread_name_prefix="engram-disk")
    return _pool


def _fadvise(fd: int, offset: int, length: int, advice: int) -> None:
    posix_fadvise = getattr(os, "posix_fadvise", None)
    if posix_fadvise is None:
        return
    try:
        posix_fadvise(fd, offset, length, advice)
    except (OSError, OverflowError):
        pass


def _safetensors_weight_map(checkpoint_dir: str) -> dict[str, str]:
    index = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            return json.load(f)["weight_map"]
    files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
    if len(files) != 1:
        raise ValueError(
            "engram disk offload needs a safetensors checkpoint in "
            f"{checkpoint_dir} (an index file or exactly one shard)"
        )
    with open(files[0], "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        return dict.fromkeys(json.loads(f.read(header_len)), os.path.basename(files[0]))


def _safetensors_entry(path: str, name: str) -> tuple[int, list[int], str]:
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        info = json.loads(f.read(header_len))[name]
    return 8 + header_len + info["data_offsets"][0], info["shape"], info["dtype"]


def _pread_coalesced(
    fd: int, base: int, width: int, sorted_ids: np.ndarray, out: np.ndarray
) -> None:
    """Read sorted unique row ids into ``out`` [n, width], coalescing runs."""
    n = int(sorted_ids.shape[0])
    i = 0
    while i < n:
        j = i + 1
        while j < n and int(sorted_ids[j]) == int(sorted_ids[j - 1]) + 1:
            j += 1
        start = int(sorted_ids[i])
        count = j - i
        nbytes = count * width
        off = base + start * width
        buf = os.pread(fd, nbytes, off)
        if len(buf) != nbytes:
            raise OSError(
                f"engram disk short read: got {len(buf)} of {nbytes} bytes "
                f"at offset {off} (row {start})"
            )
        out[i:j] = np.frombuffer(buf, dtype=np.uint8).reshape(count, width)
        _fadvise(fd, off, nbytes, _FADV_DONTNEED)
        i = j


class _RowCache:
    """4-way set-associative cache of packed rows, private to this process.

    One cache is shared by every Engram layer. The key mixes a per-table salt
    so two layers never alias a slot. A hit copies ~136 bytes; a miss is the
    caller's pread, then an insert that overwrites one way.
    """

    def __init__(self, nslots: int, weight_w: int, scale_w: int):
        nslots = max(int(nslots), _WAYS)
        self.nsets = max(nslots // _WAYS, 1)
        self.n = self.nsets * _WAYS
        self.weight_w = weight_w
        self.scale_w = scale_w
        self.keys = np.zeros(self.n, dtype=np.uint64)
        self.weight = np.zeros((self.n, weight_w), dtype=np.uint8)
        self.scale = np.zeros((self.n, scale_w), dtype=np.uint8)
        # Next way to overwrite when a set is full. Without this, every miss
        # lands on way 0 and the other three ways freeze on their first fill.
        self.victim = np.zeros(self.nsets, dtype=np.int64)
        nbytes = self.keys.nbytes + self.weight.nbytes + self.scale.nbytes
        logger.info(
            "engram disk cache: %d slots (%.2f GiB this rank)",
            self.n,
            nbytes / 2**30,
        )

    def slots_for(self, salt: int, layer: int, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        key = _row_keys(salt, layer, rows)
        mixed = key * np.uint64(11400714819323198485)
        set_id = (mixed % np.uint64(self.nsets)).astype(np.int64)
        ways = set_id[:, None] * _WAYS + np.arange(_WAYS)
        return key, ways

    def lookup(
        self, salt: int, layer: int, rows: np.ndarray, out_w: np.ndarray, out_s: np.ndarray
    ) -> np.ndarray:
        """Fill ``out_*`` (aligned with ``rows``) on hit. Returns the miss mask."""
        key, ways = self.slots_for(salt, layer, rows)
        stored = self.keys[ways]
        hit_way = stored == key[:, None]
        hit = hit_way.any(axis=1)
        if hit.any():
            which = np.argmax(hit_way, axis=1)
            slot = ways[np.arange(rows.shape[0]), which]
            out_w[hit] = self.weight[slot[hit]]
            out_s[hit] = self.scale[slot[hit]]
        return ~hit

    def insert(
        self, salt: int, layer: int, rows: np.ndarray, weight: np.ndarray, scale: np.ndarray
    ) -> None:
        key, ways = self.slots_for(salt, layer, rows)
        stored = self.keys[ways]
        empty = stored == 0
        has_empty = empty.any(axis=1)
        set_id = ways[:, 0] // _WAYS
        which = np.where(has_empty, np.argmax(empty, axis=1), self.victim[set_id])
        slot = ways[np.arange(rows.shape[0]), which]
        self.keys[slot] = key
        self.weight[slot] = weight
        self.scale[slot] = scale
        self.victim[set_id] = (which + 1) % _WAYS


class _UnboundedRowCache:
    """Grows with the rows this process actually reads. Nothing is evicted."""

    def __init__(self, weight_w: int, scale_w: int):
        self.weight_w = weight_w
        self.scale_w = scale_w
        self._n = 0
        self._cap = 0
        self._index: dict[int, int] = {}
        self.weight = np.empty((0, weight_w), dtype=np.uint8)
        self.scale = np.empty((0, scale_w), dtype=np.uint8)
        logger.info("engram disk cache: no size cap on this rank")

    def _grow(self, extra: int) -> None:
        need = self._n + extra
        if need <= self._cap:
            return
        cap = max(need, max(self._cap * 2, 1024))
        weight = np.empty((cap, self.weight_w), dtype=np.uint8)
        scale = np.empty((cap, self.scale_w), dtype=np.uint8)
        if self._n:
            weight[: self._n] = self.weight[: self._n]
            scale[: self._n] = self.scale[: self._n]
        self.weight = weight
        self.scale = scale
        self._cap = cap

    def lookup(
        self, salt: int, layer: int, rows: np.ndarray, out_w: np.ndarray, out_s: np.ndarray
    ) -> np.ndarray:
        keys = _row_keys(salt, layer, rows)
        index = self._index
        slots = np.full(keys.shape[0], -1, dtype=np.int64)
        for i, key in enumerate(keys.tolist()):
            slot = index.get(key)
            if slot is not None:
                slots[i] = slot
        hit = slots >= 0
        if hit.any():
            out_w[hit] = self.weight[slots[hit]]
            out_s[hit] = self.scale[slots[hit]]
        return ~hit

    def insert(
        self, salt: int, layer: int, rows: np.ndarray, weight: np.ndarray, scale: np.ndarray
    ) -> None:
        keys = _row_keys(salt, layer, rows)
        index = self._index
        fresh = [i for i, key in enumerate(keys.tolist()) if key not in index]
        if not fresh:
            return
        self._grow(len(fresh))
        start = self._n
        picked = np.asarray(fresh, dtype=np.int64)
        self.weight[start : start + len(fresh)] = weight[picked]
        self.scale[start : start + len(fresh)] = scale[picked]
        for offset, row in enumerate(fresh):
            index[int(keys[row])] = start + offset
        self._n += len(fresh)


_process_cache: Optional[object] = None


def reset_disk_cache() -> None:
    """Drop the process-local hot-row cache. Tests use this between tables."""
    global _process_cache, _cache_policy_logged
    _process_cache = None
    _cache_policy_logged = False


def _get_cache(tp_size: int, num_embeddings: int, weight_w: int, scale_w: int):
    global _process_cache
    if _process_cache is not None:
        if _process_cache.weight_w != weight_w or _process_cache.scale_w != scale_w:
            return None
        return _process_cache
    budget = cache_budget_bytes(tp_size)
    _log_cache_policy(tp_size, budget)
    if budget is None:
        _process_cache = _UnboundedRowCache(weight_w, scale_w)
        return _process_cache
    # key (8) + packed row + one victim int64 shared by the four ways.
    slot = 8 + weight_w + scale_w + 8 // _WAYS
    nslots = min(max(budget // slot, 1), max(int(num_embeddings), 1))
    _process_cache = _RowCache(nslots, weight_w, scale_w)
    return _process_cache


class _DiskSlice:
    """uint8 [nrows, width] tensor living at a byte offset in a checkpoint file."""

    def __init__(self, fd: int, offset: int, nrows: int, width: int, path: str):
        self.fd = fd
        self.offset = offset
        self.nrows = nrows
        self.width = width
        self.path = path
        _fadvise(fd, offset, nrows * width, _FADV_RANDOM)
        _fadvise(fd, offset, nrows * width, _FADV_NOREUSE)

    def read_sorted(self, sorted_ids: np.ndarray, out: np.ndarray) -> None:
        """Fill ``out`` [n, width] from already-sorted unique row ids."""
        if sorted_ids.shape[0] == 0:
            return
        chunks = min(_GATHER_THREADS, max(1, int(sorted_ids.shape[0]) // 1024))
        if chunks <= 1:
            _pread_coalesced(self.fd, self.offset, self.width, sorted_ids, out)
            return
        step = -(-int(sorted_ids.shape[0]) // chunks)
        futs = [
            _pool_get().submit(
                _pread_coalesced,
                self.fd,
                self.offset,
                self.width,
                sorted_ids[i : i + step],
                out[i : i + step],
            )
            for i in range(0, int(sorted_ids.shape[0]), step)
        ]
        for fut in futs:
            fut.result()


class CheckpointTable:
    """One Engram layer's table, opened on the original safetensors shards.

    GPU and pinned host memory only ever see the current batch's rows. The OS
    page cache is told not to keep the rest.
    """

    def __init__(self, layer_id: int, num_embeddings: int, dim: int):
        self.layer_id = layer_id
        self.num_embeddings = num_embeddings
        self.dim = dim
        self.packed = False
        self.weight: Optional[_DiskSlice] = None
        self.scale: Optional[_DiskSlice] = None
        self._fds: list[int] = []
        # Full-table by default; EngramEmbedding.bind's shard before map so each
        # rank reads and caches only the rows it owns.
        self.row_start = 0
        self.shard_rows = num_embeddings
        self.tp_size = 1
        self._salt = int(id(self) & 0xFFFF)
        self._cache = None

    @property
    def ready(self) -> bool:
        return self.weight is not None and self.scale is not None

    def bind_shard(self, row_start: int, nrows: int, tp_size: int) -> None:
        """This rank's [row_start, row_start + nrows) of the full table."""
        self.row_start = int(row_start)
        self.shard_rows = int(nrows)
        self.tp_size = max(int(tp_size), 1)

    def map(self, checkpoint_dir: str) -> None:
        if self.ready:
            return
        weight_map = _safetensors_weight_map(checkpoint_dir)
        weight_key = self._find_key(weight_map, _ENGRAM_WEIGHT_KEYS)
        scale_key = self._find_key(weight_map, _ENGRAM_SCALE_KEYS)
        if weight_key is None or scale_key is None:
            raise ValueError(
                f"engram disk offload: layer {self.layer_id}'s table or scales "
                f"are missing from the checkpoint in {checkpoint_dir}"
            )
        opened: dict[str, int] = {}
        self.weight = self._open_slice(
            checkpoint_dir, weight_map, weight_key, self.dim, opened, allow_packed=True
        )
        self.scale = self._open_slice(
            checkpoint_dir,
            weight_map,
            scale_key,
            self.dim // FP8_BLOCK_SIZE,
            opened,
        )
        self._fds = list(opened.values())
        self._cache = None
        if self.shard_rows > 0 and self.weight is not None and self.scale is not None:
            self._cache = _get_cache(
                self.tp_size,
                self.num_embeddings,
                self.weight.width,
                self.scale.width,
            )
        footprint = self.num_embeddings * (
            (self.dim // 2 if self.packed else self.dim) + self.dim // FP8_BLOCK_SIZE
        )
        logger.info(
            "engram disk table layer %s: %d rows x %d (%s) via pread from %s, "
            "%.2f GiB stay on disk (page cache DONTNEED)",
            self.layer_id,
            self.num_embeddings,
            self.dim,
            "mxfp4-packed, unpacked to fp8 on gather" if self.packed else "fp8",
            checkpoint_dir,
            footprint / 2**30,
        )

    def gather(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pinned uint8 rows aligned with ``ids``.

        Ids outside this rank's shard are zeros (the forward all-reduce fills
        them from the owner). Owned ids hit the hot-row cache or, on a miss,
        one coalesced ``pread``.
        """
        assert self.ready and self.weight is not None and self.scale is not None, (
            "engram disk table is mapped in finish_load"
        )
        if ids.device.type != "cpu":
            ids = ids.detach().cpu()
        ids_np = ids.contiguous().view(-1).to(torch.int64).numpy()
        n = int(ids_np.shape[0])
        ww, sw = self.weight.width, self.scale.width
        pin = torch.cuda.is_available()
        out_w = torch.zeros((n, ww), dtype=torch.uint8, pin_memory=pin)
        out_s = torch.zeros((n, sw), dtype=torch.uint8, pin_memory=pin)
        if n == 0 or self.shard_rows <= 0:
            return out_w, out_s
        owned = (ids_np >= self.row_start) & (ids_np < self.row_start + self.shard_rows)
        if not owned.any():
            return out_w, out_s
        idx = np.flatnonzero(owned)
        rows = ids_np[idx]
        w_np = out_w.numpy()
        s_np = out_s.numpy()
        owned_w = w_np[idx]
        owned_s = s_np[idx]
        cache = self._cache
        if cache is None:
            miss = np.ones(rows.shape[0], dtype=bool)
        else:
            miss = cache.lookup(self._salt, self.layer_id, rows, owned_w, owned_s)
        if miss.any():
            miss_rows = rows[miss]
            uniq, inverse = np.unique(miss_rows, return_inverse=True)
            tmp_w = np.empty((uniq.shape[0], ww), dtype=np.uint8)
            tmp_s = np.empty((uniq.shape[0], sw), dtype=np.uint8)
            self.weight.read_sorted(uniq, tmp_w)
            self.scale.read_sorted(uniq, tmp_s)
            if cache is not None:
                cache.insert(self._salt, self.layer_id, uniq, tmp_w, tmp_s)
            owned_w[miss] = tmp_w[inverse]
            owned_s[miss] = tmp_s[inverse]
        # Integer-array indexing returned a copy; write the owned rows back.
        w_np[idx] = owned_w
        s_np[idx] = owned_s
        return out_w, out_s

    def close(self) -> None:
        for fd in self._fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds = []
        self.weight = self.scale = None

    def _find_key(
        self, weight_map: dict[str, str], candidates: tuple[str, ...]
    ) -> Optional[str]:
        for pattern in candidates:
            key = pattern.format(self.layer_id)
            if key in weight_map:
                return key
        return None

    def _open_slice(
        self,
        checkpoint_dir: str,
        weight_map: dict[str, str],
        key: str,
        width: int,
        opened: dict[str, int],
        allow_packed: bool = False,
    ) -> _DiskSlice:
        path = os.path.join(checkpoint_dir, weight_map[key])
        offset, shape, dtype = _safetensors_entry(path, key)
        packed = (
            allow_packed
            and dtype == "U8"
            and len(shape) == 2
            and shape[0] == self.num_embeddings
            and shape[1] == width // 2
        )
        if packed:
            self.packed = True
            width = width // 2
        elif (
            dtype not in _SAFETENSORS_BYTE_DTYPES
            or len(shape) != 2
            or shape[0] != self.num_embeddings
            or shape[1] != width
        ):
            raise ValueError(
                f"engram disk offload: {key} is {dtype}{shape}; expected "
                f"one-byte elements in a [{self.num_embeddings}, {width}] table"
                + (
                    f" or an MXFP4-packed [{self.num_embeddings}, {width // 2}] one"
                    if allow_packed
                    else ""
                )
            )
        if path not in opened:
            fd = os.open(path, os.O_RDONLY)
            # Whole-file hint: hash lookups are random, and the compact cache
            # is the copy we want to keep. Length 0 means the entire file.
            _fadvise(fd, 0, 0, _FADV_RANDOM)
            _fadvise(fd, 0, 0, _FADV_NOREUSE)
            opened[path] = fd
        return _DiskSlice(opened[path], offset, shape[0], width, path)
