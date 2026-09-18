"""Sidon speech restoration over speaker channels (GPU).

Ported from `fisher/sidon_restore/sidon_restore.py`, which restored 8 kHz
telephone audio. Two things changed to make it corpus-agnostic:

* **Any input rate.** Audio is resampled to Sidon's 16 kHz feature rate from
  whatever it arrives at, and the output length is matched to
  `round(n_in * out_sr / in_sr)`, so a restored file is sample-exact against its
  source and the supervisions stay valid unchanged.
* **The gate tapers to a floor, not to zero.** Log-energy features floor out on
  digital silence and a hard edge is a wideband click the aligner can read as a
  phone, so outside a channel's own speech the signal is attenuated to
  `gate_floor_db` through raised-cosine edges.

Signal path
    in -> 16 kHz -> 80-bin kaldi fbank (25/10 ms, povey) -> stride-2 stack
    -> w2v-BERT feature predictor -> DAC decoder (48 kHz) -> centre-stitched
    chunks -> out_sr -> length-match -> gate -> FLAC

Sidon's reference inference truncates to one chunk. Long-form audio is cut into
overlapping chunks; since the SSL encoder is context dependent the overlaps are
not coherent (correlation ~0.2 measured on Fisher), so each chunk keeps its
well-conditioned centre and neighbours are butt-joined with a short
equal-power fade placed at the quietest point of the input.

Needs torch + torchaudio + soundfile (the `sidon` env on Delta).
"""

from __future__ import annotations

import json
import math
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .sharding import shard_of

SSL_SR = 16_000          # feature extractor input rate
DEC_SR = 48_000          # DAC decoder output rate
FRAME_SHIFT = 160        # 10 ms @ 16 kHz
STRIDE = 2               # fbank frames stacked per SSL frame
HOP_16K = FRAME_SHIFT * STRIDE
UPSAMPLE = DEC_SR // SSL_SR


class SidonRestorer:
    def __init__(self, feature_extractor, decoder, device="cuda",
                 chunk_seconds=20.0, overlap_seconds=2.0, batch_size=8,
                 dtype="fp32", edge_trim_seconds=0.9, join_ms=15.0):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        self.feature_extractor = torch.jit.load(feature_extractor,
                                                map_location=self.device).eval()
        self.decoder = torch.jit.load(decoder, map_location=self.device).eval()
        self.autocast_dtype = {"fp32": None, "fp16": torch.float16,
                               "bf16": torch.bfloat16}[dtype]

        # Chunk and hop are whole SSL frames so decoder output stays on a 3x grid.
        self.chunk = int(round(chunk_seconds * SSL_SR / HOP_16K)) * HOP_16K
        overlap = int(round(overlap_seconds * SSL_SR / HOP_16K)) * HOP_16K
        overlap = min(overlap, self.chunk - HOP_16K)
        self.hop = self.chunk - overlap
        # fbank drops trailing frames: a chunk of C samples gives C/HOP - 1 frames.
        self.frames_per_chunk = self.chunk // HOP_16K - 1
        # The DAC conv stack drops a few edge samples, so measure the real length.
        self.out_per_chunk = self._probe_out_len()
        if self.out_per_chunk <= self.hop * UPSAMPLE:
            raise ValueError("overlap too small for the trailing-frame loss")
        self.batch_size = batch_size
        self._resamplers = {}

        self.hop_48 = self.hop * UPSAMPLE
        self.slack = self.out_per_chunk - self.hop_48
        self.join = 2 * max(1, int(round(join_ms * DEC_SR / 1000)) // 2)
        max_trim = (self.slack - self.join) // 2
        if max_trim < 0:
            raise ValueError(f"join of {join_ms} ms exceeds the overlap")
        self.trim = min(int(round(edge_trim_seconds * DEC_SR)), max_trim)
        t = torch.linspace(0.0, 1.0, self.join, device=self.device)
        self.fade_in = torch.sin(t * (math.pi / 2))
        self.fade_out = torch.cos(t * (math.pi / 2))

    # ------------------------------------------------------------ internals

    def _fbank(self, wav):
        import torchaudio
        return torchaudio.compliance.kaldi.fbank(
            waveform=wav.unsqueeze(0), sample_frequency=SSL_SR, num_mel_bins=80,
            frame_length=25, frame_shift=10, dither=0.0,
            preemphasis_coefficient=0.97, remove_dc_offset=True,
            window_type="povey", use_energy=False,
            energy_floor=1.192092955078125e-07)

    @staticmethod
    def _stack(feats):
        n = (feats.shape[0] // STRIDE) * STRIDE
        return feats[:n].reshape(n // STRIDE, feats.shape[1] * STRIDE)

    def _probe_out_len(self):
        torch = self.torch
        with torch.inference_mode():
            feats = torch.zeros(1, self.frames_per_chunk, 80 * STRIDE, device=self.device)
            hidden = self.feature_extractor(feats)["last_hidden_state"]
            return int(self.decoder(hidden.transpose(1, 2)).shape[-1])

    def _resample(self, wav, src, dst):
        if src == dst:
            return wav
        import torchaudio
        key = (src, dst)
        if key not in self._resamplers:
            self._resamplers[key] = torchaudio.transforms.Resample(
                src, dst, dtype=self.torch.float32).to(self.device)
        return self._resamplers[key](wav)

    def _run_model(self, feats):
        torch = self.torch
        if self.autocast_dtype is None:
            hidden = self.feature_extractor(feats)["last_hidden_state"]
            return self.decoder(hidden.transpose(1, 2)).squeeze(1)
        with torch.autocast("cuda", dtype=self.autocast_dtype):
            hidden = self.feature_extractor(feats)["last_hidden_state"]
            audio = self.decoder(hidden.transpose(1, 2))
        return audio.float().squeeze(1)

    def _join_points(self, wav16, starts):
        """48 kHz join index per chunk boundary, at the quietest input point
        inside the window where both neighbours are well conditioned."""
        torch = self.torch
        half = self.join // 2
        win16 = max(1, self.join // UPSAMPLE)
        step16 = max(1, win16 // 3)
        env = torch.nn.functional.avg_pool1d(
            (wav16 ** 2).view(1, 1, -1), kernel_size=win16, stride=step16,
            ceil_mode=True).view(-1)
        joins = []
        for i in range(len(starts) - 1):
            base = starts[i] * UPSAMPLE
            lo = base + self.hop_48 + self.trim + half
            hi = base + self.out_per_chunk - self.trim - half
            if hi <= lo:
                joins.append(lo)
                continue
            e_lo = max(0, (lo // UPSAMPLE - win16 // 2) // step16)
            e_hi = min(env.numel(), (hi // UPSAMPLE - win16 // 2) // step16 + 1)
            if e_hi <= e_lo:
                joins.append((lo + hi) // 2)
                continue
            e = e_lo + int(torch.argmin(env[e_lo:e_hi]).item())
            j = (e * step16 + win16 // 2) * UPSAMPLE
            joins.append(int(min(max(j, lo), hi)))
        return joins

    def _stitch(self, decoded, wav16, starts):
        torch = self.torch
        n = len(decoded)
        half = self.join // 2
        covered = starts[-1] * UPSAMPLE + self.out_per_chunk
        out = torch.zeros(covered, device=self.device)
        joins = self._join_points(wav16, starts)
        for i, audio in enumerate(decoded):
            base = starts[i] * UPSAMPLE
            lo = 0 if i == 0 else joins[i - 1] - half
            hi = covered if i == n - 1 else joins[i] + half
            seg = audio[lo - base: hi - base].clone()
            if i > 0:
                seg[: self.join] *= self.fade_in
            if i < n - 1:
                seg[-self.join:] *= self.fade_out
            out[lo:hi] += seg
        return out

    # ---------------------------------------------------------------- public

    def restore(self, wav, in_sr, out_sr=24_000):
        """Mono float tensor at `in_sr` -> mono `out_sr` tensor (on device) of
        exactly round(len * out_sr / in_sr) samples."""
        torch = self.torch
        with torch.inference_mode():
            n_in = wav.shape[-1]
            wav = self._resample(wav.to(self.device), in_sr, SSL_SR)
            n_16k = wav.shape[-1]

            n_chunks = max(1, -(-(max(n_16k - self.chunk, 0)) // self.hop) + 1)
            while (n_chunks - 1) * self.hop * UPSAMPLE + self.out_per_chunk < n_16k * UPSAMPLE:
                n_chunks += 1
            padded = (n_chunks - 1) * self.hop + self.chunk
            if padded > n_16k:
                wav = torch.nn.functional.pad(wav, (0, padded - n_16k))

            starts = [i * self.hop for i in range(n_chunks)]
            feats = torch.stack([self._stack(self._fbank(wav[s: s + self.chunk]))
                                 for s in starts])
            # Normalise with statistics pooled over the whole channel, so
            # silence-only chunks are not blown up to unit variance.
            flat = feats.reshape(-1, feats.shape[-1])
            feats = (feats - flat.mean(0, keepdim=True)) / torch.sqrt(
                flat.var(0, keepdim=True) + 1e-5)

            decoded = []
            for b in range(0, n_chunks, self.batch_size):
                audio = self._run_model(feats[b: b + self.batch_size])
                if audio.shape[-1] < self.out_per_chunk:
                    audio = torch.nn.functional.pad(
                        audio, (0, self.out_per_chunk - audio.shape[-1]))
                decoded.extend(audio[:, : self.out_per_chunk])

            out = self._resample(self._stitch(decoded, wav, starts), DEC_SR, out_sr)
            target = int(round(n_in * out_sr / in_sr))
            if out.shape[-1] < target:
                out = torch.nn.functional.pad(out, (0, target - out.shape[-1]))
            return out[:target]

    def gate(self, out, spans, sr, pad=0.1, fade_ms=20.0, floor_db=-60.0):
        """Attenuate everything outside `spans` [(start, end)] to `floor_db`.

        Returns (gated audio, fraction of samples kept at unity)."""
        torch = self.torch
        n = out.shape[-1]
        mask = torch.zeros(n, device=out.device)
        for s, e in spans:
            a, b = max(0, int(round((s - pad) * sr))), min(n, int(round((e + pad) * sr)))
            if b > a:
                mask[a:b] = 1.0
        kept = float(mask.mean().item())
        fade = int(round(fade_ms * sr / 1000))
        if fade > 0:
            k = torch.hann_window(2 * fade + 1, periodic=False, device=out.device)
            mask = torch.nn.functional.conv1d(
                mask.view(1, 1, -1), (k / k.sum()).view(1, 1, -1), padding=fade).view(-1)[:n]
        floor = 10.0 ** (floor_db / 20.0) if floor_db is not None else 0.0
        return out * (floor + (1.0 - floor) * mask), kept


def restore(source, shard=0, n_shards=1, limit=0, overwrite=False, device="cuda",
            report=None, log=print) -> dict:
    """Restore every channel of `source` whose recording falls in `shard`.

    The corpus supplies `restore_items()` -> (Channel, src, dst); a channel's
    `turns` are its own speech, which is what the gate keeps.
    """
    import soundfile as sf
    import torch

    cfg = dict(source.config.get("restore", {}))
    out_sr = int(cfg.get("out_sr", 24_000))
    restorer = SidonRestorer(
        cfg["feature_extractor"], cfg["decoder"], device=device,
        chunk_seconds=float(cfg.get("chunk_seconds", 20.0)),
        overlap_seconds=float(cfg.get("overlap_seconds", 2.0)),
        batch_size=int(cfg.get("batch_size", 8)),
        dtype=cfg.get("dtype", "fp32"),
        edge_trim_seconds=float(cfg.get("edge_trim_seconds", 0.9)),
        join_ms=float(cfg.get("join_ms", 15.0)))
    do_gate = bool(cfg.get("gate", True))
    log(f"[restore] {source.name} shard {shard}/{n_shards} out_sr={out_sr} "
        f"chunk={restorer.chunk / SSL_SR:.1f}s hop={restorer.hop / SSL_SR:.1f}s "
        f"gate={do_gate}", flush=True)

    def write(path: Path, audio, sr):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        sf.write(str(tmp), audio, sr, subtype="PCM_16", format="FLAC")
        os.replace(tmp, path)

    pool = ThreadPoolExecutor(max_workers=4)
    pending = []
    done = skipped = failures = 0
    audio_s = 0.0
    kept = []
    failed: list[str] = []
    t0 = time.perf_counter()
    try:
        for ch, src, dst in source.restore_items(limit=limit):
            if shard_of(ch.group_index, n_shards) != shard:
                continue
            if dst.exists() and not overwrite:
                skipped += 1
                continue
            try:
                data, sr = sf.read(str(src), dtype="float32", always_2d=False)
                if data.ndim > 1:
                    data = data.mean(axis=1)
                out = restorer.restore(torch.from_numpy(data), sr, out_sr)
                if do_gate:
                    out, k = restorer.gate(
                        out, [(s, e) for s, e, _ in ch.turns], out_sr,
                        pad=float(cfg.get("gate_pad", 0.1)),
                        fade_ms=float(cfg.get("gate_fade_ms", 20.0)),
                        floor_db=cfg.get("gate_floor_db", -60.0))
                    kept.append(k)
                # Sidon can overshoot full scale; clip rather than let PCM_16 wrap.
                out = out.clamp(-1.0, 1.0).cpu().numpy()
            except Exception:
                failures += 1
                failed.append(ch.file_id)
                log(f"[restore] ERROR {ch.file_id} ({src})", flush=True)
                traceback.print_exc()
                continue
            audio_s += len(data) / sr
            done += 1
            pending.append(pool.submit(write, dst, out, out_sr))
            pending = [f for f in pending if not f.done() or f.result() is None]
            el = time.perf_counter() - t0
            log(f"[restore] {done} {ch.file_id} {len(data) / sr / 60:.1f} min | "
                f"{audio_s / 3600:.2f} h in {el / 60:.1f} min "
                f"({audio_s / max(el, 1e-6):.0f}x RT)", flush=True)
    finally:
        pool.shutdown(wait=True)
        for f in pending:
            f.result()

    el = time.perf_counter() - t0
    stats = {
        "shard": shard, "n_shards": n_shards, "restored": done,
        "skipped_existing": skipped, "failures": failures, "failed": failed,
        "audio_hours": round(audio_s / 3600, 3), "wall_minutes": round(el / 60, 2),
        "realtime_factor": round(audio_s / max(el, 1e-6), 1),
        "gate_mean_kept": round(sum(kept) / len(kept), 4) if kept else None,
        "out_sr": out_sr,
    }
    if report:
        Path(report).write_text(json.dumps(stats, indent=2))
    return stats
