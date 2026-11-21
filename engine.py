#!/usr/bin/env python3
"""Lazy-loaded transcription engine for faster-whisper (NVIDIA CUDA / CPU + VAD + Ensemble)."""

from __future__ import annotations

import ctypes
import gc
import glob
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "Trascrizioni"
OUTPUT_DIR.mkdir(exist_ok=True)

MEDICAL_INITIAL_PROMPT = (
    "Lezione universitaria di medicina, chirurgia e radiologia diagnostica. "
    "Terminologia medica, anatomia, fisiopatologia, patologia, clinica, semeiotica, "
    "farmacologia, TC, RM, risonanza magnetica, tomografia computerizzata, ecografia, "
    "radiografia, PET, scintigrafia, mezzo di contrasto, parenchima, lesione, "
    "neoplasia, metastasi, ischemia, stenosi, ipertensione, diagnosi differenziale."
)

MODEL_OPTIONS: list[tuple[str, str]] = [
    ("large-v3", "large-v3 (Massima precisione - Predefinito)"),
    ("ensemble", "Combinato: large-v3 + turbo (Doppio controllo Ensemble)"),
    ("large-v3-turbo", "large-v3-turbo (Ultra veloce)"),
]

LANGUAGE_OPTIONS: list[tuple[str, str]] = [
    ("auto", "Automatica (IT / EN)"),
    ("it", "Italiano"),
    ("en", "Inglese"),
]

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".mp3",
        ".wav",
        ".m4a",
        ".mp4",
        ".mkv",
        ".ogg",
        ".flac",
        ".webm",
        ".aac",
        ".opus",
        ".mov",
        ".avi",
        ".wma",
        ".mpeg",
        ".mpg",
    }
)

# Lazy backend state — never loaded synchronously on module import.
_backend_lock = threading.Lock()
_backend_done = threading.Event()
_backend_started = False
_whisper_model_cls: Any = None
_backend_error: Exception | None = None


def _preload_cuda_libs() -> None:
    """Preload bundled cuBLAS and cuDNN shared libraries from the project .venv."""
    venv_lib = BASE_DIR / ".venv" / "lib"
    for pattern in (
        "*/site-packages/nvidia/cublas/lib/lib*.so*",
        "*/site-packages/nvidia/cudnn/lib/lib*.so*",
    ):
        for so_path in sorted(glob.glob(str(venv_lib / pattern))):
            try:
                ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


def _load_backend_impl() -> None:
    global _whisper_model_cls, _backend_error
    try:
        _preload_cuda_libs()
        from faster_whisper import WhisperModel

        _whisper_model_cls = WhisperModel
        _backend_error = None
    except Exception as exc:
        _backend_error = exc
    finally:
        _backend_done.set()


def preload_backend_async(
    on_ready: Callable[[], None] | None = None,
    on_error: Callable[[Exception], None] | None = None,
) -> None:
    """Warm up CUDA shared libraries and faster_whisper in a background thread (0 VRAM)."""
    global _backend_started
    with _backend_lock:
        if _backend_started:
            already_done = _backend_done.is_set()
            err = _backend_error
        else:
            _backend_started = True
            already_done = False
            err = None

            def _runner() -> None:
                _load_backend_impl()
                if _backend_error is not None:
                    if on_error is not None:
                        on_error(_backend_error)
                elif on_ready is not None:
                    on_ready()

            threading.Thread(
                target=_runner, name="cuda-backend-preload", daemon=True
            ).start()
            return

    if already_done:
        if err is not None and on_error is not None:
            on_error(err)
        elif err is None and on_ready is not None:
            on_ready()


def ensure_backend_loaded() -> Any:
    """Block until CUDA libraries and WhisperModel are ready, then return WhisperModel."""
    if not _backend_done.is_set():
        preload_backend_async()
        _backend_done.wait()
    if _backend_error is not None:
        raise RuntimeError(
            f"Impossibile caricare faster-whisper/CUDA: {_backend_error}"
        ) from _backend_error
    return _whisper_model_cls


def format_srt(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_short(seconds: float) -> str:
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def load_whisper(model_name: str) -> tuple[Any, str]:
    """Load WhisperModel on CUDA float16, falling back to CPU int8 if CUDA fails."""
    whisper_cls = ensure_backend_loaded()
    try:
        return whisper_cls(model_name, device="cuda", compute_type="float16"), "cuda"
    except Exception:
        return whisper_cls(model_name, device="cpu", compute_type="int8"), "cpu"


def unload_whisper(model: Any | None) -> None:
    """Release WhisperModel weights from GPU VRAM and trim native heap memory."""
    if model is None:
        return
    try:
        model.model.unload_model()
    except Exception:
        pass
    del model
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def pick_best_text(seg_v3: Any, turbo_segs: list[dict[str, Any]]) -> str:
    v3_text = seg_v3.text.strip()
    if not turbo_segs or not v3_text:
        return v3_text
    mid = 0.5 * (seg_v3.start + seg_v3.end)
    best_match = None
    for tseg in turbo_segs:
        if tseg["start"] - 0.8 <= mid <= tseg["end"] + 0.8:
            best_match = tseg
            break
    if not best_match:
        return v3_text
    t_text = best_match["text"]
    len_ratio = len(t_text) / max(1, len(v3_text))
    if (
        0.75 <= len_ratio <= 1.35
        and best_match["avg_logprob"] > seg_v3.avg_logprob + 0.12
    ):
        return t_text
    return v3_text


class _CancelledError(Exception):
    """Internal signal raised when the user cancels a running transcription."""


@dataclass(frozen=True)
class TranscriptionJob:
    file_path: Path
    language: str = "auto"
    model_mode: str = "large-v3"
    use_medical_context: bool = True


class TranscriptionWorker:
    """Single-worker sequential FIFO queue with live updates, cancellation and GPU cleanup."""

    def __init__(self, on_update: Callable[[dict[str, Any]], None]) -> None:
        self._on_update = on_update
        self._lock = threading.Lock()
        self._queue: deque[TranscriptionJob] = deque()
        self._cancel_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._state: dict[str, Any] = {
            "status": "idle",
            "progress": 0.0,
            "message": "Pronto. Trascina uno o più file audio o video.",
            "filename": "",
            "language": "",
            "duration": 0.0,
            "elapsed": 0.0,
            "job_started_at": 0.0,
            "lines": [],
            "ts_lines": [],
            "txt_path": "",
            "srt_path": "",
            "queue_size": 0,
            "error": "",
        }

    def get_snapshot(self) -> dict[str, Any]:
        with self._lock:
            snap = dict(self._state)
            snap["lines"] = list(self._state["lines"])
            snap["ts_lines"] = list(self._state["ts_lines"])
            snap["queue_size"] = len(self._queue)
            return snap

    def is_busy(self) -> bool:
        with self._lock:
            return self._worker_thread is not None and self._worker_thread.is_alive()

    def enqueue(self, jobs: list[TranscriptionJob]) -> int:
        valid_jobs = [job for job in jobs if job.file_path.is_file()]
        if not valid_jobs:
            return 0

        with self._lock:
            self._queue.extend(valid_jobs)
            should_start = (
                self._worker_thread is None or not self._worker_thread.is_alive()
            )
            if should_start:
                self._cancel_event.clear()
                self._worker_thread = threading.Thread(
                    target=self._run_loop, name="whisper-worker", daemon=True
                )
                self._worker_thread.start()
            else:
                self._state["queue_size"] = len(self._queue)
                snap = self._snapshot_locked()
                self._emit_unlocked(snap)
        return len(valid_jobs)

    def cancel(self) -> None:
        with self._lock:
            self._cancel_event.set()
            self._queue.clear()
            self._state["queue_size"] = 0
            if self._state["status"] in ("loading", "transcribing"):
                self._state["message"] = "Annullamento in corso e rilascio GPU..."
                snap = self._snapshot_locked()
            else:
                snap = None
        if snap is not None:
            self._emit_unlocked(snap)

    def shutdown(self, timeout: float = 5.0) -> None:
        self.cancel()
        with self._lock:
            worker = self._worker_thread
        if worker is not None and worker.is_alive():
            worker.join(timeout=timeout)

    def _snapshot_locked(self) -> dict[str, Any]:
        snap = dict(self._state)
        snap["lines"] = list(self._state["lines"])
        snap["ts_lines"] = list(self._state["ts_lines"])
        snap["queue_size"] = len(self._queue)
        return snap

    def _update_state(self, **changes: Any) -> None:
        with self._lock:
            self._state.update(changes)
            snap = self._snapshot_locked()
        self._emit_unlocked(snap)

    def _emit_unlocked(self, snap: dict[str, Any]) -> None:
        try:
            self._on_update(snap)
        except Exception:
            pass

    def _check_cancelled(self) -> None:
        if self._cancel_event.is_set():
            raise _CancelledError()

    def _run_loop(self) -> None:
        cached_model: Any = None
        cached_model_name: str | None = None
        cached_device: str = "cuda"

        try:
            while True:
                with self._lock:
                    if self._cancel_event.is_set() or not self._queue:
                        break
                    job = self._queue.popleft()
                    remaining = len(self._queue)

                cached_model, cached_model_name, cached_device = self._process_job(
                    job, remaining, cached_model, cached_model_name, cached_device
                )
        finally:
            unload_whisper(cached_model)
            cached_model = None
            cached_model_name = None
            emit_cancelled = False
            with self._lock:
                self._worker_thread = None
                if self._cancel_event.is_set() and self._state.get("status") != "cancelled":
                    emit_cancelled = True
            if emit_cancelled:
                self._update_state(
                    status="cancelled",
                    progress=0.0,
                    job_started_at=0.0,
                    message="Trascrizione annullata (GPU liberata).",
                )

    def _process_job(
        self,
        job: TranscriptionJob,
        remaining_after: int,
        cached_model: Any,
        cached_model_name: str | None,
        cached_device: str,
    ) -> tuple[Any, str | None, str]:
        t0 = time.time()
        orig_name = job.file_path.name
        model_mode = (
            job.model_mode
            if job.model_mode in ("large-v3", "ensemble", "large-v3-turbo")
            else "large-v3"
        )
        is_ensemble = model_mode == "ensemble"
        transient_model: Any = None

        try:
            self._check_cancelled()
            self._update_state(
                status="loading",
                progress=0.0,
                filename=orig_name,
                language="",
                duration=0.0,
                elapsed=0.0,
                job_started_at=t0,
                lines=[],
                ts_lines=[],
                txt_path="",
                srt_path="",
                error="",
                message=f"Preparazione modello per {orig_name}...",
            )

            ensure_backend_loaded()
            self._check_cancelled()

            kwargs: dict[str, Any] = {
                "beam_size": 5,
                "vad_filter": True,
                "vad_parameters": {"min_silence_duration_ms": 500},
            }
            if job.use_medical_context:
                kwargs["initial_prompt"] = MEDICAL_INITIAL_PROMPT
            if job.language and job.language != "auto":
                kwargs["language"] = job.language

            turbo_segs: list[dict[str, Any]] = []

            if is_ensemble:
                # Ensemble alternates models to keep peak GPU VRAM usage low.
                if cached_model is not None:
                    unload_whisper(cached_model)
                    cached_model = None
                    cached_model_name = None

                self._update_state(
                    status="loading",
                    message=f"Passaggio 1/2: caricamento large-v3-turbo per {orig_name}...",
                )
                transient_model, dev1 = load_whisper("large-v3-turbo")
                self._check_cancelled()

                dev_label = "GPU CUDA" if dev1 == "cuda" else "CPU"
                self._update_state(
                    status="transcribing",
                    message=f"Passaggio 1/2 (large-v3-turbo • {dev_label}): {orig_name}",
                )

                segs_1, info_1 = transient_model.transcribe(str(job.file_path), **kwargs)
                self._update_state(
                    language=f"{info_1.language} ({info_1.language_probability * 100:.0f}%)",
                    duration=float(info_1.duration or 0.0),
                )

                for seg in segs_1:
                    self._check_cancelled()
                    txt = seg.text.strip()
                    if txt:
                        turbo_segs.append(
                            {
                                "start": seg.start,
                                "end": seg.end,
                                "text": txt,
                                "avg_logprob": seg.avg_logprob,
                            }
                        )
                    pct = (
                        min(38.0, (seg.end / info_1.duration * 38.0))
                        if info_1.duration > 0
                        else 0.0
                    )
                    self._update_state(
                        progress=round(pct, 1),
                        elapsed=round(time.time() - t0, 1),
                    )

                unload_whisper(transient_model)
                transient_model = None

            target_model = (
                "large-v3" if model_mode in ("large-v3", "ensemble") else "large-v3-turbo"
            )

            if cached_model is not None and cached_model_name == target_model:
                model = cached_model
                device_used = cached_device
            else:
                if cached_model is not None:
                    unload_whisper(cached_model)
                    cached_model = None
                    cached_model_name = None

                self._update_state(
                    status="loading",
                    message=(
                        f"Passaggio 2/2: caricamento {target_model} (raffinamento)..."
                        if is_ensemble
                        else f"Caricamento {target_model}..."
                    ),
                )
                model, device_used = load_whisper(target_model)
                cached_model = model
                cached_model_name = target_model
                cached_device = device_used

            self._check_cancelled()
            dev_label = "GPU CUDA" if device_used == "cuda" else "CPU"
            self._update_state(
                status="transcribing",
                message=(
                    f"Passaggio 2/2 ({target_model} + Ensemble • {dev_label}): {orig_name}"
                    if is_ensemble
                    else f"Trascrizione ({target_model} • {dev_label}): {orig_name}"
                ),
            )

            segments, info = model.transcribe(str(job.file_path), **kwargs)
            self._update_state(
                language=f"{info.language} ({info.language_probability * 100:.0f}%)",
                duration=float(info.duration or 0.0),
            )

            txt_paragraphs: list[str] = []
            ts_paragraphs: list[str] = []
            current_texts: list[str] = []
            current_start: float | None = None
            last_end: float | None = None
            srt_blocks: list[str] = []

            for idx, seg in enumerate(segments, start=1):
                self._check_cancelled()
                text = (
                    pick_best_text(seg, turbo_segs)
                    if is_ensemble
                    else seg.text.strip()
                )
                if not text:
                    continue

                if current_start is None:
                    current_start = seg.start

                gap = (seg.start - last_end) if last_end is not None else 0.0
                joined_len = sum(len(t) for t in current_texts) + len(current_texts)
                prev_ends_sentence = bool(
                    current_texts and current_texts[-1][-1:] in ".!?"
                )

                if current_texts and (
                    gap >= 1.5 or (joined_len >= 420 and prev_ends_sentence)
                ):
                    para = " ".join(current_texts)
                    txt_paragraphs.append(para)
                    ts_paragraphs.append(f"({format_short(current_start)}) {para}")
                    current_texts = [text]
                    current_start = seg.start
                else:
                    current_texts.append(text)

                last_end = seg.end
                srt_blocks.append(
                    f"{idx}\n{format_srt(seg.start)} --> {format_srt(seg.end)}\n{text}\n"
                )

                if info.duration > 0:
                    raw_pct = seg.end / info.duration
                    pct = (38.0 + raw_pct * 61.0) if is_ensemble else (raw_pct * 99.0)
                    pct = min(99.0, pct)
                else:
                    pct = 0.0

                live_clean = txt_paragraphs + (
                    [" ".join(current_texts)] if current_texts else []
                )
                live_ts = ts_paragraphs + (
                    [f"({format_short(current_start)}) {' '.join(current_texts)}"]
                    if current_texts
                    else []
                )
                self._update_state(
                    lines=live_clean,
                    ts_lines=live_ts,
                    progress=round(pct, 1),
                    elapsed=round(time.time() - t0, 1),
                )

            self._check_cancelled()

            if current_texts and current_start is not None:
                para = " ".join(current_texts)
                txt_paragraphs.append(para)
                ts_paragraphs.append(f"({format_short(current_start)}) {para}")

            OUTPUT_DIR.mkdir(exist_ok=True)
            stem = job.file_path.stem
            out_txt = OUTPUT_DIR / f"{stem}.txt"
            out_srt = OUTPUT_DIR / f"{stem}.srt"
            txt_content = ("\n\n".join(txt_paragraphs) + "\n") if txt_paragraphs else ""
            out_txt.write_text(txt_content, encoding="utf-8")
            out_srt.write_text("\n".join(srt_blocks), encoding="utf-8")

            with self._lock:
                has_more = len(self._queue) > 0 and not self._cancel_event.is_set()

            if not has_more:
                unload_whisper(cached_model)
                cached_model = None
                cached_model_name = None

            done_msg = (
                f"Completato {orig_name} — passaggio al file successivo..."
                if has_more
                else f"Completato! Salvato in Trascrizioni/{out_txt.name} (GPU liberata)"
            )
            self._update_state(
                status="done",
                progress=100.0,
                elapsed=round(time.time() - t0, 1),
                job_started_at=0.0,
                message=done_msg,
                lines=txt_paragraphs,
                ts_lines=ts_paragraphs,
                txt_path=str(out_txt),
                srt_path=str(out_srt),
            )
            return cached_model, cached_model_name, cached_device

        except _CancelledError:
            unload_whisper(transient_model)
            unload_whisper(cached_model)
            self._update_state(
                status="cancelled",
                progress=0.0,
                elapsed=round(time.time() - t0, 1),
                job_started_at=0.0,
                message="Trascrizione annullata (GPU liberata).",
            )
            return None, None, cached_device

        except Exception as exc:
            unload_whisper(transient_model)
            unload_whisper(cached_model)
            self._update_state(
                status="error",
                elapsed=round(time.time() - t0, 1),
                job_started_at=0.0,
                error=str(exc),
                message=f"Errore: {exc}",
            )
            return None, None, cached_device
