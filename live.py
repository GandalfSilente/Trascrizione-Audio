#!/usr/bin/env python3
"""Real-time microphone LIVE transcription (Nemotron 3.5 when NeMo is installed, faster-whisper CUDA fallback)."""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

import engine


def _find_audio_capture_cmd() -> list[str]:
    """Return a native Linux 16 kHz mono s16le stdout capture command."""
    if shutil.which("pw-record"):
        return [
            "pw-record",
            "--format=s16",
            "--rate=16000",
            "--channels=1",
            "-",
        ]
    if shutil.which("parec"):
        return [
            "parec",
            "--format=s16le",
            "--rate=16000",
            "--channels=1",
            "--raw",
        ]
    if shutil.which("arecord"):
        return [
            "arecord",
            "-f",
            "S16_LE",
            "-r",
            "16000",
            "-c",
            "1",
            "-t",
            "raw",
            "-q",
            "-",
        ]
    raise RuntimeError(
        "Nessun tool di cattura microfono trovato (pw-record, parec o arecord)."
    )


def _try_load_nemotron() -> Any | None:
    """Try loading NVIDIA Nemotron 3.5 ASR Streaming if NeMo toolkit is installed."""
    try:
        import nemo.collections.asr as nemo_asr  # type: ignore

        return nemo_asr.models.ASRModel.from_pretrained(
            "nvidia/nemotron-3.5-asr-streaming-0.6b"
        )
    except Exception:
        return None


class LiveTranscriber:
    """Streams microphone audio, transcribes in real time, and saves .txt on stop."""

    def __init__(self, on_update: Callable[[dict[str, Any]], None]) -> None:
        self._on_update = on_update
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()

    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self, language: str = "auto", initial_prompt: str = "") -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                args=(language, initial_prompt),
                name="live-transcriber",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        with self._lock:
            proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

    def _emit(self, **state: Any) -> None:
        try:
            self._on_update(state)
        except Exception:
            pass

    def _run(self, language: str, initial_prompt: str) -> None:
        t0 = time.time()
        whisper_model: Any = None
        nemotron_model: Any = None
        blocks: list[dict[str, Any]] = []

        try:
            cmd = _find_audio_capture_cmd()
            self._emit(
                status="loading",
                progress=0.0,
                filename="LIVE (Microfono)",
                language=language,
                duration=0.0,
                elapsed=0.0,
                job_started_at=t0,
                lines=[],
                ts_lines=[],
                speaker_blocks=[],
                speakers=[],
                speaker_names={},
                txt_path="",
                error="",
                message="LIVE: avvio Nemotron 3.5 / CUDA streaming...",
            )

            nemotron_model = _try_load_nemotron()
            if nemotron_model is not None:
                backend_label = "Nemotron 3.5 Streaming"
            else:
                whisper_model, dev = engine.load_whisper("large-v3-turbo")
                backend_label = (
                    "LIVE Streaming (CUDA)" if dev == "cuda" else "LIVE Streaming (CPU)"
                )

            if self._stop_event.is_set():
                return

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            with self._lock:
                self._proc = proc

            self._emit(
                status="transcribing",
                progress=0.0,
                message=f"{backend_label} attivo — parla nel microfono...",
            )

            # Read 2.5s chunks (16000 Hz * 2 bytes * 2.5s = 80000 bytes)
            chunk_bytes = 16000 * 2 * 3
            assert proc.stdout is not None
            while not self._stop_event.is_set():
                raw = proc.stdout.read(chunk_bytes)
                if not raw:
                    break
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                if len(audio) < 3200:
                    continue
                # Skip near-silent chunks quickly
                if float(np.sqrt(np.mean(audio**2))) < 0.004:
                    self._emit(elapsed=round(time.time() - t0, 1))
                    continue

                elapsed_now = time.time() - t0
                if nemotron_model is not None:
                    res = nemotron_model.transcribe([audio])
                    text = str(res[0].text if hasattr(res[0], "text") else res[0]).strip()
                else:
                    kw: dict[str, Any] = {
                        "beam_size": 3,
                        "condition_on_previous_text": False,
                        "vad_filter": True,
                        "vad_parameters": dict(engine.VAD_PARAMETERS),
                    }
                    if initial_prompt.strip():
                        kw["hotwords"] = initial_prompt.strip()
                    if language and language != "auto":
                        kw["language"] = language
                    segs, _ = whisper_model.transcribe(audio, **kw)
                    text = " ".join(s.text.strip() for s in segs if s.text.strip()).strip()

                if text:
                    blocks.append(
                        {
                            "speaker": "",
                            "start": max(0.0, elapsed_now - 3.0),
                            "text": text,
                        }
                    )
                    lines = engine.format_blocks(blocks, show_timestamps=False)
                    ts_lines = engine.format_blocks(blocks, show_timestamps=True)
                    self._emit(
                        status="transcribing",
                        lines=lines,
                        ts_lines=ts_lines,
                        speaker_blocks=blocks,
                        duration=round(elapsed_now, 1),
                        elapsed=round(elapsed_now, 1),
                        message=f"{backend_label} attivo — parla nel microfono...",
                    )

            engine.OUTPUT_DIR.mkdir(exist_ok=True)
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            out_txt = engine.OUTPUT_DIR / f"LIVE_{stamp}.txt"
            engine.save_blocks_to_txt(out_txt, blocks)

            lines = engine.format_blocks(blocks, show_timestamps=False)
            ts_lines = engine.format_blocks(blocks, show_timestamps=True)
            self._emit(
                status="done",
                progress=100.0,
                elapsed=round(time.time() - t0, 1),
                job_started_at=0.0,
                message=f"LIVE terminato! Salvato in Trascrizioni/{out_txt.name}",
                lines=lines,
                ts_lines=ts_lines,
                speaker_blocks=blocks,
                speakers=[],
                speaker_names={},
                txt_path=str(out_txt),
            )
        except Exception as exc:
            self._emit(
                status="error",
                elapsed=round(time.time() - t0, 1),
                job_started_at=0.0,
                error=str(exc),
                message=f"Errore LIVE: {exc}",
            )
        finally:
            with self._lock:
                proc = self._proc
                self._proc = None
                self._thread = None
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=1.0)
                except Exception:
                    pass
            engine.unload_whisper(whisper_model)
