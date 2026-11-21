#!/usr/bin/env python3
"""Native Linux desktop application (GTK 4 + Libadwaita) for faster-whisper transcription."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

# Allow .venv Python to access Ubuntu's system PyGObject (GTK 4 / Libadwaita)
# while preserving .venv package priority (faster-whisper, ctranslate2, CUDA).
try:
    import gi
except ImportError:
    dist_packages = "/usr/lib/python3/dist-packages"
    if dist_packages not in sys.path:
        sys.path.append(dist_packages)
    import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, Gio, GLib, Gtk  # noqa: E402

import engine  # noqa: E402

APP_ID = "org.trascrizioneaudio.App"
APP_TITLE = "Trascrizione Audio"

APP_CSS = """
window.background {
    background-color: #0f1117;
    color: #e6eaf2;
}

headerbar {
    background-color: #141821;
    border-bottom: 1px solid #232a3a;
    box-shadow: none;
}

.card-panel {
    background-color: #181c26;
    border: 1px solid #2a3142;
    border-radius: 12px;
    padding: 14px 16px;
}

.muted-label {
    color: #8b95a8;
    font-size: 13px;
}

.field-label {
    color: #c5cedf;
    font-size: 13px;
    font-weight: 600;
}

.drop-zone {
    background-color: #181c26;
    border: 2px dashed #2e3850;
    border-radius: 14px;
    padding: 26px 20px;
    transition: background-color 140ms ease, border-color 140ms ease;
}

.drop-zone:hover {
    border-color: #4264a8;
    background-color: #1b2232;
}

.drop-zone.drop-over {
    border-color: #4f8cff;
    background-color: #1e2942;
}

.drop-title {
    font-size: 16px;
    font-weight: 700;
    color: #e6eaf2;
}

.drop-subtitle {
    font-size: 13px;
    color: #8b95a8;
}

progressbar trough {
    min-height: 10px;
    border-radius: 999px;
    background-color: #181c26;
    border: 1px solid #2a3142;
}

progressbar progress {
    min-height: 10px;
    border-radius: 999px;
    background-color: #4f8cff;
}

.transcript-scroll {
    background-color: #181c26;
    border: 1px solid #2a3142;
    border-radius: 12px;
}

.transcript-view {
    background-color: #181c26;
    color: #e6eaf2;
    font-size: 15px;
}

.transcript-view.placeholder-text {
    color: #7c879c;
}

.transcript-view text {
    background-color: #181c26;
    color: inherit;
}

button.accent-btn {
    background-color: #4f8cff;
    color: #ffffff;
    font-weight: 600;
    border-radius: 8px;
}

button.accent-btn:hover {
    background-color: #649bff;
}

button.secondary-btn {
    background-color: #181c26;
    color: #e6eaf2;
    border: 1px solid #2a3142;
    border-radius: 8px;
}

button.secondary-btn:hover {
    background-color: #202738;
    border-color: #3a4663;
}

button.ready-btn {
    background-color: #182338;
    color: #8ab4ff;
    border: 1px solid #4f8cff;
    border-radius: 8px;
}

button.ready-btn:hover {
    background-color: #1f2e4a;
}

.queue-pill {
    background-color: #1d2b49;
    color: #8ab4ff;
    border: 1px solid #355796;
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 12px;
    font-weight: 600;
}

.backend-pill {
    color: #8b95a8;
    font-size: 12px;
    padding: 2px 8px;
}

.backend-pill.ready {
    color: #6dd58c;
}
"""


class TranscriptionWindow(Adw.ApplicationWindow):
    def __init__(self, app: Adw.Application) -> None:
        super().__init__(application=app, title=APP_TITLE)
        self.set_default_size(900, 780)
        self.set_size_request(680, 560)

        self._last_state: dict[str, Any] = {}
        self._has_transcript = False
        self._closing_confirmed = False
        self._pulse_source_id: int | None = None
        self._timer_source_id: int | None = None
        self._copy_reset_source_id: int | None = None

        self._worker = engine.TranscriptionWorker(self._on_worker_update)

        self._build_ui()
        self.connect("close-request", self._on_close_request)

    def _build_ui(self) -> None:
        root_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)

        # Header Bar
        header = Adw.HeaderBar()
        title_widget = Adw.WindowTitle(
            title="Trascrizione Audio Locale",
            subtitle="GPU CUDA / CPU + VAD • Salvataggio automatico in Trascrizioni/",
        )
        header.set_title_widget(title_widget)

        self._backend_label = Gtk.Label(label="CUDA in avvio...")
        self._backend_label.add_css_class("backend-pill")
        header.pack_end(self._backend_label)

        root_box.append(header)

        # Toast overlay wrapping main workspace
        self._toast_overlay = Adw.ToastOverlay()
        self._toast_overlay.set_vexpand(True)

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=14)
        content.set_margin_top(16)
        content.set_margin_bottom(16)
        content.set_margin_start(20)
        content.set_margin_end(20)
        content.set_vexpand(True)

        # 1. Controls Card
        controls_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        controls_card.add_css_class("card-panel")

        row_top = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=14)

        model_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        model_box.set_hexpand(True)
        model_lbl = Gtk.Label(label="Modello Whisper", xalign=0.0)
        model_lbl.add_css_class("field-label")
        self._model_dropdown = Gtk.DropDown.new_from_strings(
            [label for _, label in engine.MODEL_OPTIONS]
        )
        self._model_dropdown.set_selected(0)
        self._model_dropdown.set_hexpand(True)
        model_box.append(model_lbl)
        model_box.append(self._model_dropdown)

        lang_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        lang_lbl = Gtk.Label(label="Lingua", xalign=0.0)
        lang_lbl.add_css_class("field-label")
        self._lang_dropdown = Gtk.DropDown.new_from_strings(
            [label for _, label in engine.LANGUAGE_OPTIONS]
        )
        self._lang_dropdown.set_selected(0)
        self._lang_dropdown.set_size_request(210, -1)
        lang_box.append(lang_lbl)
        lang_box.append(self._lang_dropdown)

        row_top.append(model_box)
        row_top.append(lang_box)
        controls_card.append(row_top)

        row_toggles = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=16)

        med_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self._medical_switch = Gtk.Switch()
        self._medical_switch.set_active(True)
        self._medical_switch.set_valign(Gtk.Align.CENTER)
        self._medical_switch.set_tooltip_text(
            "Usa il prompt medico specialistico (anatomia, radiologia, clinica, patologia) "
            "per migliorare l'accuratezza dei termini tecnici."
        )
        med_lbl = Gtk.Label(label="Contesto medico")
        med_lbl.add_css_class("field-label")
        med_lbl.set_tooltip_text(engine.MEDICAL_INITIAL_PROMPT)
        med_box.append(self._medical_switch)
        med_box.append(med_lbl)

        spacer = Gtk.Box()
        spacer.set_hexpand(True)

        self._show_ts_check = Gtk.CheckButton(label="Mostra timestamp")
        self._show_ts_check.set_active(False)
        self._show_ts_check.connect("toggled", lambda _btn: self._render_transcript())

        row_toggles.append(med_box)
        row_toggles.append(spacer)
        row_toggles.append(self._show_ts_check)
        controls_card.append(row_toggles)

        content.append(controls_card)

        # 2. Drag & Drop + Browse Zone
        self._drop_zone = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self._drop_zone.add_css_class("drop-zone")
        self._drop_zone.set_halign(Gtk.Align.FILL)
        self._drop_zone.set_cursor_from_name("pointer")

        drop_icon = Gtk.Image.new_from_icon_name("audio-input-microphone-symbolic")
        drop_icon.set_pixel_size(32)
        drop_icon.set_opacity(0.85)

        drop_title = Gtk.Label(label="Trascina qui uno o più file audio o video")
        drop_title.add_css_class("drop-title")

        drop_sub = Gtk.Label(
            label="oppure clicca per sceglierli (.mp3, .wav, .m4a, .mp4, .mkv, .ogg, .flac, .webm) • i file extra vengono messi in coda"
        )
        drop_sub.add_css_class("drop-subtitle")
        drop_sub.set_wrap(True)
        drop_sub.set_justify(Gtk.Justification.CENTER)

        browse_btn = Gtk.Button(label="Sfoglia file...")
        browse_btn.add_css_class("accent-btn")
        browse_btn.set_halign(Gtk.Align.CENTER)
        browse_btn.set_margin_top(6)
        browse_btn.connect("clicked", lambda _btn: self._open_file_dialog())

        self._drop_zone.append(drop_icon)
        self._drop_zone.append(drop_title)
        self._drop_zone.append(drop_sub)
        self._drop_zone.append(browse_btn)

        # Clicking anywhere inside the drop zone opens the file picker
        click_gesture = Gtk.GestureClick()
        click_gesture.connect("released", self._on_drop_zone_clicked)
        self._drop_zone.add_controller(click_gesture)

        # Native Wayland Drag & Drop on both drop zone and entire window
        self._attach_drop_target(self._drop_zone)
        self._attach_drop_target(content)

        content.append(self._drop_zone)

        # 3. Progress Bar & Metadata Row
        progress_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)

        self._progress_bar = Gtk.ProgressBar()
        self._progress_bar.set_fraction(0.0)
        progress_box.append(self._progress_bar)

        meta_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)

        self._status_label = Gtk.Label(
            label="Pronto. Trascina uno o più file audio o video.", xalign=0.0
        )
        self._status_label.add_css_class("muted-label")
        self._status_label.set_ellipsize(3)  # Pango.EllipsizeMode.END
        self._status_label.set_hexpand(True)

        self._queue_badge = Gtk.Label(label="")
        self._queue_badge.add_css_class("queue-pill")
        self._queue_badge.set_visible(False)

        self._info_label = Gtk.Label(label="", xalign=1.0)
        self._info_label.add_css_class("muted-label")

        meta_row.append(self._status_label)
        meta_row.append(self._queue_badge)
        meta_row.append(self._info_label)
        progress_box.append(meta_row)

        content.append(progress_box)

        # 4. Transcript Viewer
        self._scroll = Gtk.ScrolledWindow()
        self._scroll.add_css_class("transcript-scroll")
        self._scroll.set_vexpand(True)
        self._scroll.set_hexpand(True)
        self._scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

        self._text_view = Gtk.TextView()
        self._text_view.add_css_class("transcript-view")
        self._text_view.add_css_class("placeholder-text")
        self._text_view.set_editable(False)
        self._text_view.set_cursor_visible(False)
        self._text_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self._text_view.set_top_margin(16)
        self._text_view.set_bottom_margin(16)
        self._text_view.set_left_margin(18)
        self._text_view.set_right_margin(18)
        self._text_view.set_pixels_below_lines(6)
        self._text_view.set_pixels_inside_wrap(3)

        self._text_buffer = self._text_view.get_buffer()
        self._text_buffer.set_text("Il testo trascritto apparirà qui in tempo reale...")

        self._scroll.set_child(self._text_view)
        content.append(self._scroll)

        # 5. Bottom Action Bar
        actions_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)

        self._copy_btn = Gtk.Button(label="Copia testo")
        self._copy_btn.add_css_class("secondary-btn")
        self._copy_btn.set_sensitive(False)
        self._copy_btn.connect("clicked", self._on_copy_clicked)

        self._open_txt_btn = Gtk.Button(label="Apri .txt")
        self._open_txt_btn.add_css_class("secondary-btn")
        self._open_txt_btn.set_sensitive(False)
        self._open_txt_btn.connect("clicked", self._on_open_txt_clicked)

        self._open_srt_btn = Gtk.Button(label="Apri .srt")
        self._open_srt_btn.add_css_class("secondary-btn")
        self._open_srt_btn.set_sensitive(False)
        self._open_srt_btn.connect("clicked", self._on_open_srt_clicked)

        self._open_dir_btn = Gtk.Button(label="Apri cartella Trascrizioni")
        self._open_dir_btn.add_css_class("secondary-btn")
        self._open_dir_btn.connect("clicked", self._on_open_dir_clicked)

        actions_spacer = Gtk.Box()
        actions_spacer.set_hexpand(True)

        self._cancel_btn = Gtk.Button(label="Annulla")
        self._cancel_btn.add_css_class("destructive-action")
        self._cancel_btn.set_visible(False)
        self._cancel_btn.connect("clicked", self._on_cancel_clicked)

        actions_row.append(self._copy_btn)
        actions_row.append(self._open_txt_btn)
        actions_row.append(self._open_srt_btn)
        actions_row.append(self._open_dir_btn)
        actions_row.append(actions_spacer)
        actions_row.append(self._cancel_btn)

        content.append(actions_row)

        self._toast_overlay.set_child(content)
        root_box.append(self._toast_overlay)
        self.set_content(root_box)

    def set_backend_status(self, text: str, ready: bool = False) -> None:
        self._backend_label.set_text(text)
        if ready:
            self._backend_label.add_css_class("ready")
        else:
            self._backend_label.remove_css_class("ready")

    def _attach_drop_target(self, widget: Gtk.Widget) -> None:
        drop_target = Gtk.DropTarget.new(Gdk.FileList.__gtype__, Gdk.DragAction.COPY)
        drop_target.set_gtypes([Gdk.FileList.__gtype__, Gio.File.__gtype__])
        drop_target.connect("enter", self._on_drag_enter)
        drop_target.connect("leave", self._on_drag_leave)
        drop_target.connect("drop", self._on_drop)
        widget.add_controller(drop_target)

    def _on_drag_enter(
        self, _target: Gtk.DropTarget, _x: float, _y: float
    ) -> Gdk.DragAction:
        self._drop_zone.add_css_class("drop-over")
        return Gdk.DragAction.COPY

    def _on_drag_leave(self, _target: Gtk.DropTarget) -> None:
        self._drop_zone.remove_css_class("drop-over")

    def _on_drop(
        self, _target: Gtk.DropTarget, value: Any, _x: float, _y: float
    ) -> bool:
        self._drop_zone.remove_css_class("drop-over")
        paths: list[Path] = []
        if isinstance(value, Gdk.FileList):
            for gfile in value.get_files():
                p = gfile.get_path()
                if p:
                    paths.append(Path(p))
        elif isinstance(value, Gio.File):
            p = value.get_path()
            if p:
                paths.append(Path(p))
        if not paths:
            return False
        self.enqueue_paths(paths)
        return True

    def _on_drop_zone_clicked(
        self, _gesture: Gtk.GestureClick, _n_press: int, x: float, y: float
    ) -> None:
        picked = self._drop_zone.pick(x, y, Gtk.PickFlags.DEFAULT)
        if picked is not None and picked.get_ancestor(Gtk.Button) is not None:
            return
        self._open_file_dialog()

    def _open_file_dialog(self) -> None:
        dialog = Gtk.FileDialog()
        dialog.set_title("Seleziona uno o più file audio o video")

        media_filter = Gtk.FileFilter()
        media_filter.set_name("File Audio e Video")
        media_filter.add_mime_type("audio/*")
        media_filter.add_mime_type("video/*")
        for ext in sorted(engine.SUPPORTED_EXTENSIONS):
            media_filter.add_pattern(f"*{ext}")
            media_filter.add_pattern(f"*{ext.upper()}")

        all_filter = Gtk.FileFilter()
        all_filter.set_name("Tutti i file")
        all_filter.add_pattern("*")

        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(media_filter)
        filters.append(all_filter)
        dialog.set_filters(filters)
        dialog.set_default_filter(media_filter)

        dialog.open_multiple(self, None, self._on_file_dialog_response)

    def _on_file_dialog_response(
        self, dialog: Gtk.FileDialog, result: Gio.AsyncResult
    ) -> None:
        try:
            files_model = dialog.open_multiple_finish(result)
        except GLib.Error:
            return
        if files_model is None:
            return
        paths: list[Path] = []
        for idx in range(files_model.get_n_items()):
            gfile = files_model.get_item(idx)
            if isinstance(gfile, Gio.File):
                p = gfile.get_path()
                if p:
                    paths.append(Path(p))
        if paths:
            self.enqueue_paths(paths)

    def _selected_model(self) -> str:
        idx = self._model_dropdown.get_selected()
        if 0 <= idx < len(engine.MODEL_OPTIONS):
            return engine.MODEL_OPTIONS[idx][0]
        return "large-v3"

    def _selected_language(self) -> str:
        idx = self._lang_dropdown.get_selected()
        if 0 <= idx < len(engine.LANGUAGE_OPTIONS):
            return engine.LANGUAGE_OPTIONS[idx][0]
        return "auto"

    def enqueue_paths(self, paths: list[Path]) -> None:
        valid_files = [p for p in paths if p.is_file()]
        skipped = len(paths) - len(valid_files)
        if not valid_files:
            self._show_toast("Seleziona almeno un file audio o video valido.")
            return

        was_busy = self._worker.is_busy()
        model_mode = self._selected_model()
        language = self._selected_language()
        use_medical = self._medical_switch.get_active()

        jobs = [
            engine.TranscriptionJob(
                file_path=p,
                language=language,
                model_mode=model_mode,
                use_medical_context=use_medical,
            )
            for p in valid_files
        ]
        added = self._worker.enqueue(jobs)

        if added == 0:
            return
        if was_busy:
            msg = (
                f"Aggiunto alla coda: {valid_files[0].name}"
                if added == 1
                else f"Aggiunti {added} file alla coda"
            )
            self._show_toast(msg)
        elif added > 1:
            self._show_toast(
                f"Avviata trascrizione di {valid_files[0].name} (+{added - 1} in coda)"
            )
        if skipped > 0:
            self._show_toast(f"Ignorati {skipped} elementi non validi (es. cartelle).")

    def _on_worker_update(self, snap: dict[str, Any]) -> None:
        GLib.idle_add(self._apply_state, snap)

    def _apply_state(self, snap: dict[str, Any]) -> bool:
        self._last_state = snap
        status = snap.get("status", "idle")
        progress = float(snap.get("progress", 0.0))
        queue_size = int(snap.get("queue_size", 0))

        self._status_label.set_text(snap.get("message", ""))

        if queue_size > 0:
            label = "In coda: 1 file" if queue_size == 1 else f"In coda: {queue_size} file"
            self._queue_badge.set_text(label)
            self._queue_badge.set_visible(True)
        else:
            self._queue_badge.set_visible(False)

        # Progress bar and timers
        if status == "loading":
            self._start_pulse()
            self._start_elapsed_timer()
            self._cancel_btn.set_visible(True)
            self._cancel_btn.set_sensitive(True)
        elif status == "transcribing":
            self._stop_pulse()
            self._progress_bar.set_fraction(max(0.0, min(1.0, progress / 100.0)))
            self._start_elapsed_timer()
            self._cancel_btn.set_visible(True)
            self._cancel_btn.set_sensitive(True)
        else:
            self._stop_pulse()
            self._stop_elapsed_timer()
            self._cancel_btn.set_visible(False)
            if status == "done":
                self._progress_bar.set_fraction(1.0)
            else:
                self._progress_bar.set_fraction(0.0)

        self._update_info_label(snap)

        # Update transcript viewer
        lines = snap.get("lines") or []
        if lines:
            self._has_transcript = True
            self._render_transcript(auto_scroll=True)
        elif status == "loading":
            self._has_transcript = False
            self._set_placeholder(f"Caricamento in corso per {snap.get('filename', '')}...")
        elif status == "error":
            self._has_transcript = False
            self._set_placeholder(f"Errore: {snap.get('error', '')}")
        elif status == "done" and not lines:
            self._has_transcript = False
            self._set_placeholder("(Nessun parlato rilevato nel file audio.)")

        # Action buttons state
        self._copy_btn.set_sensitive(self._has_transcript)
        txt_ready = bool(snap.get("txt_path")) and Path(snap["txt_path"]).is_file()
        srt_ready = bool(snap.get("srt_path")) and Path(snap["srt_path"]).is_file()
        self._set_file_button_ready(self._open_txt_btn, txt_ready)
        self._set_file_button_ready(self._open_srt_btn, srt_ready)

        if status == "done" and queue_size == 0:
            txt_name = Path(snap["txt_path"]).name if snap.get("txt_path") else ""
            if txt_name:
                self._show_toast(f"Salvato in Trascrizioni/{txt_name}")

        return GLib.SOURCE_REMOVE

    def _set_file_button_ready(self, button: Gtk.Button, ready: bool) -> None:
        button.set_sensitive(ready)
        if ready:
            button.remove_css_class("secondary-btn")
            button.add_css_class("ready-btn")
        else:
            button.remove_css_class("ready-btn")
            button.add_css_class("secondary-btn")

    def _update_info_label(self, snap: dict[str, Any]) -> None:
        parts: list[str] = []
        status = snap.get("status", "idle")
        progress = float(snap.get("progress", 0.0))
        if status == "transcribing":
            parts.append(f"{progress:.0f}%")
        if snap.get("language"):
            parts.append(f"Lingua: {snap['language']}")
        duration = float(snap.get("duration", 0.0))
        if duration > 0:
            parts.append(f"Durata: {engine.format_short(duration)}")

        started_at = float(snap.get("job_started_at", 0.0))
        if status in ("loading", "transcribing") and started_at > 0:
            elapsed = max(0.0, time.time() - started_at)
        else:
            elapsed = float(snap.get("elapsed", 0.0))
        if elapsed > 0:
            parts.append(f"Tempo: {elapsed:.1f}s")

        self._info_label.set_text(" • ".join(parts))

    def _start_pulse(self) -> None:
        if self._pulse_source_id is None:
            self._progress_bar.set_pulse_step(0.08)
            self._pulse_source_id = GLib.timeout_add(110, self._on_pulse_tick)

    def _stop_pulse(self) -> None:
        if self._pulse_source_id is not None:
            GLib.source_remove(self._pulse_source_id)
            self._pulse_source_id = None

    def _on_pulse_tick(self) -> bool:
        self._progress_bar.pulse()
        return GLib.SOURCE_CONTINUE

    def _start_elapsed_timer(self) -> None:
        if self._timer_source_id is None:
            self._timer_source_id = GLib.timeout_add(500, self._on_elapsed_tick)

    def _stop_elapsed_timer(self) -> None:
        if self._timer_source_id is not None:
            GLib.source_remove(self._timer_source_id)
            self._timer_source_id = None

    def _on_elapsed_tick(self) -> bool:
        if self._last_state.get("status") not in ("loading", "transcribing"):
            self._timer_source_id = None
            return GLib.SOURCE_REMOVE
        self._update_info_label(self._last_state)
        return GLib.SOURCE_CONTINUE

    def _set_placeholder(self, text: str) -> None:
        self._text_view.add_css_class("placeholder-text")
        self._text_buffer.set_text(text)

    def _current_transcript_text(self) -> str:
        if not self._last_state:
            return ""
        use_ts = self._show_ts_check.get_active()
        arr = (
            self._last_state.get("ts_lines")
            if use_ts and self._last_state.get("ts_lines")
            else self._last_state.get("lines")
        )
        return "\n\n".join(arr) if arr else ""

    def _render_transcript(self, auto_scroll: bool = False) -> None:
        text = self._current_transcript_text()
        if not text:
            return
        vadj = self._scroll.get_vadjustment()
        at_bottom = (vadj.get_value() + vadj.get_page_size()) >= (
            vadj.get_upper() - 36.0
        )

        self._text_view.remove_css_class("placeholder-text")
        self._text_buffer.set_text(text)

        if auto_scroll and at_bottom:
            GLib.idle_add(self._scroll_to_bottom)

    def _scroll_to_bottom(self) -> bool:
        vadj = self._scroll.get_vadjustment()
        vadj.set_value(max(0.0, vadj.get_upper() - vadj.get_page_size()))
        return GLib.SOURCE_REMOVE

    def _on_copy_clicked(self, _button: Gtk.Button) -> None:
        text = self._current_transcript_text()
        if not text:
            return
        display = Gdk.Display.get_default()
        if display is not None:
            display.get_clipboard().set(text)
        self._copy_btn.set_label("Copiato!")
        self._show_toast("Testo copiato negli appunti")
        if self._copy_reset_source_id is not None:
            GLib.source_remove(self._copy_reset_source_id)
        self._copy_reset_source_id = GLib.timeout_add(1500, self._reset_copy_label)

    def _reset_copy_label(self) -> bool:
        self._copy_btn.set_label("Copia testo")
        self._copy_reset_source_id = None
        return GLib.SOURCE_REMOVE

    def _launch_path(self, path_str: str) -> None:
        if not path_str:
            return
        target = Path(path_str)
        if not target.exists():
            self._show_toast("File o cartella non trovato.")
            return
        gfile = Gio.File.new_for_path(str(target))
        launcher = Gtk.FileLauncher.new(gfile)
        launcher.launch(self, None, self._on_launch_done)

    def _on_launch_done(
        self, launcher: Gtk.FileLauncher, result: Gio.AsyncResult
    ) -> None:
        try:
            launcher.launch_finish(result)
        except GLib.Error as err:
            # Fallback to Gio.AppInfo if portal launcher is unavailable
            gfile = launcher.get_file()
            if gfile is not None:
                try:
                    Gio.AppInfo.launch_default_for_uri(gfile.get_uri(), None)
                    return
                except GLib.Error:
                    pass
            self._show_toast(f"Impossibile aprire: {err.message}")

    def _on_open_txt_clicked(self, _button: Gtk.Button) -> None:
        self._launch_path(self._last_state.get("txt_path", ""))

    def _on_open_srt_clicked(self, _button: Gtk.Button) -> None:
        self._launch_path(self._last_state.get("srt_path", ""))

    def _on_open_dir_clicked(self, _button: Gtk.Button) -> None:
        engine.OUTPUT_DIR.mkdir(exist_ok=True)
        self._launch_path(str(engine.OUTPUT_DIR))

    def _on_cancel_clicked(self, _button: Gtk.Button) -> None:
        self._cancel_btn.set_sensitive(False)
        self._worker.cancel()

    def _show_toast(self, message: str) -> None:
        toast = Adw.Toast.new(message)
        toast.set_timeout(3)
        self._toast_overlay.add_toast(toast)

    def _on_close_request(self, _window: Gtk.Window) -> bool:
        if self._closing_confirmed or not self._worker.is_busy():
            self._stop_pulse()
            self._stop_elapsed_timer()
            self._worker.shutdown(timeout=2.0)
            return False

        dialog = Adw.AlertDialog(
            heading="Trascrizione in corso",
            body="Vuoi annullare la trascrizione attiva, liberare la GPU e chiudere l'applicazione?",
        )
        dialog.add_response("stay", "Continua")
        dialog.add_response("close", "Annulla e chiudi")
        dialog.set_response_appearance("close", Adw.ResponseAppearance.DESTRUCTIVE)
        dialog.set_default_response("stay")
        dialog.set_close_response("stay")
        dialog.connect("response", self._on_close_dialog_response)
        dialog.present(self)
        return True

    def _on_close_dialog_response(
        self, _dialog: Adw.AlertDialog, response: str
    ) -> None:
        if response == "close":
            self._closing_confirmed = True
            self._stop_pulse()
            self._stop_elapsed_timer()
            self._worker.shutdown(timeout=5.0)
            self.close()


class TranscriptionApp(Adw.Application):
    def __init__(self) -> None:
        super().__init__(
            application_id=APP_ID,
            flags=Gio.ApplicationFlags.HANDLES_OPEN,
        )
        self._window: TranscriptionWindow | None = None
        self._preload_scheduled = False

    def do_startup(self) -> None:
        Adw.Application.do_startup(self)
        GLib.set_application_name(APP_TITLE)
        GLib.set_prgname(APP_ID)

        close_action = Gio.SimpleAction.new("close", None)
        close_action.connect("activate", self._on_close_action)
        self.add_action(close_action)
        self.set_accels_for_action("app.close", ["<Primary>w", "<Primary>q"])

        style_manager = Adw.StyleManager.get_default()
        style_manager.set_color_scheme(Adw.ColorScheme.FORCE_DARK)

        display = Gdk.Display.get_default()
        if display is not None:
            provider = Gtk.CssProvider()
            provider.load_from_string(APP_CSS)
            Gtk.StyleContext.add_provider_for_display(
                display, provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
            )

    def _on_close_action(self, _action: Gio.SimpleAction, _param: Any) -> None:
        if self._window is not None:
            self._window.close()

    def _ensure_window(self) -> TranscriptionWindow:
        if self._window is None:
            self._window = TranscriptionWindow(self)
        self._window.present()
        if not self._preload_scheduled:
            self._preload_scheduled = True
            GLib.idle_add(self._start_background_preload)
        return self._window

    def _start_background_preload(self) -> bool:
        def on_ready() -> None:
            GLib.idle_add(self._notify_backend_ready)

        def on_error(exc: Exception) -> None:
            GLib.idle_add(self._notify_backend_error, str(exc))

        engine.preload_backend_async(on_ready=on_ready, on_error=on_error)
        return GLib.SOURCE_REMOVE

    def _notify_backend_ready(self) -> bool:
        if self._window is not None:
            self._window.set_backend_status("CUDA pronto", ready=True)
        return GLib.SOURCE_REMOVE

    def _notify_backend_error(self, message: str) -> bool:
        if self._window is not None:
            self._window.set_backend_status("Errore CUDA", ready=False)
            self._window._show_toast(f"Errore caricamento backend: {message}")
        return GLib.SOURCE_REMOVE

    def do_activate(self) -> None:
        self._ensure_window()

    def do_open(self, files: list[Gio.File], _n_files: int, _hint: str) -> None:
        win = self._ensure_window()
        paths = [Path(f.get_path()) for f in files if f.get_path()]
        if paths:
            win.enqueue_paths(paths)


def main() -> int:
    app = TranscriptionApp()
    return app.run(sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())
