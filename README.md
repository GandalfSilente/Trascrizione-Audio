# Trascrizione Audio Locale (GTK 4 + Libadwaita + faster-whisper)

Applicazione desktop nativa per Linux (con script CLI incluso) per trascrivere file audio e video lunghi **100% in locale e offline**, sfruttando `faster-whisper` con accelerazione **GPU NVIDIA CUDA** (e fallback automatico su CPU) e filtro vocale **VAD**.

Progettata per avviarsi istantaneamente senza browser né server web locali, con supporto a Drag & Drop nativo Wayland, coda multi-file e profili di contesto personalizzabili per adattare il riconoscimento a qualsiasi disciplina (medicina, fisica, informatica, ecc.).

---

## Funzionalità principali

- **Interfaccia desktop nativa Linux (`GTK 4 + Libadwaita`)**: tema scuro leggero, avvio istantaneo tramite *lazy loading* delle librerie CUDA e di `faster-whisper` in background (0 VRAM occupata all'avvio).
- **Drag & Drop multi-file e coda sequenziale**: trascina uno o più file audio/video (o usa il pulsante *Sfoglia file...*); eventuali file aggiunti durante una trascrizione vengono messi in coda automaticamente.
- **3 modalità di trascrizione**:
  - `large-v3` *(predefinito)*: massima accuratezza.
  - `Combinato (Ensemble: large-v3-turbo + large-v3)`: doppio passaggio con confronto sulla confidenza (`avg_logprob`) per migliorare i segmenti incerti mantenendo basso il picco di VRAM.
  - `large-v3-turbo`: ultra veloce, ideale per audio molto lunghi o PC meno potenti.
- **Profili di contesto personalizzabili**: menu a tendina con preset pronti (*Generale*, *Medicina e Radiologia IT*, *Fisica / Fisica Nucleare EN*, *Fisica e Ingegneria IT*, *Informatica e AI EN/IT*) più un pannello espandibile per scrivere/modificare il prompt tecnico e salvare o eliminare i propri profili personalizzati in `~/.config/trascrizione-audio/config.json` (fuori dal repository Git). Ogni profilo imposta automaticamente la lingua consigliata (sempre sovrascrivibile manualmente) e l'app ricorda le ultime scelte tra un avvio e l'altro.
- **Rilevamento lingua e VAD calibrato per aule**: selezione lingua (`Automatica IT/EN`, `Italiano`, `Inglese`) e Silero VAD ottimizzato per microfoni d'aula/distanti (`threshold=0.35`, `min_silence_duration_ms=600`, `speech_pad_ms=400`) per evitare tagli delle parole iniziali e ridurre le allucinazioni nei silenzi.
- **Visualizzazione live & export automatico**:
  - Streaming del testo in tempo reale raggruppato in paragrafi leggibili.
  - Toggle **Mostra timestamp** `(MM:SS)` senza ripetere la trascrizione.
  - Salvataggio automatico di `<nome_file>.txt` e `<nome_file>.srt` nella cartella `Trascrizioni/`.
  - Pulsanti rapidi *Copia testo*, *Apri .txt*, *Apri .srt*, *Apri cartella Trascrizioni* e *Annulla*.
- **Gestione pulita della GPU**: rilascio automatico della VRAM al termine della coda o in caso di annullamento/chiusura.
- **Istanza singola e scorciatoie**: riaprire l'app porta in primo piano la finestra esistente; chiusura rapida con `Ctrl+W` o `Ctrl+Q` (con conferma se una trascrizione è in corso).
- **Script CLI alternativo (`trascrivi.py`)**: per trascrivere rapidamente da terminale o tramite selettore `zenity`.

---

## Chi può usarlo e compatibilità dispositivi

### A chi è pensato
- **Studenti universitari** (in particolare Medicina, Chirurgia, Professioni Sanitarie e facoltà tecnico-scientifiche) per sbobinare lezioni lunghe senza limiti di minuti né abbonamenti cloud.
- **Ricercatori, giornalisti, professionisti e docenti** che trattano interviste, riunioni o appunti vocali riservati e vogliono che i file **non lascino mai il proprio computer**.

### Sistemi operativi supportati
| Sistema Operativo | App Desktop (`app.py`) | CLI (`trascrivi.py`) | Note |
| --- | :---: | :---: | --- |
| **Linux** (Ubuntu 22.04/24.04+, Fedora, Debian, Arch con GNOME/Wayland o X11) | ✅ Pieno supporto | ✅ Pieno supporto | Consigliato GTK 4.12+ e Libadwaita 1.4+ |
| **Windows / macOS** | ❌ Non supportata | ⚠️ Parziale | `app.py` usa GTK 4/Libadwaita e `.so` Linux; `trascrivi.py` può girare su CPU (o CUDA su Windows) |

### Requisiti hardware (GPU / CPU / RAM)
| Hardware | Esperienza d'uso | Modello consigliato | Requisiti indicativi |
| --- | --- | --- | --- |
| **GPU NVIDIA con CUDA 12+** *(consigliato)* | Molto veloce (molte volte più rapido del tempo reale, `float16`) | `large-v3`, `Ensemble` o `large-v3-turbo` | Driver NVIDIA aggiornati; **≥ 6–8 GB VRAM** per `large-v3` / `Ensemble`, **≥ 4 GB VRAM** per `large-v3-turbo` |
| **Solo CPU (senza GPU NVIDIA)** | Fallback automatico `int8` (più lento ma pienamente funzionante) | `large-v3-turbo` | CPU multi-core moderna, **≥ 8–16 GB RAM** |
| **GPU AMD / Intel / Apple Silicon** | Usa automaticamente il fallback CPU `int8` | `large-v3-turbo` | CTranslate2 utilizza CUDA su NVIDIA |

> **Spazio su disco**: alla prima esecuzione i modelli Whisper vengono scaricati automaticamente da Hugging Face nella cache locale (~1,6 GB per `large-v3-turbo`, ~3 GB per `large-v3`). Le trascrizioni successive funzionano completamente offline.

---

## Formati audio/video supportati

Supporta tutti i principali formati decodificabili da PyAV/FFmpeg:
`.mp3`, `.wav`, `.m4a`, `.mp4`, `.mkv`, `.ogg`, `.flac`, `.webm`, `.aac`, `.opus`, `.mov`, `.avi`, `.wma`, `.mpeg`, `.mpg`.

---

## Installazione (Ubuntu / Debian)

### 1. Dipendenze di sistema (GTK 4 + Libadwaita)
Su Ubuntu 24.04 GNOME sono spesso già presenti; in caso contrario installale con:

```bash
sudo apt update
sudo apt install python3-venv python3-gi gir1.2-gtk-4.0 gir1.2-adw-1
```

*(Su altre distribuzioni installa i corrispondenti pacchetti `pygobject`, `gtk4` e `libadwaita`.)*

### 2. Clona il repository e crea l'ambiente virtuale

```bash
git clone https://github.com/GandalfSilente/Trascrizione-Audio.git
cd Trascrizione-Audio
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

> **Solo CPU (senza GPU NVIDIA)**: puoi installare unicamente `.venv/bin/pip install faster-whisper` senza i pacchetti `nvidia-cublas-cu12` e `nvidia-cudnn-cu12`.

---

## Utilizzo

### Avvio dell'app desktop
```bash
.venv/bin/python app.py
```

Puoi anche passare uno o più file direttamente da riga di comando:
```bash
.venv/bin/python app.py lezione1.mp3 lezione2.mp4
```

### Avvio della versione CLI
```bash
.venv/bin/python trascrivi.py percorso/del/file.mp3
```
Se eseguito senza argomenti, `trascrivi.py` apre il selettore file (`zenity`, se installato) oppure chiede il percorso da terminale e salva `.txt` e `.srt` accanto al file sorgente.

### Integrazione nel Desktop / Menu Applicazioni Linux (opzionale)
Crea un file `~/.local/share/applications/Trascrizione-Audio.desktop` (o sul Desktop) sostituendo `/PERCORSO/DEL/PROGETTO` con la cartella in cui hai clonato il repository:

```ini
[Desktop Entry]
Type=Application
Name=Trascrizione Audio
Comment=Trascrivi audio e video localmente (App Desktop Nativa)
Exec=/PERCORSO/DEL/PROGETTO/.venv/bin/python /PERCORSO/DEL/PROGETTO/app.py %F
Icon=audio-input-microphone
Terminal=false
Categories=AudioVideo;Utility;
StartupWMClass=org.trascrizioneaudio.App
```

Rendi eseguibile il launcher:
```bash
chmod +x ~/.local/share/applications/Trascrizione-Audio.desktop
```

---

## Struttura del progetto

- `app.py` — Interfaccia desktop nativa GTK 4 + Libadwaita (Drag & Drop, coda, profili di contesto personalizzabili, visualizzazione live, scorciatoie `Ctrl+W`/`Ctrl+Q`).
- `engine.py` — Motore di trascrizione disaccoppiato dalla GUI (lazy loading CUDA/`faster-whisper`, gestione coda FIFO, preset/persistenza XDG, VAD d'aula, modalità Ensemble, annullamento e rilascio VRAM).
- `trascrivi.py` — Utility CLI indipendente con `large-v3-turbo`.
- `requirements.txt` — Dipendenze Python per la `.venv`.
- `Trascrizioni/` — Cartella generata automaticamente per i file `.txt` e `.srt` prodotti (ignorata da Git).

---

## Privacy

- Nessun dato audio, video o testuale viene mai inviato a server esterni.
- I file sorgente vengono letti direttamente dal disco senza creare copie temporanee e **non vengono mai modificati né eliminati**.
- I profili personalizzati e le preferenze dell'interfaccia sono salvati localmente in `~/.config/trascrizione-audio/config.json`, separati dal repository.
- La cartella `Trascrizioni/` è esclusa dal tracciamento Git tramite `.gitignore`.
