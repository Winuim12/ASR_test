# 🎙️ ASR Test - Speech-to-Text + Diarization + Translation

A powerful Automatic Speech Recognition (ASR) pipeline built with **WhisperX**, supporting:

- 🎧 Speech-to-text (Vietnamese → English)
- 👥 Speaker diarization (no token required)
- ✨ Grammar & spelling correction (optional via Gemini)
- 🌐 Translation (free, using Google Translate)
- 📝 Export subtitles (SRT, VTT) + text + JSON

---

## 🚀 Features

- Accurate transcription using WhisperX (`large-v3`)
- Automatic speaker detection via clustering (no API token)
- Optional AI correction using Gemini
- Free translation pipeline
- Bilingual subtitles output (EN + VI)
- Clean logging (suppressed noisy logs)

---

## 📦 Installation

### 1. Install PyTorch (CUDA 12.8)
```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
````

### 2. Install dependencies

```bash
pip install git+https://github.com/m-bain/whisperx.git
pip install google-generativeai
pip install deep-translator
pip install scikit-learn librosa soundfile
```

### 3. Install FFmpeg

Required for audio preprocessing
Download: [https://ffmpeg.org/](https://ffmpeg.org/)

---

## ⚙️ Usage

### Basic command

```bash
python new.py --input "audio.mp3"
```

### Full pipeline (recommended)

```bash
python new.py \
  --input "test2.mp3" \
  --model large-v3 \
  --batch_size 8 \
  --diarize \
  --gemini_batch_size 30 \
  --out_dir "outputs"
```

---

## 🧠 Optional: Gemini Correction

Add your API key to enable grammar correction:

```bash
python new.py --input "audio.mp3" --gemini_key YOUR_API_KEY
```

If not provided:

> System will skip correction step.

---

## 👥 Speaker Diarization

Enable with:

```bash
--diarize
```

Optional:

```bash
--num_speakers 2
```

If not set:

> System automatically estimates number of speakers using clustering.

---

## 🌐 Language Settings

Inside code:

```python
INPUT_LANG = "vi"
OUTPUT_LANG = "en"
```

---

## 📂 Output Files

All outputs are saved in `--out_dir`:

| File                 | Description              |
| -------------------- | ------------------------ |
| `.json`              | Full structured segments |
| `.srt`               | Subtitles (EN + VI)      |
| `.vtt`               | Web subtitles            |
| `_raw.txt`           | Original transcription   |
| `_corrected.txt`     | Grammar-corrected text   |
| `_translated_en.txt` | Translated text          |

---

## 📝 Subtitle Format Example

```
1
00:00:01,000 --> 00:00:03,000
[SPEAKER_00] Hello everyone
[SPEAKER_00] Xin chào mọi người
```

---

## ⚠️ Notes

* Requires GPU for best performance (`cuda`)
* CPU mode supported but slower
* Gemini step is optional
* Translation uses free API → may be rate-limited
* Diarization uses clustering → not as accurate as pyannote

---

## 🛠️ Pipeline Overview

```
Audio → FFmpeg → WhisperX → Alignment
       → Diarization → (Gemini Correction)
       → Translation → Export (SRT/VTT/TXT/JSON)
```

---

## 💡 Example

```bash
python new.py \
  --input "meeting.mp3" \
  --diarize \
  --gemini_key YOUR_KEY \
  --out_dir results
```

---

## 📌 TODO

* Replace clustering diarization with pyannote
* Add real-time streaming
* Add web UI
* Multi-language support

```
