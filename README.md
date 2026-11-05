# Audio Watermark Stress Test (RIAA x Duke)

We test whether AI music watermarks survive the edits real tracks go through, and what that means for laws that require machine-readable disclosure of AI-generated audio: EU AI Act Article 50 and California's AI Transparency Act (SB 942 / AB 853), both in force since August 2, 2026.

## Research questions

1. **Detection:** after an edit, does the detector still find the watermark?
2. **Payload:** can the information embedded in the watermark still be read intact? California requires the latent disclosure to carry a unique ID, system name and version, and creation date, so a mark that is detected but garbled fails the law's purpose.

## This part of the project

This repo covers one fifth of the team's testing: **SynthID** (Google) and **AudioSeal** (Meta), tested on the same edits so they can be compared side by side, plus **ViSQOL** scores for how audible the AudioSeal mark is. Other watermarks are tested by teammates.

Results and their limits are written up in `findings.md`.

## Audio files

The clean source tracks are in the team Google Drive, not in this repo, because each WAV is about 35 MB:

https://drive.google.com/drive/folders/1vrRCooxcsKfvh9Sg4GQtL-73ycfX16Hv?usp=drive_link

Download them into the top level of this repo folder before running anything. They are 8 tracks (`lyria_{dense,sparse,pop,ambient}_{1,2}.wav`), 48 kHz, 16-bit stereo, generated on Google Flow Music (Lyria 3.5), which embeds SynthID in every track. Prompts are listed in `attacks.md`.

## One-time setup (Mac)

```bash
brew install sox ffmpeg
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The stem-split edit uses Demucs, which lives in its own environment so it cannot change the package versions AudioSeal uses:

```bash
python3 -m venv .venv-demucs
.venv-demucs/bin/pip install demucs "torch<2.9" "torchaudio<2.9" soundfile
```

ViSQOL (sound quality scores) also has its own environment:

```bash
python3 -m venv .venv-visqol
.venv-visqol/bin/pip install visqol-python==3.8.0 soundfile numpy
```

## Running the tests

Always start with `source .venv/bin/activate` from this folder.

| Command | What it does | Output |
| --- | --- | --- |
| `python run_audioseal.py` | Watermarks all 8 tracks with AudioSeal, applies every edit, logs detection and payload. About 30 to 45 minutes. | `aseal/audioseal_results.csv` |
| `python shortcut_test.py` | Checks whether our embedding method (watermark computed at 16 kHz, added to the 48 kHz track) damages the code, versus AudioSeal used natively. | Printed table (recorded in `findings.md`) |
| `.venv-visqol/bin/python visqol_scores.py` | Scores how audible the AudioSeal mark is, next to MP3 as a reference. Needs `aseal/marked/` from `run_audioseal.py`. About 15 to 20 minutes. | `aseal/visqol_results.csv` |
| `python make_synthid_copies.py` | Makes the edited copies of 4 tracks for manual SynthID checks in Gemini. Skips files that already exist. | `synthid_copies/` and new rows in `synthid_results.csv` |

## What is in this repo

| File | Purpose |
| --- | --- |
| `attacks.py` | The edits, in one place, shared by the AudioSeal and SynthID scripts so both get identical settings |
| `run_audioseal.py` | AudioSeal embed, edit, detect loop |
| `shortcut_test.py` | Embedding-method check for AudioSeal |
| `make_synthid_copies.py` | Builds the files to upload to Gemini |
| `visqol_scores.py` | ViSQOL sound quality scores for AudioSeal |
| `attacks.md` | Every edit, its settings and rationale, and the SynthID manual protocol |
| `findings.md` | Results, what they show, and the limits a skeptic may raise |
| `synthid_results.csv` | SynthID results logged by hand from Gemini |
| `aseal/audioseal_results.csv` | AudioSeal results |
| `aseal/visqol_results.csv` | ViSQOL scores |
| `requirements.txt` | Exact Python package versions (`setup_repo.sh` regenerates it with `pip freeze`) |

## Status

| Watermark | Method | Owner | Status |
| --- | --- | --- | --- |
| SynthID (Google) | Manual, through the Gemini app | Griffin | All 36 edited files checked; control songs pending |
| AudioSeal (Meta) | `run_audioseal.py` | Griffin | Done (final run 2026-11-02) |
| ViSQOL quality scores | `visqol_scores.py`, for AudioSeal | Griffin | Done (2026-11-03) |
| AudioWMark | Needs a source build on Mac | Teammate | Not part of this fifth |
| WavMark, Perth, SilentCipher, others | To be scripted | Teammates | Not part of this fifth |

## Method rules

- SynthID and AudioSeal get the same 10 edits with the same settings (see `attacks.md`). AudioSeal also gets 2 extra edits for comparison with the Sony RAW-Bench paper.
- Edits are applied to copies. Clean originals are never modified.
- Every edit writes a WAV at the original sample rate, so file format and sample rate never act as a hidden second edit.
- Every AudioSeal run includes a control: the detector is run on unmarked originals to measure false positives. SynthID needs control songs without SynthID for the same reason.
