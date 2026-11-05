# Edits ("Attacks") and Test Protocol

Each edit mimics something that happens to real tracks after release. All edits are implemented in `attacks.py`, which both `run_audioseal.py` and `make_synthid_copies.py` import, so SynthID and AudioSeal get exactly the same processing.

Every edit takes a WAV and writes a WAV at the input's sample rate (our tracks are 48 kHz). Compressed formats are decoded back to WAV, which is what a player does before audio is heard.

## How this list was chosen

An earlier 8-edit list was picked without reference to a benchmark. It was revised on 2026-11-01 against the Sony RAW-Bench paper (Özer et al., Interspeech 2025, https://arxiv.org/abs/2505.19663; code at https://github.com/SonyResearch/raw_bench):

- **RAW-Bench is not run directly.** It cannot test SynthID (no callable detector), it already published AudioSeal results across 20 edits, and its evaluation code was listed as unreleased at the time.
- **Settings are borrowed from it where edits overlap** (noise SNR, gain range, MP3, AAC, time stretch, polarity), so our AudioSeal numbers can be compared with the paper's as a check on our setup.
- **Music-specific edits RAW-Bench lacks are added:** pitch shift, "slowed + reverb", stem separation, and short-form clips.

## Shared edits (SynthID and AudioSeal)

| ID | Edit | Real-world analogue | Implementation |
| --- | --- | --- | --- |
| none | No change | Baseline | Watermarked file as is |
| mp3128 | MP3 at 128 kbps, decoded back to WAV | Web downloads and uploads | `ffmpeg -i IN.wav -b:a 128k tmp.mp3`, then `ffmpeg -i tmp.mp3 OUT.wav` |
| speedup10 | Speed +10% (tempo and pitch) | "Sped up" short-form edits | `sox IN.wav OUT.wav speed 1.1` |
| slowreverb | Speed -15% plus reverb | "Slowed + reverb" fan edits | `sox IN.wav OUT.wav speed 0.85 reverb` |
| pitch2 | Pitch +2 semitones, tempo unchanged | Key changes in remixes | `sox IN.wav OUT.wav pitch 200` |
| noise20db | Gaussian white noise at 20 dB SNR | Background hiss, low-quality capture | numpy: noise power = song's mean power / 100, fixed seed 1234 |
| clip30 | 30-second excerpt from 0:30 | Short-form platforms play excerpts | `sox IN.wav OUT.wav trim 30 30` |
| gain6db | Volume halved (-6 dB) | A platform turning a loud track down | numpy: samples x 0.5 |
| stemsplit | Vocals removed (Demucs htdemucs, all other stems kept) | Karaoke and instrumental versions | `demucs --two-stems=vocals -n htdemucs`, keep `no_vocals.wav`, `sox ... rate 48000` |
| polarity | Waveform inverted | Happens by accident in mixing; inaudible | numpy: samples x -1 |

## AudioSeal-only edits (RAW-Bench comparison)

These run only on AudioSeal so they cost no Gemini checks.

| ID | Edit | Why | Implementation |
| --- | --- | --- | --- |
| timestretch5 | Tempo +5%, pitch unchanged | RAW-Bench's mildest stretch. With pitch2 (pitch only), shows whether timing or pitch breaks a mark | `sox IN.wav OUT.wav tempo 1.05` |
| aac128 | AAC at 128 kbps, decoded back to WAV | The format YouTube, Apple Music and Instagram use | `ffmpeg -i IN.wav -c:a aac -b:a 128k tmp.m4a`, then decode to WAV |

## Notes on settings

- **Why numpy for noise, gain and polarity:** SNR has to be measured against the song's own power, and SoX would add dither noise on gain changes, which is a hidden second edit.
- **noise20db:** SNR is measured over the whole track, so quiet passages get relatively more noise. In near-silent sections the result is essentially noise only.
- **stemsplit:** Demucs rebuilds the track from separated stems. On instrumental tracks there are no vocals to remove, so it mainly tests that rebuild. Demucs outputs 44.1 kHz, so the result is converted back to 48 kHz.
- **C2PA:** every edit writes a new file containing only audio, so C2PA metadata never survives. That reflects re-saving, not the specific edit.
- **Bit accuracy:** random guessing scores about 50% on a 16-bit code. Treat results near 50% as total payload loss. Results well below 50% mean the bits are systematically inverted (see polarity in `findings.md`).
- **Dropped from the earlier list:** `loudnorm14` (it also resampled to 44.1 kHz, a hidden second edit, and single-pass loudnorm compresses dynamics rather than applying the static gain most platforms use), the old `noise` setting (`sox -m` halved the music too), and a manual phone re-recording test (not repeatable enough for this round).

## SynthID manual protocol (Gemini app)

SynthID has no public API, so it is tested by hand at gemini.google.com, signed in with a personal Google account.

- **Files:** upload only from `synthid_copies/`. 4 tracks (`dense_1`, `sparse_1`, `pop_1`, `ambient_1`, one per style) x 9 edits = 36 files. The clean tracks were checked separately.
- **Limit:** about 10 audio checks per rolling 24 hours per account.
- **One new chat per file**, so earlier answers cannot influence later ones.
- **Prompt (v2, use this exact wording every time):**

> Check this audio file for AI provenance signals and report two results separately:
> 1. SynthID watermark: detected or not, and the time range if detected.
> 2. C2PA / Content Credentials metadata: present or not, and whether it is valid.
> Base your answer only on these detection results, not on how the audio sounds.

- **Log every check in `synthid_results.csv`:** date, detected (yes / no / partial), time range, C2PA status, and Gemini's full answer.
- **Partial time ranges matter:** they show a mark degrading in part of the track rather than simply surviving or failing.
- **Controls:** check at least one human-made track and, if available, one non-Google AI track (for example Suno). Gemini should report "not detected" on both. Without them there is no evidence Gemini can say no.

## Generation prompts (Flow Music, Lyria 3.5)

| Track set | Prompt |
| --- | --- |
| dense | berlin techno, around 132 bpm. dark and hypnotic, like 4am in a warehouse. kick should be driving with a long rumbling tail in the low end. keep it minimal and stripped back, metallic hats, and an acid line that repeats and slowly mutates over the track. no big drop, just a long build that keeps pulling you in. instrumental, no vocals. |
| sparse | Slow, quiet solo piano piece, 70 BPM, gentle and reflective, lots of space between notes, soft sustain pedal, intimate close-mic recording. Instrumental, no vocals. |
| pop | Upbeat indie pop song, 110 BPM, clear female lead vocal singing about a summer road trip, acoustic guitar, bass, live drums, catchy chorus. |
| ambient | Slow evolving ambient soundscape, warm analog synth pads, soft field-recording textures of rain, no drums, long reverb tails, calm and spacious. Instrumental, no vocals. |

Each prompt produced two takes (`_1` and `_2`).
