# Findings

Running record of results for the RIAA presentation. Raw data lives in `aseal/audioseal_results.csv` (AudioSeal) and `synthid_results.csv` (SynthID).

## AudioSeal does not always carry its code intact, even with no edits

**Date run:** 2026-11-01
**Script:** `shortcut_test.py` (run with `python shortcut_test.py`)

**Question:** In the first AudioSeal run, 3 of 8 songs read back only 15 of 16 code bits with no edit at all. Was that caused by our embedding method (computing the watermark at 16 kHz and adding it to the 48 kHz song), or by AudioSeal itself?

**Method:** Each song was watermarked two ways, then decoded with no edits:
- **Native:** the song converted to 16 kHz, watermarked and decoded the standard way. This is AudioSeal's best case.
- **Shortcut:** the exact method used in `run_audioseal.py`, including saving the file.

Clipped samples count where adding the watermark pushed the audio past full scale.

| Song | Native | Shortcut | Clipped samples (of ~8 million) |
| --- | --- | --- | --- |
| lyria_ambient_1 | 94% | 94% | 9 |
| lyria_ambient_2 | 94% | 94% | 0 |
| lyria_dense_1 | 100% | 100% | 2 |
| lyria_dense_2 | 100% | 100% | 0 |
| lyria_pop_1 | 100% | 100% | 0 |
| lyria_pop_2 | 94% | 94% | 1 |
| lyria_sparse_1 | 100% | 100% | 0 |
| lyria_sparse_2 | 100% | 100% | 0 |

(94% = 15 of 16 bits correct, so the exact code was not recovered.)

**What it shows:**
- Our shortcut is not the cause: native and shortcut results are identical on every song.
- Clipping is not the cause: it is negligible everywhere, and `lyria_ambient_2` fails with zero clipped samples.
- **AudioSeal itself failed to return the exact 16-bit code on 3 of 8 unedited Lyria tracks.** For a law that requires the latent disclosure to carry specific information (California SB 942 / AB 853), a mark that cannot reliably deliver its payload on untouched audio is a problem before any removal attempt.

**Limits a skeptic may raise:**
- 8 songs is a small sample; "3 of 8" shows the problem exists but is not a reliable failure rate.
- One fixed code per song. A different code might fare differently on the same song.
- AudioSeal was trained mainly on speech; these results are specific to music, and to AI-generated music in particular.

## AudioSeal: final run (8 songs x 12 edits)

**Date run:** 2026-11-02
**Script:** `run_audioseal.py` (edits defined in `attacks.py`)
**Raw data:** `aseal/audioseal_results.csv`

"Detected" means AudioSeal's detection score was above 0.5. "Exact code" means all 16 bits matched. Random guessing averages about 50% of bits.

| Edit | Detected | Exact code | Avg bits correct |
| --- | --- | --- | --- |
| Control (no watermark) | 0/8 | n/a | 45% |
| None | 8/8 | 5/8 | 98% |
| MP3 128 kbps | 8/8 | 5/8 | 98% |
| AAC 128 kbps | 8/8 | 4/8 | 97% |
| Volume -6 dB | 8/8 | 5/8 | 97% |
| Stem split (vocals removed, Demucs) | 8/8 | 5/8 | 97% |
| Noise at 20 dB SNR | 8/8 | 3/8 | 95% |
| 30-second clip | 8/8 | 3/8 | 93% |
| Speed up 10% (tempo and pitch) | 0/8 | 0/8 | 45% |
| Slowed 15% + reverb | 0/8 | 0/8 | 44% |
| Pitch +2 semitones | 0/8 | 0/8 | 45% |
| Time stretch 5% (tempo only) | 0/8 | 0/8 | 68% |
| Polarity flip | 7/8 | 0/8 | 16% |

**What it shows:**
- **No false positives:** the detector never reported a watermark in unmarked audio.
- **Any change to timing or pitch removes AudioSeal**, including a barely noticeable 5% tempo change.
- **Compression, volume, noise, clipping and stem separation leave the mark detectable**, but the exact code is often lost. It was lost on 3 of 8 songs even with no edit (see the shortcut test above).
- **A polarity flip, which is completely inaudible, inverts the code.** Detection still triggers (borderline scores of 0.47 to 0.82 instead of about 1.00), but the bits come back flipped: 16% correct on average, 0% on `lyria_sparse_2`. The mark says "AI-generated" while carrying the wrong ID. A detector could try reading the code both ways, but AudioSeal's standard detector does not.

**Limits a skeptic may raise:**
- 8 songs; counts like "3/8" show behaviour, not reliable rates.
- Each edit used one setting. Milder or stronger versions may behave differently.
- Detection threshold of 0.5 is AudioSeal's default, not a tuned value.

## SynthID: results (Gemini app)

**Songs:** `lyria_dense_1`, `lyria_sparse_1`, `lyria_pop_1`, `lyria_ambient_1` (one per style)
**Raw data:** `synthid_results.csv` (prompt v2, one new chat per file)
**Status:** all 36 edited files checked, 2026-11-01 to 2026-11-03. Non-Google AI control done; human-made control pending.

**Control (non-Google AI):** a Suno track generated from the identical pop prompt ("Take the Long Way", downloaded unedited as M4A, since WAV needs a paid plan) returned **SynthID not detected**. Gemini can answer "no", and it is detecting Google's watermark specifically, not AI-generated pop in general.

- **SynthID was detected after every edit: 35 of 36 full-length, 1 partial**, including the speed, pitch, slowed + reverb and polarity edits that removed or broke AudioSeal.
- **One partial result:** `sparse_1_noise20db` was detected from 0:00 to 2:42 of a 2:46 file. The final 4 seconds of the clean song are near-silent (about -60 dB), so after adding noise they are effectively only noise. This also shows Gemini's time ranges reflect real section-by-section detection, not just the file length.
- **C2PA:** "present but invalid" on the original downloads, "not present" on every edited copy. Our tools write a new file containing only the audio, so this shows C2PA metadata does not survive re-saving, not that a specific edit removes it.
- **C2PA on Suno:** the unedited Suno file was also "present but invalid", the same verdict as Google's own files. Two providers getting the identical verdict raises the possibility that Gemini cannot validate audio C2PA at all. To be confirmed with the official `c2patool` before claiming the manifests themselves are invalid.

**Limits a skeptic may raise:**
- Gemini reports detection only. SynthID's payload cannot be read, so "detected" is not comparable to AudioSeal's exact-code result.
- Only one non-SynthID control so far, and it was M4A while the test files were WAV. A human-made control (and optionally the Suno file converted to WAV) would strengthen this.
- Gemini is a black box whose behaviour Google may change at any time; results are tied to the check dates logged.

## ViSQOL: can you hear the AudioSeal watermark?

**Date run:** 2026-11-03
**Script:** `visqol_scores.py` (run with `.venv-visqol/bin/python visqol_scores.py`)
**Raw data:** `aseal/visqol_results.csv`

**Question:** AudioSeal changes the audio to hide its mark. Does that change make the music sound worse?

**Method:** ViSQOL (visqol-python 3.8.0, audio/music mode, 48 kHz) compares a copy to the original and predicts a listener score (MOS-LQO). In this mode the top score is about 4.73 to 4.75, not 5. Each song was scored three ways on the same 30-second excerpt (0:30 to 1:00), with left and right averaged to mono:
- **Identical:** original vs itself. Shows the tool's ceiling.
- **AudioSeal:** original vs the watermarked copy from `run_audioseal.py`.
- **MP3 128 kbps:** original vs an MP3 of it. A familiar "sounds fine to most people" reference point.

| Song | Identical | AudioSeal | MP3 128 kbps |
| --- | --- | --- | --- |
| lyria_ambient_1 | 4.732 | 4.727 | 3.961 |
| lyria_ambient_2 | 4.732 | 4.727 | 4.136 |
| lyria_dense_1 | 4.732 | 4.728 | 4.488 |
| lyria_dense_2 | 4.732 | 4.729 | 4.571 |
| lyria_pop_1 | 4.732 | 4.728 | 4.382 |
| lyria_pop_2 | 4.732 | 4.728 | 4.428 |
| lyria_sparse_1 | 4.732 | 4.723 | 4.634 |
| lyria_sparse_2 | 4.732 | 4.728 | 4.729 |

**What it shows:**
- **The AudioSeal watermark is predicted to be inaudible.** On every song it scores within 0.01 of the original compared with itself.
- **The tool can tell copies apart:** MP3 scored lower on 7 of 8 songs, by up to 0.77 (ambient_1). So the near-perfect AudioSeal scores are not the tool failing to notice changes.
- **MP3 hurt the songs differently.** Ambient tracks (rain textures, long reverb, lots of hiss-like detail) lost the most. Solo piano lost almost nothing, likely because it has few sounds at once and little high-frequency detail for MP3 to throw away.
- **Takeaway for the talk:** AudioSeal's weakness is not sound quality. It is that timing and pitch edits remove it, and its code is not always carried intact (see above).

**Limits a skeptic may raise:**
- ViSQOL is a computer prediction of listener opinion, not a listening test with people.
- The Python port is unofficial (one author). Its published tests match Google's official C++ ViSQOL v3.3.3 to within 0.0002 in audio mode, but it is not Google's release.
- One 30-second excerpt per song, in mono. Stereo-only differences, or other parts of the song, are not measured.
- AudioSeal only. SynthID cannot be scored this way because there is no un-watermarked original of a Lyria song to compare against.
- Near the top of the scale, ViSQOL may not separate "inaudible" from "barely audible" well, so a 0.005 difference should not be read as meaningful either way.
