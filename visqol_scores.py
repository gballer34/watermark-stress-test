#!/usr/bin/env python3
"""
ViSQOL quality scores: can you hear the AudioSeal watermark?

For each clean track, scores three versions against the original (on the
same 30-second excerpt, 0:30 to 1:00) using
ViSQOL's audio (music) mode. MOS-LQO runs from 1 (bad) to about 4.75
(indistinguishable from the original).
  identical: original vs itself. Should hit the top score; proves the tool works.
  audioseal: original vs the AudioSeal-watermarked copy. The real test.
  mp3128:    original vs a 128 kbps MP3 of it. A familiar "sounds fine to
             most listeners" reference point.

Why the Python port (visqol-python) instead of Google's C++ build: it installs
in one command so teammates can rerun it, and its published conformance tests
match the official C++ ViSQOL v3.3.3 in audio mode to within 0.0002 MOS.

Needs the watermarked files from run_audioseal.py (aseal/marked/).
Run from this folder, using the ViSQOL environment:
    .venv-visqol/bin/python visqol_scores.py
"""

import csv
import glob
import os

import numpy as np
import soundfile as sf
from visqol import VisqolApi

# Why: reuse the project's own MP3 edit so the reference matches mp3128 elsewhere.
from attacks import apply_attack

OUT = "aseal"
TMP = os.path.join(OUT, "tmp")
RESULTS = os.path.join(OUT, "visqol_results.csv")

# Why an excerpt: the Python port takes about 2.5 s per second of audio, so
# full tracks would take ~2 hours. ViSQOL scores short patches and averages
# them anyway, and AudioSeal's mark is spread evenly through the track.
# 0:30 to 1:00 is the same window as the clip30 edit.
EXCERPT_START_S = 30
EXCERPT_LEN_S = 30


def load_mono(path):
    """Why mono: ViSQOL compares single channels. Averaging left and right
    is stated explicitly here rather than left to the library."""
    x, sr = sf.read(path, always_2d=True, dtype="float64")
    start = EXCERPT_START_S * sr
    return x[start:start + EXCERPT_LEN_S * sr].mean(axis=1), sr


def score(api, ref_path, deg_path):
    ref, sr = load_mono(ref_path)
    deg, sr_deg = load_mono(deg_path)
    assert sr == sr_deg == 48000, "ViSQOL audio mode expects 48 kHz"
    # Why: MP3 decoding can shift lengths by a few samples.
    n = min(len(ref), len(deg))
    return api.measure_from_arrays(ref[:n], deg[:n], sample_rate=sr).moslqo


def main():
    os.makedirs(TMP, exist_ok=True)
    api = VisqolApi()
    api.create(mode="audio")   # Why: audio mode is ViSQOL's music setting.

    with open(RESULTS, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track", "comparison", "moslqo"])
        print(f"\n{'song':<18}{'identical':>10}{'audioseal':>11}{'mp3128':>9}")
        for clean in sorted(glob.glob("lyria_*.wav")):
            name = os.path.splitext(clean)[0]
            marked = os.path.join(OUT, "marked", f"{name}_aseal.wav")
            mp3 = apply_attack("mp3128", clean, os.path.join(TMP, f"{name}_mp3.wav"), TMP)

            scores = {
                "identical": score(api, clean, clean),
                "audioseal": score(api, clean, marked),
                "mp3128": score(api, clean, mp3),
            }
            os.remove(mp3)   # Why: temp file; the score is what we keep.
            for k, v in scores.items():
                w.writerow([name, k, f"{v:.3f}"])
            f.flush()
            print(f"{name:<18}{scores['identical']:>10.2f}"
                  f"{scores['audioseal']:>11.2f}{scores['mp3128']:>9.2f}")

    print(f"\nDone. Results saved to {RESULTS}")


if __name__ == "__main__":
    main()
