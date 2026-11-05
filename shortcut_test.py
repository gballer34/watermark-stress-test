#!/usr/bin/env python3
"""
Shortcut test: is our embedding method damaging the AudioSeal code?

run_audioseal.py uses a shortcut: compute the watermark at 16 kHz, upsample
only the watermark, and add it to the full-quality 48 kHz song. In the first
run, some songs read back only 15 of 16 bits with NO edit at all. This test
asks whether the shortcut causes that, or AudioSeal does.

For each song, with no edits, it compares:
  native:   the song at 16 kHz, watermarked and checked the standard way
            (AudioSeal's best case)
  shortcut: exactly what run_audioseal.py does, including saving the file
It also counts clipped samples: places where adding the watermark pushed
the audio past the loudness ceiling and got flattened.

Run from this folder:
    python shortcut_test.py
"""

import glob
import os

import numpy as np
import soundfile as sf
import torch
from audioseal import AudioSeal

# Why: reuse the main script's own functions, so "shortcut" here is
# exactly the method the real run uses, not a re-creation of it.
from run_audioseal import MODEL_SR, TMP, detect, embed, payload_bits, resample


def bit_acc(bits, got):
    return 100 * sum(a == b for a, b in zip(bits, got)) / 16


def main():
    os.makedirs(TMP, exist_ok=True)
    generator = AudioSeal.load_generator("audioseal_wm_16bits").eval()
    detector = AudioSeal.load_detector("audioseal_detector_16bits").eval()

    print(f"\n{'song':<18}{'native':>8}{'shortcut':>10}{'clipped samples':>17}")
    for clean in sorted(glob.glob("lyria_*.wav")):
        name = os.path.splitext(clean)[0]
        bits, _ = payload_bits(name)
        msg = torch.tensor([bits])

        x, sr = sf.read(clean, always_2d=True, dtype="float32")
        t = torch.from_numpy(resample(x.mean(axis=1), sr, MODEL_SR)).view(1, 1, -1)
        with torch.no_grad():
            wm = generator.get_watermark(t, sample_rate=MODEL_SR, message=msg)
            # Native: add the watermark at 16 kHz and detect without saving.
            _, got = detector.detect_watermark(t + wm, sample_rate=MODEL_SR)
        native = bit_acc(bits, [int(b > 0.5) for b in got.squeeze().tolist()])

        # Why: count how many samples go past +/-1.0 (the ceiling) when the
        # upsampled watermark is added, using the same math as embed().
        wm_full = resample(wm.squeeze().numpy(), MODEL_SR, sr)
        wm_full = np.pad(wm_full, (0, max(0, len(x) - len(wm_full))))[:len(x)]
        clipped = int(np.sum(np.abs(x + wm_full[:, None]) > 1.0))

        # Shortcut: the real embed() and detect() from run_audioseal.py.
        path = os.path.join(TMP, f"{name}_shortcut.wav")
        embed(clean, path, bits, generator)
        _, got = detect(path, detector)
        shortcut = bit_acc(bits, got)
        os.remove(path)   # Why: temp file; the numbers are what we keep.

        print(f"{name:<18}{native:>7.0f}%{shortcut:>9.0f}%{clipped:>17,}")


if __name__ == "__main__":
    main()
