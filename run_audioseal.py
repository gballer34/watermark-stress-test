#!/usr/bin/env python3
"""
AudioSeal robustness test for the RIAA / Duke watermarking project.

For every clean lyria_*.wav in this folder, it:
  1. Embeds an AudioSeal watermark carrying a 16-bit payload
  2. Applies each attack (same settings as the SynthID/Gemini tests)
  3. Runs the AudioSeal detector and reads the payload back
  4. Logs detection probability, whether the exact payload survived,
     and what share of the 16 bits survived

Why both detection AND payload: California's AI Transparency Act requires
the latent disclosure to carry specific information (unique ID, system
name/version, date). A mark that is detected but garbled fails that
purpose, so we measure the payload, not just presence.

Run from the folder holding your clean tracks:
    python run_audioseal.py
"""

import csv
import datetime
import glob
import hashlib
import math
import os
import subprocess
import sys

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from audioseal import AudioSeal

# Why: the edits live in attacks.py, shared with make_synthid_copies.py,
# so SynthID and AudioSeal get identical edits with identical settings.
from attacks import ATTACKS, RAWBENCH_EXTRAS, apply_attack

MODEL_SR = 16000          # Why: AudioSeal's model was trained on 16 kHz audio.
OUT = "aseal"
TMP = os.path.join(OUT, "tmp")
RESULTS = os.path.join(OUT, "audioseal_results.csv")


def resample(x, sr_in, sr_out):
    """Why: polyphase resampling is high quality and needs no extra audio libraries."""
    if sr_in == sr_out:
        return x
    g = math.gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def payload_bits(name):
    """
    Why: build the payload from the fields California requires (provider,
    system, date, unique ID), then hash it down to AudioSeal's 16 bits.
    16 bits can only index a lookup database (65,536 values); it cannot
    carry the required fields directly. That capacity limit is itself a
    finding worth reporting.
    """
    src = (f"Provider:DukeRIAATest|System:audioseal|"
           f"Date:2026-11-01|ID:{name}")
    v = int.from_bytes(hashlib.sha256(src.encode()).digest()[:2], "big")
    return [(v >> (15 - i)) & 1 for i in range(16)], src


def embed(clean_path, out_path, bits, generator):
    """
    Why this design: AudioSeal works at 16 kHz, but our tracks are 48 kHz.
    Downsampling the whole song would wreck its quality, so we compute the
    watermark at 16 kHz, upsample just the watermark, and add it to the
    full-quality original. Returns SNR as a quick imperceptibility check
    (higher = quieter watermark; ViSQOL comes later for a perceptual score).
    """
    x, sr = sf.read(clean_path, always_2d=True, dtype="float32")
    mono = x.mean(axis=1)
    m16 = resample(mono, sr, MODEL_SR)
    t = torch.from_numpy(m16).view(1, 1, -1)       # (batch, channels, samples)
    msg = torch.tensor([bits])
    with torch.no_grad():
        wm = generator.get_watermark(t, sample_rate=MODEL_SR, message=msg)
    wm_full = resample(wm.squeeze().numpy(), MODEL_SR, sr)
    # Why: resampling can change length by a few samples; match exactly.
    wm_full = np.pad(wm_full, (0, max(0, len(mono) - len(wm_full))))[:len(mono)]
    marked = np.clip(x + wm_full[:, None], -1.0, 1.0)
    sf.write(out_path, marked, sr, subtype="PCM_24")
    noise_power = np.sum((marked - x) ** 2)
    return 10 * math.log10(np.sum(x ** 2) / noise_power) if noise_power > 0 else float("inf")


def detect(path, detector):
    """Returns (probability the audio is watermarked, decoded 16 bits)."""
    x, sr = sf.read(path, always_2d=True, dtype="float32")
    m16 = resample(x.mean(axis=1), sr, MODEL_SR)
    t = torch.from_numpy(m16).view(1, 1, -1)
    with torch.no_grad():
        prob, msg = detector.detect_watermark(t, sample_rate=MODEL_SR)
    bits = [int(b > 0.5) for b in msg.squeeze().tolist()]
    return float(prob), bits


def main():
    tracks = sorted(glob.glob("lyria_*.wav"))
    if not tracks:
        sys.exit("No lyria_*.wav files here. cd into your tracks folder first.")
    for tool in ("sox", "ffmpeg"):
        if subprocess.run(["which", tool], capture_output=True).returncode != 0:
            sys.exit(f"Missing tool: {tool}")

    os.makedirs(os.path.join(OUT, "marked"), exist_ok=True)
    os.makedirs(TMP, exist_ok=True)

    print("Loading AudioSeal models (first run downloads them)...")
    generator = AudioSeal.load_generator("audioseal_wm_16bits")
    detector = AudioSeal.load_detector("audioseal_detector_16bits")
    generator.eval()
    detector.eval()

    with open(RESULTS, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track", "attack", "detection_prob", "detected",
                    "payload_exact", "bit_accuracy_pct", "embed_snr_db",
                    "payload_source_string"])

        for clean in tracks:
            name = os.path.splitext(clean)[0]
            print(f"\n=== {name} ===")
            bits, src = payload_bits(name)

            # Why: a control. Running the detector on the UNmarked original
            # shows the false-positive baseline. A detector that "finds" a
            # mark in clean audio can't be trusted for anything else.
            prob, got = detect(clean, detector)
            # Why: score the code read from unmarked audio too. This is the
            # pure-guessing baseline (about 50%) to compare attacks against.
            matches = sum(a == b for a, b in zip(bits, got))
            acc = 100 * matches / 16
            w.writerow([name, "control_unmarked", f"{prob:.3f}", prob > 0.5,
                        matches == 16, f"{acc:.1f}", "n/a", src])
            print(f"  {'control_unmarked':<17} prob={prob:.3f}  bits={acc:.0f}%  (should be low / near 50%)")

            marked = os.path.join(OUT, "marked", f"{name}_aseal.wav")
            snr = embed(clean, marked, bits, generator)

            for attack in ATTACKS + RAWBENCH_EXTRAS:
                dst = os.path.join(TMP, f"{name}_{attack}.wav")
                try:
                    path = apply_attack(attack, marked, dst, TMP)
                    prob, got = detect(path, detector)
                except (subprocess.CalledProcessError, FileNotFoundError) as e:
                    # Why: show the tool's last error line so a failed attack can be diagnosed.
                    err = (getattr(e, "stderr", None) or str(e)).strip()
                    reason = err.splitlines()[-1] if err else "no error text"
                    print(f"  {attack:<17} ATTACK FAILED: {reason}")
                    continue
                matches = sum(a == b for a, b in zip(bits, got))
                acc = 100 * matches / 16
                w.writerow([name, attack, f"{prob:.3f}", prob > 0.5,
                            matches == 16, f"{acc:.1f}", f"{snr:.1f}", src])
                print(f"  {attack:<17} prob={prob:.3f}  payload exact={matches == 16!s:<5}"
                      f"  bits={acc:.0f}%")
                # Why: delete each attacked file once measured, so 64 large
                # WAVs don't fill the disk. The CSV keeps the results.
                if path != marked and os.path.exists(path):
                    os.remove(path)
            f.flush()

    print(f"\nDone. Results saved to {RESULTS}")


if __name__ == "__main__":
    main()
