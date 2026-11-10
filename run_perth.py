#!/usr/bin/env python3
"""Downmix to mono BEFORE embedding a Perth watermark.

For each original lyria_*.wav:
1. Average left and right into mono.
2. Check the mono original as a control.
3. Embed Perth into that mono signal.
4. Apply each team attack independently.
5. Report unrounded detection scores.

Threshold 0.5 is an experiment setting.
Scores are not calibrated probabilities.
"""

import csv
import glob
import os
import shutil
import tempfile
from datetime import datetime

import numpy as np
import perth
import soundfile as sf

from attacks import ATTACKS, RAWBENCH_EXTRAS, apply_attack


THRESHOLD = 0.5
OUT = "perth_results_mono"


def detect(path, watermarker):
    audio, sr = sf.read(
        path, always_2d=True, dtype="float32"
    )

    if len(audio) == 0:
        raise ValueError("Empty audio.")
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains non-finite samples.")

    # Mono inputs stay mono. If an attack creates stereo,
    # downmix its output for detection and record the channels.
    mono = audio.mean(axis=1)

    score = watermarker.get_watermark(
        mono.copy(), sample_rate=sr, round=False
    )
    score = float(np.asarray(score).item())

    if not np.isfinite(score):
        raise ValueError("Non-finite detection score.")

    return score, audio.shape[1]


def embed_mono(src, mono_path, marked_path, watermarker):
    audio, sr = sf.read(
        src, always_2d=True, dtype="float32"
    )

    if len(audio) == 0:
        raise ValueError("Empty source audio.")
    if audio.shape[1] not in (1, 2):
        raise ValueError("Expected mono or stereo source.")
    if not np.isfinite(audio).all():
        raise ValueError("Source contains non-finite samples.")

    # Crucial difference: combine channels BEFORE embedding.
    mono = audio.mean(axis=1)

    # Save a float WAV to avoid introducing PCM quantization
    # into the mono control.
    sf.write(mono_path, mono, sr, subtype="FLOAT")

    marked = watermarker.apply_watermark(
        mono.copy(), watermark=None, sample_rate=sr
    )
    marked = np.asarray(marked).reshape(-1)

    if len(marked) == 0:
        raise ValueError("Perth returned empty audio.")
    if not np.isfinite(marked).all():
        raise ValueError("Perth returned non-finite samples.")

    # Experiment guardrail, not an official Perth tolerance.
    if abs(len(marked) - len(mono)) > int(0.1 * sr):
        raise ValueError(
            f"Length difference exceeds 0.1 seconds: "
            f"original={len(mono)}, marked={len(marked)}"
        )

    n = min(len(mono), len(marked))
    clipping_fraction = float(np.mean(np.abs(marked[:n]) > 1))

    sf.write(
        marked_path,
        np.clip(marked[:n], -1.0, 1.0),
        sr,
        subtype="PCM_24",
    )

    print(
        f"  Embedded mono audio: original={len(mono)}, "
        f"Perth output={len(marked)}, kept={n} samples",
        flush=True,
    )

    return {
        "source_channels": audio.shape[1],
        "source_frames": len(mono),
        "perth_output_frames": len(marked),
        "kept_frames": n,
        "clipping_fraction": clipping_fraction,
    }


def main():
    tracks = sorted(glob.glob("lyria_*.wav"))
    if not tracks:
        raise SystemExit(
            "No lyria_*.wav files in the current folder."
        )

    for tool in ("sox", "ffmpeg"):
        if shutil.which(tool) is None:
            raise SystemExit(f"Missing tool: {tool}")

    os.makedirs(OUT, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = tempfile.mkdtemp(
        prefix=f"run_{stamp}_", dir=OUT
    )

    mono_folder = os.path.join(run_folder, "mono_originals")
    marked_folder = os.path.join(run_folder, "marked")
    os.makedirs(mono_folder)
    os.makedirs(marked_folder)

    results = os.path.join(run_folder, "perth_mono_results.csv")

    fields = [
        "track", "attack", "detection_score", "detected",
        "threshold", "detector_input_channels",
        "source_channels", "source_frames",
        "perth_output_frames", "kept_frames",
        "clipping_fraction", "status", "error",
    ]

    attacks = list(dict.fromkeys(
        list(ATTACKS) + list(RAWBENCH_EXTRAS)
    ))

    print("Loading Perth Implicit...", flush=True)
    watermarker = perth.PerthImplicitWatermarker()

    with open(results, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        def record(track, attack, path=None, metadata=None,
                   error=None):
            row = {
                "track": track,
                "attack": attack,
                "threshold": THRESHOLD,
                "status": "ok",
                "error": "",
            }
            if metadata:
                row.update(metadata)

            try:
                if error is not None:
                    raise RuntimeError(error)

                score, channels = detect(path, watermarker)
                row["detection_score"] = repr(score)
                row["detected"] = score >= THRESHOLD
                row["detector_input_channels"] = channels

                print(
                    f"  {attack:<20} score={score:.8f} "
                    f" detected={score >= THRESHOLD}",
                    flush=True,
                )
            except Exception as exc:
                message = getattr(exc, "stderr", None) or str(exc)
                if isinstance(message, bytes):
                    message = message.decode(
                        "utf-8", errors="replace"
                    )
                row["status"] = "error"
                row["error"] = message.strip()
                print(
                    f"  {attack:<20} ERROR: {row['error']}",
                    flush=True,
                )

            writer.writerow(row)
            f.flush()

        for source in tracks:
            name = os.path.splitext(os.path.basename(source))[0]
            print(f"\n=== {name} ===", flush=True)

            mono_path = os.path.join(
                mono_folder, f"{name}_mono.wav"
            )
            marked_path = os.path.join(
                marked_folder, f"{name}_mono_perth.wav"
            )

            # Create and check the mono control independently
            # so embedding failures do not hide control results.
            try:
                audio, sr = sf.read(
                    source, always_2d=True, dtype="float32"
                )
                if (
                    len(audio) == 0
                    or audio.shape[1] not in (1, 2)
                    or not np.isfinite(audio).all()
                ):
                    raise ValueError("Invalid source audio.")

                sf.write(
                    mono_path, audio.mean(axis=1),
                    sr, subtype="FLOAT"
                )
                record(name, "control_unmarked", mono_path)
            except Exception as exc:
                record(name, "control_unmarked", error=str(exc))
                continue

            try:
                metadata = embed_mono(
                    source, mono_path, marked_path, watermarker
                )
            except Exception as exc:
                record(name, "embedding", error=str(exc))
                continue

            # Always check the saved watermark before attacks.
            record(
                name, "baseline_marked",
                marked_path, metadata
            )

            for attack in attacks:
                try:
                    with tempfile.TemporaryDirectory(
                        prefix="attack_", dir=run_folder
                    ) as tmp:
                        dst = os.path.join(tmp, "attacked.wav")
                        path = apply_attack(
                            attack, marked_path, dst, tmp
                        )
                        record(name, attack, path, metadata)
                except Exception as exc:
                    message = getattr(exc, "stderr", None) or str(exc)
                    if isinstance(message, bytes):
                        message = message.decode(
                            "utf-8", errors="replace"
                        )
                    record(
                        name, attack,
                        metadata=metadata, error=message
                    )

    print(f"\nResults saved to: {results}", flush=True)


if __name__ == "__main__":
    main()