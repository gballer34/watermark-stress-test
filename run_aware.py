#!/usr/bin/env python3
"""
Model profile: AWARE
Payload: one reproducible 20-bit message per track.
Detection threshold: 0.5, an explicit experiment setting.

Scores are recorded as model outputs, not validated probabilities.

"""

import csv
import hashlib
import json
import shutil
import tempfile
from importlib.metadata import version
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from aware.utils.models import load
from aware.service import embed_watermark, detect_watermark
from attacks import ATTACKS, RAWBENCH_EXTRAS, apply_attack


ROOT = Path(__file__).resolve().parent
SR = 16000
THRESHOLD = 0.5
PROFILE = "AWARE"


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def read_mono(path):
    audio, _ = librosa.load(
        str(path), sr=SR, mono=True
    )
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("Audio is empty or contains invalid samples.")
    return np.ascontiguousarray(audio, dtype=np.float32)


def payload_for(track):
    # Deterministic per-track payload, independent of file ordering.
    seed = int.from_bytes(
        hashlib.sha256(track.encode("utf-8")).digest()[:8],
        "little",
    )
    return np.random.default_rng(seed).integers(
        0, 2, size=20, dtype=np.int32
    )


def detect(path, detector, expected):
    audio = read_mono(path)
    pattern, confidence = detect_watermark(
        audio, SR, detector
    )

    confidence = as_numpy(confidence)
    if confidence.size != 1:
        raise ValueError(
            f"Expected one confidence value, got {confidence.shape}."
        )
    score = float(confidence.item())
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError(f"Invalid confidence: {score}")

    bits = as_numpy(pattern).reshape(-1)
    if bits.size != 20 or not np.isin(bits, [0, 1]).all():
        raise ValueError(
            "Expected 20 decoded binary bits; "
            f"received shape {bits.shape}."
        )

    bits = bits.astype(np.int32)
    accuracy = float(np.mean(bits == expected) * 100)

    return {
        "detection_score": score,
        "detected": score >= THRESHOLD,
        "recovered_bits": "".join(map(str, bits.tolist())),
        "payload_exact": bool(np.array_equal(bits, expected)),
        "bit_accuracy_pct": accuracy,
    }


def main():
    tracks = sorted(ROOT.glob("lyria_*.wav"))
    if not tracks:
        raise SystemExit(
            "No lyria_*.wav source files beside this script."
        )

    for tool in ("sox", "ffmpeg"):
        if shutil.which(tool) is None:
            raise SystemExit(f"Missing tool: {tool}")

    output_root = ROOT / "aware_results_mono"
    output_root.mkdir(exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="run_", dir=output_root))

    mono_dir = run / "mono_originals"
    marked_dir = run / "marked"
    mono_dir.mkdir()
    marked_dir.mkdir()

    attacks = list(dict.fromkeys(
        list(ATTACKS) + list(RAWBENCH_EXTRAS)
    ))

    # Record configuration and the exact attack code used.
    shutil.copy2(ROOT / "attacks.py", run / "attacks_used.py")
    shutil.copy2(Path(__file__), run / "runner_used.py")
    (run / "settings.json").write_text(
        json.dumps({
            "aware_version": version("aware"),
            "profile": PROFILE,
            "sample_rate": SR,
            "threshold": THRESHOLD,
            "channel_handling": "downmix before embedding",
            "saved_watermark_format": "PCM_24",
            "attacks": attacks,
            "source_tracks": [p.name for p in tracks],
        }, indent=2),
        encoding="utf-8",
    )

    print("Loading AWARE models...", flush=True)
    embedder, detector = load(name=PROFILE)

    results = run / "aware_results.csv"
    fields = [
        "track", "attack", "detection_score", "detected",
        "threshold", "payload_bits", "recovered_bits",
        "payload_exact", "bit_accuracy_pct",
        "input_frames_16khz", "output_frames_16khz",
        "status", "error",
    ]

    with results.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        def record(name, attack, payload, path=None,
                   metadata=None, error=None):
            row = {
                "track": name,
                "attack": attack,
                "threshold": THRESHOLD,
                "payload_bits": "".join(
                    map(str, payload.tolist())
                ),
                "status": "ok",
                "error": "",
            }
            if metadata:
                row.update(metadata)

            try:
                if error is not None:
                    raise RuntimeError(error)

                row.update(detect(path, detector, payload))

                # No message was embedded in the control.
                # Bit accuracy there is only a chance-reference check.
                if attack == "control_unmarked":
                    row["payload_exact"] = "n/a"

                print(
                    f"  {attack:<20} "
                    f"score={row['detection_score']:.8f} "
                    f"detected={row['detected']} "
                    f"bits={row['bit_accuracy_pct']:.1f}%",
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
            handle.flush()

        for source in tracks:
            name = source.stem
            payload = payload_for(name)
            mono_path = mono_dir / f"{name}_mono16k.wav"
            marked_path = marked_dir / f"{name}_aware.wav"
            print(f"\n=== {name} ===", flush=True)

            try:
                mono = read_mono(source)
                sf.write(mono_path, mono, SR, subtype="FLOAT")
            except Exception as exc:
                record(
                    name, "control_unmarked", payload,
                    error=str(exc),
                )
                continue

            record(name, "control_unmarked", payload, mono_path)

            try:
                marked = embed_watermark(
                    mono.copy(), SR, payload.copy(), embedder
                )
                marked = as_numpy(marked).squeeze()

                if (
                    marked.ndim != 1
                    or marked.size == 0
                    or not np.isfinite(marked).all()
                ):
                    raise ValueError("Invalid embedded audio.")

                metadata = {
                    "input_frames_16khz": len(mono),
                    "output_frames_16khz": len(marked),
                }

                # Flag unexpected length/range changes rather than
                # silently trimming or clipping the AWARE output.
                if abs(len(marked) - len(mono)) > int(0.1 * SR):
                     raise ValueError(
                        f"Length difference exceeds 0.1 seconds: "
                        f"{len(mono)} -> {len(marked)}"
                )

                n = min(len(mono), len(marked))
                print(
                    f"  Embedding lengths: original={len(mono)}, "
                    f"AWARE output={len(marked)}, keeping={n} samples",
                    flush=True,
                )
                marked = marked[:n]
                if np.max(np.abs(marked)) > 1:
                    raise ValueError(
                        "AWARE output exceeds WAV full scale; "
                        "inspect before choosing normalization."
                    )

                sf.write(
                    marked_path, marked, SR, subtype="PCM_24"
                )
            except Exception as exc:
                record(
                    name, "embedding", payload, error=str(exc)
                )
                continue

            record(
                name, "baseline_marked", payload,
                marked_path, metadata,
            )

            for attack in attacks:
                try:
                    with tempfile.TemporaryDirectory(
                        prefix="attack_", dir=run
                    ) as tmp:
                        dst = str(Path(tmp) / "attacked.wav")
                        path = apply_attack(
                            attack, str(marked_path), dst, tmp
                        )
                        record(
                            name, attack, payload, path, metadata
                        )
                except Exception as exc:
                    message = getattr(exc, "stderr", None) or str(exc)
                    if isinstance(message, bytes):
                        message = message.decode(
                            "utf-8", errors="replace"
                        )
                    record(
                        name, attack, payload,
                        metadata=metadata, error=message,
                    )

    print(f"\nResults: {results}", flush=True)


if __name__ == "__main__":
    main()