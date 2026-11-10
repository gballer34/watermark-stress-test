import argparse
import csv
import hashlib
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import wavmark

from attacks import ATTACKS, apply_attack

ROOT = Path(__file__).resolve().parent


def resample(audio, source_rate, target_rate):
    if source_rate == target_rate:
        return np.asarray(audio, dtype=np.float32)

    result = subprocess.run(
        [
            "ffmpeg", "-v", "error",
            "-f", "f32le", "-ar", str(source_rate), "-ac", "1",
            "-i", "pipe:0",
            "-f", "f32le", "-ar", str(target_rate), "-ac", "1",
            "pipe:1",
        ],
        input=np.asarray(audio, dtype="<f4").tobytes(),
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace"))
    return np.frombuffer(result.stdout, dtype="<f4").copy()


def read_for_decoder(path):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    return resample(audio.mean(axis=1), rate, 16000)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", nargs="+", required=True)
    args = parser.parse_args()

    tracks = [ROOT / name for name in args.track]
    for track in tracks:
        if not track.is_file():
            raise FileNotFoundError(track)

    run = ROOT / "wavmark" / "runs" / datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    run.mkdir(parents=True)
    (run / "attacks_used.py").write_bytes(
        (ROOT / "attacks.py").read_bytes()
    )

    settings = {
        "tracks": [p.name for p in tracks],
        "attacks": ATTACKS,
        "payload_bits": 16,
        "embedding_device": "cpu",
        "decoding_device": "mps",
        "decode_batch_size": 10,
        "seed": 42,
        "torch_version": torch.__version__,
        "audio_method": (
            "Embed in 16kHz mono; resample watermark difference "
            "and add to both original stereo channels. Save PCM24. "
            "Apply shared edits at original sample rate. "
            "Downmix and resample to 16kHz for decoding."
        ),
        "attacks_sha256": hashlib.sha256(
            (ROOT / "attacks.py").read_bytes()
        ).hexdigest(),
    }
    (run / "settings.json").write_text(json.dumps(settings, indent=2))

    torch.set_num_threads(4)
    model = wavmark.load_model().to("cpu").eval()

    fields = [
        "track", "attack", "status", "payload_exact",
        "bit_errors", "expected_payload", "recovered_payload",
        "elapsed_seconds", "error",
    ]

    results = run / "results.csv"
    with results.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        handle.flush()

        for original in tracks:
            folder = run / original.stem
            folder.mkdir()
            temporary = folder / "tmp"
            temporary.mkdir()

            digest = hashlib.sha256(
                f"DukeRIAATest|wavmark|{original.name}".encode()
            ).digest()
            payload = np.unpackbits(
                np.frombuffer(digest[:2], dtype=np.uint8)
            ).astype(int)
            expected = "".join(map(str, payload.tolist()))

            (folder / "source_sha256.txt").write_text(
                hashlib.sha256(original.read_bytes()).hexdigest()
            )

            print(f"\nEmbedding: {original.name}", flush=True)
            audio, rate = sf.read(
                original, dtype="float32", always_2d=True
            )
            mono = resample(audio.mean(axis=1), rate, 16000)

            np.random.seed(42)
            torch.manual_seed(42)
            model = model.to("cpu")
            with torch.no_grad():
                marked_mono, _ = wavmark.encode_watermark(
                    model, mono, payload, show_progress=True
                )

            marked_mono = np.asarray(marked_mono, dtype=np.float32)
            if marked_mono.shape != mono.shape:
                raise RuntimeError("Unexpected encoded audio length.")

            # Add only the watermark difference to the original audio.
            difference = resample(marked_mono - mono, 16000, rate)
            if len(difference) < len(audio):
                difference = np.pad(
                    difference, (0, len(audio) - len(difference))
                )
            marked_audio = audio + difference[:len(audio), None]

            clipped = int(np.count_nonzero(np.abs(marked_audio) > 1))
            (folder / "embedding.json").write_text(json.dumps({
                "expected_payload": expected,
                "sample_rate": rate,
                "channels": audio.shape[1],
                "samples_clipped": clipped,
            }, indent=2))

            marked = folder / "watermarked.wav"
            sf.write(
                marked, np.clip(marked_audio, -1, 1),
                rate, subtype="PCM_24"
            )

            model = model.to("mps")
            baseline_passed = False

            # Check the saved baseline before attempting any edits.
            tests = ["none", "control_unmarked"] + [
                attack for attack in ATTACKS if attack != "none"
            ]

            for attack in tests:
                print(f"\n  Running {attack}...", flush=True)
                started = time.perf_counter()
                row = dict.fromkeys(fields, "")
                row.update(
                    track=original.stem,
                    attack=attack,
                    status="ok",
                    expected_payload=expected,
                )

                try:
                    if attack == "control_unmarked":
                        path = original
                    else:
                        path = Path(apply_attack(
                            attack,
                            str(marked),
                            str(folder / f"{attack}.wav"),
                            str(temporary),
                        ))

                    signal = read_for_decoder(path)
                    with torch.no_grad():
                        recovered, info = wavmark.decode_watermark(
                            model, signal,
                            decode_batch_size=10,
                            show_progress=True,
                        )

                    exact = bool(
                        recovered is not None
                        and np.array_equal(recovered, payload)
                    )
                    row["payload_exact"] = exact

                    if recovered is not None:
                        bits = np.asarray(recovered).astype(int)
                        row["recovered_payload"] = "".join(
                            map(str, bits.tolist())
                        )
                        row["bit_errors"] = int(
                            np.count_nonzero(bits != payload)
                        )

                    details = {
                        "duration_seconds": len(signal) / 16000,
                        "matching_marker_windows": len(
                            info.get("results", [])
                        ),
                        "payload_exact": exact,
                        "recovered_payload": row["recovered_payload"],
                    }
                    (folder / f"{attack}.json").write_text(
                        json.dumps(details, indent=2)
                    )

                    if attack == "none":
                        baseline_passed = exact

                except Exception as exc:
                    row["status"] = "error"
                    detail = getattr(exc, "stderr", None) or str(exc)
                    row["error"] = str(detail)
                    (folder / f"{attack}_error.log").write_text(
                        str(detail)
                    )

                row["elapsed_seconds"] = round(
                    time.perf_counter() - started, 2
                )
                writer.writerow(row)
                handle.flush()
                print(
                    f"    status={row['status']} "
                    f"exact_payload={row['payload_exact']}",
                    flush=True,
                )

                if attack == "none" and not baseline_passed:
                    print(
                        "Baseline did not pass. Stopping this song "
                        "before edits; share the output for review.",
                        flush=True,
                    )
                    break

    print(f"\nResults saved to:\n{results}", flush=True)


if __name__ == "__main__":
    main()