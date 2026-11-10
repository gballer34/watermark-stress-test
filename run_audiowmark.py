import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from attacks import ATTACKS, apply_attack


ROOT = Path(__file__).resolve().parent
KEY = Path.home() / "watermark-testing/audio/test.key"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", help="Test just one original WAV")
    args = parser.parse_args()

    tracks = (
        [ROOT / args.track]
        if args.track
        else sorted(ROOT.glob("lyria_*.wav"))
    )

    if not tracks or any(not p.is_file() for p in tracks):
        sys.exit("Original WAV files not found.")

    for tool in ("docker", "sox", "ffmpeg"):
        if not shutil.which(tool):
            sys.exit(f"Missing tool: {tool}")

    if not KEY.is_file():
        sys.exit(f"Watermark key not found: {KEY}")

    # Check Docker before starting the experiment.
    image = subprocess.check_output(
        ["docker", "image", "inspect", "audiowmark",
         "--format", "{{.Id}}"],
        text=True,
    ).strip()

    run = ROOT / "audiowmark" / "runs" / datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    run.mkdir(parents=True)
    shutil.copy2(KEY, run / "test.key")

    # Record the exact shared test code used for this run.
    shutil.copy2(ROOT / "attacks.py", run / "attacks_used.py")
    metadata = {
        "strength": 10,
        "payload_bits": 128,
        "decoder": "default; speed detection not enabled",
        "docker_image": image,
        "attacks": ATTACKS,
        "tracks": [p.name for p in tracks],
        "attacks_sha256": hashlib.sha256(
            (ROOT / "attacks.py").read_bytes()
        ).hexdigest(),
        "python": sys.version,
    }
    (run / "settings.json").write_text(
        json.dumps(metadata, indent=2)
    )

    def container_path(path):
        return "/data/" + path.relative_to(ROOT).as_posix()

    def docker(arguments, log):
        result = subprocess.run(
            ["docker", "run", "--rm",
             "-v", f"{ROOT}:/data", image] + arguments,
            capture_output=True,
            text=True,
        )
        log.write_text(result.stdout + "\n" + result.stderr)
        if result.returncode:
            raise RuntimeError(f"Docker command failed; see {log.name}")
        return result.stdout

    results_path = run / "results.csv"

    with results_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "track", "attack", "payload_exact", "status",
            "expected_payload", "candidate_payloads",
            "elapsed_seconds", "error",
        ])
        writer.writeheader()

        for original in tracks:
            folder = run / original.stem
            folder.mkdir()
            temp = folder / "tmp"
            temp.mkdir()

            # Different, reproducible identifying code for each track.
            payload = hashlib.sha256(
                f"DukeRIAATest|audiowmark|{original.name}".encode()
            ).hexdigest()[:32]

            (folder / "source_sha256.txt").write_text(
                hashlib.sha256(original.read_bytes()).hexdigest()
            )

            marked = folder / "watermarked.wav"
            print(f"\nEmbedding: {original.name}", flush=True)

            docker([
                "add", "--key", container_path(run / "test.key"),
                "--strength", "10",
                container_path(original),
                container_path(marked),
                payload,
            ], folder / "embedding.log")

            for attack in ["control_unmarked"] + list(ATTACKS):
                print(f"  Running {attack}...", flush=True)
                started = time.perf_counter()
                row = {
                    "track": original.stem,
                    "attack": attack,
                    "payload_exact": "",
                    "status": "ok",
                    "expected_payload": payload,
                    "candidate_payloads": "",
                    "elapsed_seconds": "",
                    "error": "",
                }

                try:
                    if attack == "control_unmarked":
                        audio = original
                    else:
                        audio = Path(apply_attack(
                            attack,
                            str(marked),
                            str(folder / f"{attack}.wav"),
                            str(temp),
                        ))

                    output = docker([
                        "get",
                        "--key", container_path(run / "test.key"),
                        container_path(audio),
                        "--json", container_path(folder / f"{attack}.json"),
                    ], folder / f"{attack}.log")

                    candidates = []
                    for line in output.splitlines():
                        fields = line.split()
                        if len(fields) >= 3 and fields[0] == "pattern":
                            candidates.append(fields[2].lower())

                    row["payload_exact"] = payload in candidates
                    row["candidate_payloads"] = ";".join(candidates)

                except Exception as exc:
                    row["status"] = "error"
                    detail = getattr(exc, "stderr", None) or str(exc)
                    row["error"] = str(detail).strip()
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

    print(f"\nResults saved to:\n{results_path}", flush=True)


if __name__ == "__main__":
    main()