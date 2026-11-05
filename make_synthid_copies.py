#!/usr/bin/env python3
"""
Makes the edited copies to upload to the Gemini app for SynthID checks.

4 songs (one of each style) x 9 edits = 36 files in synthid_copies/.
The "none" edit is skipped: the clean songs were already checked.

Why only 4 songs: Gemini allows about 10 checks a day, and testing must
finish by Nov 6. One take of each style keeps the range of music.

Run from this folder:
    python make_synthid_copies.py
"""

import csv
import os
import shutil
import subprocess

from attacks import ATTACKS, apply_attack

SONGS = ["lyria_dense_1", "lyria_sparse_1", "lyria_pop_1", "lyria_ambient_1"]
OUT = "synthid_copies"
TMP = os.path.join(OUT, "tmp")
# Why outside synthid_copies/: the log is a result and belongs in the repo;
# the audio copies are ignored by git.
LOG = "synthid_results.csv"
LOG_COLUMNS = ["file", "song", "attack", "prompt_version", "date_checked",
               "detected", "time_range", "c2pa_status", "gemini_full_wording"]


def main():
    os.makedirs(TMP, exist_ok=True)
    rows = []
    for song in SONGS:
        for attack in ATTACKS:
            if attack == "none":
                continue
            dst = os.path.join(OUT, f"{song}_{attack}.wav")
            rows.append([os.path.basename(dst), song, attack, "v2"] + [""] * 5)
            # Why: skip files already made, so a rerun after a crash resumes
            # instead of redoing slow edits like stem splitting.
            if os.path.exists(dst):
                print(f"  exists   {dst}")
                continue
            try:
                apply_attack(attack, f"{song}.wav", dst, TMP)
                print(f"  made     {dst}")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                err = getattr(e, "stderr", None) or str(e)
                print(f"  FAILED   {dst}: {err.strip().splitlines()[-1]}")
    shutil.rmtree(TMP, ignore_errors=True)

    # Why append-only: this file holds results typed in by hand, so existing
    # rows are never rewritten. Only files not yet listed get a new row.
    logged = set()
    if os.path.exists(LOG):
        with open(LOG, newline="") as f:
            logged = {r["file"] for r in csv.DictReader(f)}
    new_rows = [r for r in rows if r[0] not in logged]
    with open(LOG, "a", newline="") as f:
        w = csv.writer(f)
        if not logged:
            w.writerow(LOG_COLUMNS)
        w.writerows(new_rows)
    print(f"\nAdded {len(new_rows)} rows to {LOG} to fill in.")
    print(f"Done. Upload the files in {OUT}/ to Gemini.")


if __name__ == "__main__":
    main()
