"""
The 9 edits ("attacks") used in this project, all in one place.

Why one shared file: run_audioseal.py and make_synthid_copies.py both
import from here, so SynthID and AudioSeal are guaranteed to get exactly
the same edits with exactly the same settings.

Why every attack writes a WAV at the input's sample rate: so file format
and sample rate never sneak in as a hidden second edit. (Our tracks are
48 kHz, 16-bit stereo.)
"""

import os
import shutil
import subprocess

import numpy as np
import soundfile as sf

# Why two lists: ATTACKS is shared by SynthID and AudioSeal (the side-by-side
# comparison). RAWBENCH_EXTRAS run on AudioSeal only, to check our setup
# against the AudioSeal results published in the Sony RAW-Bench paper,
# without spending scarce Gemini checks.
ATTACKS = ["none", "mp3128", "speedup10", "slowreverb", "pitch2",
           "noise20db", "clip30", "gain6db", "stemsplit", "polarity"]
RAWBENCH_EXTRAS = ["timestretch5", "aac128"]

# Why 20 dB: the noisiest level RAW-Bench uses, so results compare with
# the published benchmark. Clearly audible hiss, music still clear.
NOISE_SNR_DB = 20
# Why a fixed seed: every rerun adds the exact same hiss, so results repeat.
NOISE_SEED = 1234

# Why a separate environment: Demucs needs its own versions of some
# packages, and installing it into .venv could change the AudioSeal setup.
DEMUCS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      ".venv-demucs", "bin", "demucs")


def run(cmd):
    """Why: list-style arguments avoid quoting problems; output is kept so
    a failure can print the tool's own error message."""
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def apply_attack(attack, src, dst, tmp_dir):
    """Apply one attack to src. Returns the path of a WAV ready for detection."""
    info = sf.info(src)
    if attack == "none":
        return src
    if attack == "mp3128":
        # Why: encode to MP3, then decode back to WAV, which is exactly
        # what a listener's player does before the audio is heard.
        mp3 = os.path.join(tmp_dir, "tmp.mp3")
        run(["ffmpeg", "-y", "-i", src, "-b:a", "128k", mp3])
        run(["ffmpeg", "-y", "-i", mp3, dst])
        os.remove(mp3)
    elif attack == "speedup10":
        # Why: "sped up" edits change tempo and pitch together, like playing
        # a record faster. SoX resamples back to the original rate itself.
        run(["sox", src, dst, "speed", "1.1"])
    elif attack == "slowreverb":
        run(["sox", src, dst, "speed", "0.85", "reverb"])
    elif attack == "pitch2":
        # Why: 200 cents = 2 semitones up, tempo unchanged, like a key change.
        run(["sox", src, dst, "pitch", "200"])
    elif attack == "noise20db":
        # Why numpy instead of SoX: SNR is defined relative to the song's own
        # loudness, so we measure the song's power and scale the hiss to it.
        x, sr = sf.read(src, always_2d=True, dtype="float64")
        noise_power = np.mean(x ** 2) / 10 ** (NOISE_SNR_DB / 10)
        rng = np.random.default_rng(NOISE_SEED)
        noise = rng.normal(0.0, np.sqrt(noise_power), x.shape)
        sf.write(dst, np.clip(x + noise, -1.0, 1.0), sr, subtype=info.subtype)
    elif attack == "clip30":
        # Why: short-form platforms play excerpts; 0:30 to 1:00 skips the intro.
        run(["sox", src, dst, "trim", "30", "30"])
    elif attack == "gain6db":
        # Why numpy instead of SoX: halving the samples is the whole edit;
        # SoX would also add dither noise, a hidden second change.
        x, sr = sf.read(src, always_2d=True, dtype="float64")
        sf.write(dst, x * 0.5, sr, subtype=info.subtype)
    elif attack == "stemsplit":
        # Why: removing vocals (karaoke / instrumental versions) is a common
        # real edit. Demucs rebuilds the song from separated parts, which
        # tests the mark even on instrumental tracks with no vocals.
        if not os.path.exists(DEMUCS):
            raise FileNotFoundError("Demucs not installed in .venv-demucs (see README)")
        out = os.path.join(tmp_dir, "demucs")
        run([DEMUCS, "--two-stems=vocals", "-n", "htdemucs", "-o", out, src])
        name = os.path.splitext(os.path.basename(src))[0]
        no_vocals = os.path.join(out, "htdemucs", name, "no_vocals.wav")
        # Why: Demucs outputs 44.1 kHz; convert back so sample rate stays fixed.
        run(["sox", no_vocals, dst, "rate", str(info.samplerate)])
        shutil.rmtree(out)
    elif attack == "polarity":
        # Why: flipping the wave upside down sounds identical to a listener,
        # so a watermark that fails here can be removed at zero quality cost.
        # RAW-Bench found it breaks some watermarks.
        x, sr = sf.read(src, always_2d=True, dtype="float64")
        sf.write(dst, -x, sr, subtype=info.subtype)
    elif attack == "timestretch5":
        # Why: 5% faster with pitch unchanged (RAW-Bench's mildest stretch).
        # Paired with pitch2 (pitch only), it shows whether timing or pitch
        # is what breaks a mark in speedup10.
        run(["sox", src, dst, "tempo", "1.05"])
    elif attack == "aac128":
        # Why: AAC is what YouTube, Apple Music and Instagram use; same
        # bitrate as mp3128 so the two formats compare directly.
        m4a = os.path.join(tmp_dir, "tmp.m4a")
        run(["ffmpeg", "-y", "-i", src, "-c:a", "aac", "-b:a", "128k", m4a])
        run(["ffmpeg", "-y", "-i", m4a, dst])
        os.remove(m4a)
    else:
        raise ValueError(f"Unknown attack: {attack}")
    return dst
