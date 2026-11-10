#!/usr/bin/env python3
"""Memory-conscious 30-second CRAW robustness test for watermark-stress-test.

Run FROM the team's watermark-stress-test folder using the existing `craw` conda env.
Embeds the same 10-bit per-track payload independently in 15 consecutive 2-s
windows, joins them into a 30-s WAV, then attacks the complete 30-s WAV.
During decoding it processes 2-s windows independently and aggregates scores.

Only tests attacks that preserve duration/synchronization. Speed/pitch/tempo
changes, reverb, stem splitting and cropping require separate alignment tests.

Disk policy: Keep exactly one marked WAV per evaluated track and a CSV.
Unmarked controls are decoded from RAM and all attacked WAVs are stored in
a TemporaryDirectory outside the repository, deleted automatically.
This is a 22,050 Hz MONO benchmark, not directly matched to 48 kHz stereo AudioSeal.
"""
import argparse
import csv
import hashlib
import math
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf
import torch
import yaml
from scipy.signal import resample_poly

SR = 22050
WINDOW_SECONDS = 2
ATTACKS_DEFAULT = ['none', 'mp3128', 'noise20db', 'gain6db', 'polarity']
# All these keep timing approximately intact. Verify output duration for each.
ALLOWED_ATTACKS = {'none', 'mp3128', 'noise20db', 'gain6db', 'polarity', 'aac128'}


def resample_mono(x, sr):
    if sr == SR:
        return x.astype(np.float32, copy=False)
    g = math.gcd(sr, SR)
    return resample_poly(x, SR // g, sr // g).astype(np.float32)


def make_payload(track):
    text = f'Provider:DukeRIAATest|System:craw|Date:2026-11-01|ID:{track}'
    d = hashlib.sha256(text.encode()).digest()
    value = int.from_bytes(d[:2], 'big') >> 6
    return np.asarray([(value >> (9 - i)) & 1 for i in range(10)], dtype=np.int32), text


def decode_one(segment, decoder, ecc):
    x = torch.from_numpy(np.asarray(segment, dtype=np.float32)).reshape(1, 1, -1)
    with torch.inference_mode():
        logits = decoder.test_forward(x)
    code = (logits >= 0).cpu().numpy().astype(np.int32)
    return np.asarray(ecc.decode(code) if ecc is not None else code).reshape(-1).astype(np.int32)


def decode_windows(audio, sr, decoder, ecc, bits, expected_windows):
    """Decode audio in 2-s windows; audio is mono ndarray or WAV path."""
    if isinstance(audio, (str, Path)):
        x, sr = sf.read(str(audio), dtype="float32", always_2d=True)
        mono = x.mean(axis=1)
    else:
        mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    mono = resample_mono(mono, sr)
    n = SR * WINDOW_SECONDS
    if len(mono) < expected_windows * n:
        raise ValueError(f'Attack output too short: {len(mono)/SR:.3f}s, expected >= {expected_windows*WINDOW_SECONDS}s')
    # AAC/MP3 often pad a few samples. For this initial benchmark, start at 0.
    all_decoded = []
    for j in range(expected_windows):
        got = decode_one(mono[j*n:(j+1)*n], decoder, ecc)
        if len(got) != len(bits):
            raise ValueError(f'Unexpected decoded length: {len(got)}')
        all_decoded.append(got)
    matrix = np.stack(all_decoded)
    matches = (matrix == bits[None, :])
    exact_rate = 100.0 * np.mean(matches.all(axis=1))
    mean_acc = 100.0 * np.mean(matches)
    vote = (matrix.sum(axis=0) > len(matrix)/2).astype(np.int32)
    # For even window counts, tie goes to zero; here default 15 windows (odd).
    vote_acc = 100.0 * np.mean(vote == bits)
    return mean_acc, exact_rate, vote_acc, bool(np.all(vote == bits)), ''.join(map(str, vote)), len(matrix)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--craw-root', default='~/Documents/craw/watermarking_model')
    p.add_argument('--max-tracks', type=int, default=1, help='1 for a pilot, 0 for all tracks')
    p.add_argument('--start-sec', type=float, default=30.0, help='Start of each 30-s source excerpt')
    p.add_argument('--clip-sec', type=int, default=30, help='Must be a positive multiple of 2')
    p.add_argument('--attacks', nargs='+', default=ATTACKS_DEFAULT)
    args = p.parse_args()
    if args.clip_sec <= 0 or args.clip_sec % WINDOW_SECONDS:
        p.error('--clip-sec must be a positive multiple of 2')
    if args.start_sec < 0:
        p.error('--start-sec cannot be negative')
    invalid = set(args.attacks) - ALLOWED_ATTACKS
    if invalid:
        p.error(f'Unsupported attacks for aligned 30-s pilot: {sorted(invalid)}')
    team = Path.cwd()
    root = Path(args.craw_root).expanduser().resolve()
    if not (team / 'attacks.py').exists():
        p.error('Run this script from the team repository (attacks.py must be present)')
    if not (root / 'config/craw/model.yaml').exists():
        p.error(f'CRAW model not found at {root}')
    tracks = sorted(team.glob('lyria_*.wav'))
    if args.max_tracks > 0:
        tracks = tracks[:args.max_tracks]
    if not tracks:
        p.error('No lyria_*.wav found at team repository root')

    # CRAW model config and checkpoint paths are relative to its own directory.
    sys.path.insert(0, str(root))
    from scripts.infer import load_model
    sys.path.insert(0, str(team))
    from attacks import apply_attack
    os.chdir(str(root))
    with open('config/process.yaml') as f:
        process_cfg = yaml.safe_load(f)
    with open('config/craw/model.yaml') as f:
        model_cfg = yaml.safe_load(f)
    with open('config/craw/train.yaml') as f:
        train_cfg = yaml.safe_load(f)
    if process_cfg['audio']['sample_rate'] != SR or train_cfg['watermark']['length'] != 10:
        p.error('Unexpected CRAW sample rate/message length; inspect model configs')

    torch.set_num_threads(2)
    encoder, decoder, ecc = load_model(model_cfg, process_cfg, train_cfg, torch.device('cpu'), ckpt_epoch=20)
    out = team / 'craw'
    (out / 'marked_30s').mkdir(parents=True, exist_ok=True)
    csv_path = out / 'craw_results_30s.csv'  # Doesn't overwrite 2-s results.
    headers = ['track', 'start_sec', 'clip_sec', 'attack', 'sample_type', 'windows',
               'mean_bit_accuracy_pct', 'exact_window_rate_pct', 'majority_bit_accuracy_pct',
               'majority_payload_exact', 'recovered_majority_bits', 'embedded_bits',
               'embed_snr_db', 'payload_source_string', 'status']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for track in tracks:
            info = sf.info(str(track))
            first = round(args.start_sec * info.samplerate)
            last = first + round(args.clip_sec * info.samplerate)
            if last > info.frames:
                print(f'SKIP {track.name}: not enough audio for {args.start_sec}-{args.start_sec+args.clip_sec}s')
                continue
            source, source_sr = sf.read(str(track), start=first, stop=last, always_2d=True, dtype='float32')
            clean = resample_mono(source.mean(axis=1), source_sr)
            chunk_samples = WINDOW_SECONDS * SR
            nwindows = args.clip_sec // WINDOW_SECONDS
            clean = clean[:nwindows * chunk_samples]
            if clean.size != nwindows * chunk_samples:
                print(f'SKIP {track.name}: resampled length does not fit 2-s windows')
                continue
            bits, source_str = make_payload(track.stem)
            msg = bits.astype(np.float32).reshape(1, 1, -1)
            coded = ecc.encode(msg) if ecc is not None else msg
            msg_tensor = torch.from_numpy(np.asarray(coded, dtype=np.float32) * 2 - 1)
            marked_parts = []
            print(f'\n{track.name}: embedding {nwindows} independent 2-s chunks', flush=True)
            for j in range(nwindows):
                chunk = torch.from_numpy(clean[j*chunk_samples:(j+1)*chunk_samples].copy()).reshape(1, 1, -1)
                with torch.inference_mode():
                    marked_tensor, _ = encoder.test_forward(chunk, msg_tensor)
                    marked_tensor = marked_tensor.clamp(-1, 1)
                marked_parts.append(marked_tensor.squeeze().cpu().numpy().astype(np.float32))
                print(f'  embedded chunk {j+1}/{nwindows}', flush=True)
            marked = np.concatenate(marked_parts)
            del marked_parts
            label = f'{track.stem}_{int(args.start_sec)}s_{args.clip_sec}s'
            marked_path = out / 'marked_30s' / f'{label}_craw.wav'
            sf.write(str(marked_path), marked, SR, subtype='PCM_24')
            noise_power = float(np.sum((marked-clean)**2))
            signal_power = float(np.sum(clean**2))
            snr = 10 * math.log10(signal_power/noise_power) if signal_power > 0 and noise_power > 0 else float('nan')

            def evaluate(audio, audio_sr, attack, sample_type):
                try:
                    mean, exact, vote_acc, vote_exact, vote_bits, windows = decode_windows(
                        audio, audio_sr, decoder, ecc, bits, nwindows)
                    status = 'ok'
                    print(f'  {sample_type:11s} {attack:10s} mean_bits={mean:5.1f}% exact_windows={exact:5.1f}% vote_exact={vote_exact}', flush=True)
                except Exception as ex:
                    mean = exact = vote_acc = vote_exact = vote_bits = windows = ''
                    status = f'failed: {ex}'
                    print(f'  ERROR {attack}: {ex}', flush=True)
                writer.writerow(dict(track=track.stem, start_sec=args.start_sec, clip_sec=args.clip_sec,
                    attack=attack, sample_type=sample_type, windows=windows,
                    mean_bit_accuracy_pct=mean, exact_window_rate_pct=exact,
                    majority_bit_accuracy_pct=vote_acc, majority_payload_exact=vote_exact,
                    recovered_majority_bits=vote_bits, embedded_bits=''.join(map(str,bits)),
                    embed_snr_db=f'{snr:.2f}' if sample_type == 'watermarked' else '',
                    payload_source_string=source_str, status=status))
                f.flush()

            evaluate(clean, SR, 'control_unmarked', 'control')
            with TemporaryDirectory(prefix='craw_30s_attack_') as tmp:
                for attack in args.attacks:
                    dst = Path(tmp) / f'{label}_{attack}.wav'
                    try:
                        attacked_path = apply_attack(attack, str(marked_path), str(dst), tmp)
                        evaluate(attacked_path, SR, attack, 'watermarked')
                    except Exception as ex:
                        print(f'  ATTACK FAILED {attack}: {ex}', flush=True)
                        writer.writerow(dict(track=track.stem, start_sec=args.start_sec, clip_sec=args.clip_sec,
                            attack=attack, sample_type='watermarked', embedded_bits=''.join(map(str,bits)),
                            status=f'attack_failed: {ex}'))
                        f.flush()
    print(f'\nSaved results to: {csv_path}')
    print('Interpretation: 30-s joined excerpt; recovery tested per 2-s chunk and majority vote.')
    print('Saved: one marked WAV per song plus the CSV; no per-chunk/control/attacked WAVs kept.')
    print('Exact-message majority vote is not a calibrated detection threshold.')

if __name__ == '__main__':
    main()
