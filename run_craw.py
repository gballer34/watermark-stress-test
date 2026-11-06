#!/usr/bin/env python3
"""CRAW pilot for watermark-stress-test. Run from the TEAM repository.

Example:
  OMP_NUM_THREADS=2 python run_craw.py --craw-root ~/Documents/craw/watermarking_model
This is a 22.05kHz mono short-clip pilot, NOT a like-for-like full-song
48kHz stereo comparison with the AudioSeal benchmark.
"""
import argparse
import csv
import hashlib
import math
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf
import torch
import yaml
from scipy.signal import resample_poly


MODEL_SR = 22050
DEFAULT_ATTACKS = ['none', 'mp3128', 'noise20db', 'gain6db', 'polarity']


def resample(x, src_rate, dst_rate):
    if src_rate == dst_rate:
        return x
    divisor = math.gcd(src_rate, dst_rate)
    return resample_poly(x, dst_rate // divisor, src_rate // divisor).astype(np.float32)


def get_bits(clip_id, length):
    # Same idea as AudioSeal's hashed payload, but CRAW has only 10 data bits.
    source = f'Provider:DukeRIAATest|System:craw|Date:2026-11-01|ID:{clip_id}'
    digest = hashlib.sha256(source.encode()).digest()
    raw = int.from_bytes(digest, 'big') >> (256 - length)
    return np.array([(raw >> (length - i - 1)) & 1 for i in range(length)], dtype=np.int32), source


def decode(audio, decoder, ecc):
    waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32)).view(1, 1, -1)
    with torch.inference_mode():
        logits = decoder.test_forward(waveform)
    code_bits = (logits >= 0).cpu().numpy().astype(np.int32)
    bits = ecc.decode(code_bits) if ecc is not None else code_bits
    return np.asarray(bits).reshape(-1).astype(np.int32)


def read_mono(path):
    x, sr = sf.read(str(path), dtype='float32', always_2d=True)
    mono = x.mean(axis=1)
    return resample(mono, sr, MODEL_SR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--team-repo', default='.', help='Folder containing attacks.py and lyria_*.wav (default: current directory)')
    ap.add_argument('--craw-root', default='~/Documents/craw/watermarking_model', help='Existing CRAW model directory with config/, scripts/, checkpoint')
    ap.add_argument('--max-tracks', type=int, default=1, help='For pilot, default one track; 0=all')
    ap.add_argument('--clips-per-track', type=int, default=1, help='2-second excerpts at 0,30,60,... seconds')
    ap.add_argument('--clip-seconds', type=float, default=2.0)
    ap.add_argument('--attacks', nargs='+', default=DEFAULT_ATTACKS)
    args = ap.parse_args()

    team = Path(args.team_repo).expanduser().resolve()
    root = Path(args.craw_root).expanduser().resolve()
    if not (root / 'config/craw/model.yaml').exists():
        ap.error('CRAW model files not found at --craw-root: ' + str(root))
    if not (team / 'attacks.py').exists():
        ap.error('attacks.py not found in team repository: ' + str(team))
    sys.path.insert(0, str(root))
    from scripts.infer import load_model
    # CRAW checkpoint paths and auxiliary imports are relative to its model directory.
    # The input/output paths below are absolute, so results still live in team repo.
    os.chdir(str(root))
    sys.path.insert(0, str(team))
    from attacks import ATTACKS, RAWBENCH_EXTRAS, apply_attack
    allowed = set(ATTACKS + RAWBENCH_EXTRAS)
    unknown = set(args.attacks) - allowed
    if unknown:
        ap.error('Unknown attack names: ' + ', '.join(sorted(unknown)))
    if 'clip30' in args.attacks:
        ap.error('clip30 trims seconds 30-60; do not use for 2-second clips.')
    if 'stemsplit' in args.attacks:
        print('WARNING: stemsplit is heavy and may not be useful on short clips.', flush=True)
    if args.clip_seconds <= 0:
        ap.error('--clip-seconds must be positive.')

    tracks = sorted(team.glob('lyria_*.wav'))
    if args.max_tracks:
        tracks = tracks[:args.max_tracks]
    if not tracks:
        ap.error('No lyria_*.wav tracks found in ' + str(team))

    with open('config/process.yaml') as f:
        process_config = yaml.safe_load(f)
    with open('config/craw/model.yaml') as f:
        model_config = yaml.safe_load(f)
    with open('config/craw/train.yaml') as f:
        train_config = yaml.safe_load(f)
    if process_config['audio']['sample_rate'] != MODEL_SR:
        ap.error('Unexpected CRAW target sample rate; review MODEL_SR.')
    msg_len = int(train_config['watermark']['length'])
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    encoder, decoder, ecc = load_model(
        model_config, process_config, train_config, torch.device('cpu'), ckpt_epoch=20
    )
    output = team / 'craw'
    (output / 'marked').mkdir(parents=True, exist_ok=True)
    (output / 'control').mkdir(parents=True, exist_ok=True)
    result_file = output / 'craw_results.csv'

    columns = ['track', 'clip_start_sec', 'clip_duration_sec', 'attack',
               'sample_type', 'detected_exact', 'bit_accuracy_pct', 'payload_exact',
               'embedded_bits', 'recovered_bits', 'embed_snr_db', 'payload_source_string',
               'status']
    with open(result_file, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for track in tracks:
            info = sf.info(str(track))
            for n in range(args.clips_per_track):
                start_s = float(n * 30)
                start_frame = int(start_s * info.samplerate)
                stop_frame = start_frame + round(args.clip_seconds * info.samplerate)
                if stop_frame > info.frames:
                    print(f'Skipping {track.name} @ {start_s}s: too short', flush=True)
                    continue
                x, sr = sf.read(str(track), start=start_frame, stop=stop_frame,
                                always_2d=True, dtype='float32')
                original = resample(x.mean(axis=1), sr, MODEL_SR)
                clip_id = f'{track.stem}_{int(start_s):04d}'
                bits, source = get_bits(clip_id, msg_len)
                msg = bits.astype(np.float32).reshape(1, 1, -1)
                coded = ecc.encode(msg) if ecc is not None else msg
                msg_t = torch.from_numpy(coded.astype(np.float32) * 2 - 1)
                input_t = torch.from_numpy(original).reshape(1, 1, -1)
                print(f'Embedding {clip_id}: {len(original)/MODEL_SR:.2f}s', flush=True)
                with torch.inference_mode():
                    marked_tensor, _ = encoder.test_forward(input_t, msg_t)
                    marked_tensor = marked_tensor.clamp(-1.0, 1.0)
                marked = marked_tensor.squeeze().cpu().numpy()
                control_path = output / 'control' / f'{clip_id}.wav'
                marked_path = output / 'marked' / f'{clip_id}_craw.wav'
                sf.write(str(control_path), original, MODEL_SR, subtype='PCM_24')
                sf.write(str(marked_path), marked, MODEL_SR, subtype='PCM_24')
                difference_power = np.sum((marked - original) ** 2)
                snr = (10 * np.log10(np.sum(original ** 2) / difference_power)
                       if difference_power > 0 and np.sum(original ** 2) > 0 else float('nan'))

                def record(test_path, attack, sample_type):
                    try:
                        decoded = decode(read_mono(test_path), decoder, ecc)
                        if len(decoded) != len(bits):
                            raise ValueError(f'Expected {len(bits)} bits, got {len(decoded)}')
                        match = int(np.sum(decoded == bits))
                        accuracy = 100.0 * match / len(bits)
                        exact = match == len(bits)
                        status = 'ok'
                        print(f'  {sample_type:10s} {attack:14s} bits={accuracy:5.1f}% exact={exact}', flush=True)
                    except Exception as exc:
                        decoded, accuracy, exact = [], '', False
                        status = 'failed: ' + str(exc)
                        print(f'  FAILED {sample_type} {attack}: {exc}', flush=True)
                    writer.writerow(dict(track=track.stem, clip_start_sec=start_s,
                        clip_duration_sec=args.clip_seconds, attack=attack,
                        sample_type=sample_type, detected_exact=exact,
                        bit_accuracy_pct=accuracy, payload_exact=exact,
                        embedded_bits=''.join(map(str, bits)),
                        recovered_bits=''.join(map(str, decoded)),
                        embed_snr_db=f'{snr:.2f}' if sample_type == 'watermarked' else '',
                        payload_source_string=source, status=status))
                    handle.flush()

                record(control_path, 'control_unmarked', 'control')
                with TemporaryDirectory(prefix='craw_attack_') as tmp:
                    temp = Path(tmp)
                    for attack in args.attacks:
                        dst = temp / f'{clip_id}_{attack}.wav'
                        try:
                            path = apply_attack(attack, str(marked_path), str(dst), str(temp))
                            record(path, attack, 'watermarked')
                        except Exception as exc:
                            print(f'  ATTACK FAILED {attack}: {exc}', flush=True)
                            writer.writerow(dict(track=track.stem, clip_start_sec=start_s,
                                clip_duration_sec=args.clip_seconds, attack=attack,
                                sample_type='watermarked', detected_exact='',
                                bit_accuracy_pct='', payload_exact='',
                                embedded_bits=''.join(map(str,bits)), recovered_bits='',
                                embed_snr_db=f'{snr:.2f}', payload_source_string=source,
                                status='attack_failed: '+str(exc)))
                            handle.flush()
    print('\nResults saved to ' + str(result_file))
    print('NOTE: exact-match is a provisional detector, not a calibrated FPR threshold.')


if __name__ == '__main__':
    main()
