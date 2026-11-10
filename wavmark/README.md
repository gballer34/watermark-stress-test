# WavMark testing

Tested all eight team Lyria tracks using the shared attacks.py.
Each edit was applied separately to the full watermarked song.
The trimming test kept 30 seconds starting at 0:30.

## Method
- Embedded a 16-bit payload derived from each filename.
- Used default embedding settings and random seed 42.
- Embedded on CPU in 16 kHz mono audio.
- Resampled the watermark difference and added it to both original
  stereo channels, saving at the original sample rate as PCM24.
- Applied the shared edits to those saved files.
- Converted each test file to 16 kHz mono for decoding.
- Decoded using Apple's MPS GPU with batch size 10.
- Counted success only when all 16 payload bits matched.
- Checked the original files and unedited watermarked files as controls.

## Results
All eight unedited watermarked files returned the expected payload.
None of the original files returned the expected payload.

Exact payload recovery:
- 8/8: MP3 compression, speed increase, trimming, reduced volume,
  vocal removal, and polarity reversal.
- 5/8: added noise.
- 0/8: slowing with reverb and pitch change.

For added noise, dense_2 returned a payload with 2 incorrect bits.
Sparse_1 and sparse_2 returned no payload. Blank bit-error entries
mean no payload was returned, not zero errors.

These findings apply to the tested settings and eight tracks.
Failure to recover the payload does not prove the watermark was removed.
The original Lyria tracks may already contain SynthID.
Some sections were skipped during embedding under the default
signal-to-noise constraint; all saved baselines still passed.

The results table is wavmark_results.csv.
The testing script is ../run_wavmark.py.