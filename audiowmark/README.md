# audiowmark testing

Tested all eight team Lyria tracks using the shared attacks.py.
Each edit was applied separately to the full watermarked song.
The trimming test kept 30 seconds starting at 0:30.

## Method
- Embedded a 128-bit payload derived from each filename.
- Used strength 10 and the same test key for embedding and decoding.
- Ran audiowmark through Docker.
- Used the default decoder, without speed-detection options.
- Counted success when the expected payload appeared among decoded candidates.
- Checked the original files and unedited watermarked files as controls.

## Results
All eight unedited watermarked files returned the expected payload.
None of the original files returned the expected payload.

Exact payload recovery:
- 8/8: MP3 compression, added noise, trimming, reduced volume,
  vocal removal, and polarity reversal.
- 0/8: speed increase, slowing with reverb, and pitch change.

These findings apply to the tested settings and eight tracks.
Failure to recover the payload does not prove the watermark was removed.
The original Lyria tracks may already contain SynthID.

The results table is audiowmark_results.csv.
The testing script is ../run_audiowmark.py.