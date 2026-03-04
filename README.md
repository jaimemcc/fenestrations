# fenestrations

## assemble_data script

Run:

`python src/assemble_data.py --input data --output data/roi_data.pickle`

### Common flags

- `--input`: folder with `*_seg.npy` files (default `data/`)
- `--tif-input`: folder with `.tif` files (default is `--input`)
- `--output`: output pickle path
- `--append` / `--no-append`: append to existing output pickle
- `--continue-on-error` / `--no-continue-on-error`: keep processing if one stub fails
- `--remove-outliers` / `--no-remove-outliers`: remove robust-tail outliers before summary
- `--outlier-z-thresh`: robust z-score threshold for outlier filtering
- `--max-k`: max nearest neighbors used in analysis
- `--use-raw-cache` / `--no-use-raw-cache`: reuse cached raw ROI extraction
- `--refresh-raw-cache` / `--no-refresh-raw-cache`: force rebuilding raw ROI cache
- `--raw-cache-path`: path to raw ROI cache pickle
- `--metadata-path`: metadata file (`.xlsx`, `.xls`, `.csv`)
- `--metadata-key`: metadata merge key (`auto`, `id`, `stub`)
- `--id-token-count`: underscore token count used to build `id`
- `--log-file`: optional logfile path for terminal output

### Raw-cache behavior

- When `--use-raw-cache` is enabled, cached ROIs are reused first.
- The script scans for stubs on disk and processes only new stubs not already present in cache.
- If no new stubs are found, the run continues from cache data.
- If input or tif folders are missing during scan, the script warns and continues (useful for cache-only runs).

### Log file output

- Terminal output is saved to both the terminal and a logfile.
- Use `--log-file` to choose the path:

`python src/assemble_data.py --input data --output data/roi_data.pickle --log-file data/assemble_run.log`

- If `--log-file` is omitted, a timestamped logfile is created next to `--output`:
	- `assemble_data_YYYYMMDD_HHMMSS.log`