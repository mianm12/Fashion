# Kaggle Download Script Design

## Goal

Add a Python script that downloads a Kaggle competition dataset into the project data directory.

## Behavior

- Default competition: `h-and-m-personalized-fashion-recommendations`.
- Default destination: `data/raw/<competition>/`.
- Use KaggleHub's Python API: `kagglehub.competition_download(<competition>, output_dir=<destination>, force_download=<force>)`.
- Extract downloaded zip files by default.
- Skip the download when extracted files or zip files already exist, unless `--force` is passed.
- Print clear setup guidance when the `kagglehub` package is missing from the active Python environment.

## Interface

Run:

```powershell
python src/00_download_data.py
```

or:

```powershell
python src/00_download_data.py --competition h-and-m-personalized-fashion-recommendations
```

Useful options:

- `--data-dir data/raw`
- `--no-unzip`
- `--force`

## Testing

Unit tests cover argument defaults, target path construction, skip behavior, KaggleHub downloader invocation, and zip extraction without requiring network access or Kaggle credentials.
