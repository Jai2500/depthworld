# Downloading datasets from data.ciirc.cvut.cz

**Need access?** If you have not requested access yet, please do so through this form: https://forms.gle/naqPdvsgdb1Z1ikH7

Access uses a personal token that you receive separately from this README. Each token grants read access to one dataset and expires 7 days after it is issued. Do not share it, commit it, or paste it into shared notebooks or chats. If it leaks, contact Jai Bardhan (jai.bardhan@cvut.cz).

## Requirements

Python 3.8 or newer on Linux or macOS. No packages to install, no aria2c, no root. Episode folder names contain colons, which Windows does not allow in file names, so on Windows run the script inside WSL.

## Two datasets, two tokens

DROID-3D and DROID-3D pseudo data each have their own token, and each token tells the script which dataset to fetch. Run steps 1 to 3 once per dataset, with that dataset's token and its own `--out` directory, for example `/path/to/droid-3d` and `/path/to/droid-3d-pseudo`. To download both at once, use two terminals, each with its own token set.

## 1. Set the token

Set the token without writing it to your shell history. Paste the token and press Enter; nothing is echoed.

```bash
read -rs TOKEN && export TOKEN
```

## 2. Start with a dry run

```bash
python3 droid3d_downloader.py --dry-run --out /path/to/droid-3d
```

A dry run downloads only the manifest (the list of files) and nothing else. It confirms that your token works and when it expires, then prints the number of files, how many directories exist at each depth below the dataset root, and a few example paths. Use it to check the dataset before committing disk space and time, and to choose `--group-depth` if you want a subset (see below).

Use the same `--out` for the dry run and the real download: the real run then reuses the manifest instead of fetching it again. If the dataset is updated later, delete `manifest.txt` in that directory to fetch a fresh copy.

Adding `--limit` or `--include` to a dry run previews a subset: it prints how many files the subset contains and writes their paths to `selection.txt`, still without downloading them.

## 3. Download

### The full dataset

```bash
python3 droid3d_downloader.py --out /path/to/droid-3d
```

The dataset name is read from the token, so `--dataset` is only needed if your token does not name one. The script downloads every file in parallel (8 connections by default) and reproduces the dataset's directory tree under `--out`. Please keep `--workers` at 16 or below to limit load on the server. On a shared machine, run it inside `tmux`, `screen`, or a batch job. The token is read from the environment and never appears on the command line, so other users cannot see it with `ps`.

### A subset

`--limit N --group-depth D` keeps N complete directories at depth D (with `--group-depth 0`, the default, N single files). Files above depth D, such as the top-level `README.md`, are always included. Choose D as the shallowest depth whose directory count in the dry run equals the number of episodes, so that every selected episode arrives with all its files. In DROID-3D, episodes sit at depth 5 (for example `1.0.1/IRIS/success/2023-05-02/Tue_May__2_12:11:42_2023/`), so 100 episodes is:

```bash
python3 droid3d_downloader.py --limit 100 --group-depth 5 --out /path/to/droid-3d
```

The selection is a uniform random sample determined by `--seed` (default 0). Everyone using the same seed gets the same subset, and subsets are nested: the `--limit 10` subset is contained in the `--limit 100` subset, so growing a subset later downloads only the new files. Use a different `--seed` for an independent sample.

`--include GLOB` restricts the download to paths matching a glob and can be repeated; `*` also matches `/`. For example, `--include '1.0.1/IRIS/*'` fetches only the IRIS lab. It combines with `--limit`. Whenever a subset is used, the selected paths are written to `selection.txt` in the output directory.

## Camera extrinsics

The camera extrinsics are hosted on Hugging Face at https://huggingface.co/datasets/jaibrdhn/droid_3d_extrinsics. The repository is gated and requests are approved manually, only for people who have submitted the access form linked at the top of this README:

1. Make sure you have submitted the access form. Then log in to Hugging Face, open the repository page, and request access there. Downloads fail with an access error until your request is approved.
2. Once approved, log in from your terminal with your own Hugging Face access token with read permission (create one at https://huggingface.co/settings/tokens). This token is separate from the dataset tokens above, so do not put it in the `TOKEN` variable.

```bash
pip install -U huggingface_hub
hf auth login
hf download jaibrdhn/droid_3d_extrinsics --repo-type dataset --local-dir /path/to/droid_3d_extrinsics
```

## Progress

The manifest download, the manifest scan, and the file download each show a progress bar in a terminal. When output is redirected to a file or batch log, each prints a status line every 30 seconds instead. During the file download, the time estimate assumes the remaining files transfer at the rate of the last 20 seconds. The server does not report the manifest's size in advance, so the manifest download shows bytes, lines received and elapsed time rather than a percentage.

## Interruptions and token expiry

Rerun the same command to resume. Completed files are skipped. Partially downloaded files (`*.part`) continue from where they stopped if the server supports resuming, and restart otherwise. If the script stops with `token rejected` or reports that the token expired, request a new one, set it as in step 1, and rerun.

## Failures

Files that still fail after all retries are listed in `failed.txt` in the output directory. Rerunning retries them. If a URL keeps failing, send `failed.txt` to the contact above.
