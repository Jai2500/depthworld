# Local training/eval entrypoint package.
#
# Some Python environments install an unrelated top-level `scripts` package.
# Keeping this file ensures imports such as `import scripts.train_wm` resolve
# to this repository's scripts directory when the repo root is on PYTHONPATH.
