#!/usr/bin/env python
"""Build a per-episode index for the raw DROID dataset.

Walks <raw_root>/<LAB>/{success,failure}/<DATE>/<TS>/.
For each valid episode (has trajectory.h5, has recordings/s2m2_v2 with 3 cams,
has 3 MP4s, has parseable metadata_*.json), emit one JSONL row containing
everything the latent extractor needs:

  - traj_id           : contiguous int (assigned by uuid sort)
  - split             : "train" | "val" (hash(uuid)%100 == 99 -> val)
  - episode_uuid      : canonical "<LAB>+<hash>+<date>-<time>" key
  - raw_dir           : absolute path to episode directory
  - lab, success, current_task, trajectory_length
  - cameras: { wrist|ext1|ext2: { serial, fx, baseline, mp4_path, npz_path } }
  - language_instructions: list pulled from aggregated-annotations-030724.json

Read-only: never writes to the raw tree.
"""

import argparse
import glob
import hashlib
import json
import os
import signal
import time
from multiprocessing import Pool
from pathlib import Path

# Set from --raw_root in main().
RAW_ROOT = None
AGG_ANN_PATH = None
LABS = [
    "AUTOLab", "CLVR", "GuptaLab", "ILIAD", "IPRL", "IRIS",
    "PennPAL", "RAD", "RAIL", "REAL", "RPL", "TRI", "WEIRD",
]
PER_EPISODE_TIMEOUT_S = 15
MIN_NPZ_SIZE_BYTES = 1_000_000  # filter obviously truncated/empty files

# Filled by pool initializer in each worker.
_AGG_ANN: dict = {}


def discover_episode_dirs():
    eps = []
    for lab in LABS:
        lab_root = Path(RAW_ROOT) / lab
        if not lab_root.is_dir():
            continue
        for status in ("success", "failure"):
            status_root = lab_root / status
            if not status_root.is_dir():
                continue
            for date_dir in status_root.iterdir():
                if not date_dir.is_dir():
                    continue
                for ep_dir in date_dir.iterdir():
                    if not ep_dir.is_dir():
                        continue
                    eps.append(str(ep_dir))
    return eps


def _init_worker(agg_path):
    global _AGG_ANN
    try:
        with open(agg_path) as f:
            _AGG_ANN = json.load(f)
    except Exception:
        _AGG_ANN = {}


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def index_episode(ep_dir):
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(PER_EPISODE_TIMEOUT_S)
    try:
        # 1. find metadata_*.json (one per episode)
        meta_paths = glob.glob(os.path.join(ep_dir, "metadata_*.json"))
        if not meta_paths:
            return ("__error__", "no_metadata_json", ep_dir)
        with open(meta_paths[0]) as f:
            meta = json.load(f)

        uuid = meta.get("uuid")
        if not uuid:
            return ("__error__", "no_uuid_in_metadata", ep_dir)

        s2m2_root = os.path.join(ep_dir, "recordings", "s2m2_v2")
        if not os.path.isdir(s2m2_root):
            return ("__error__", "no_s2m2_v2", ep_dir)

        # 2. role -> serial from metadata; verify s2m2 config + npz + mp4 present
        cameras = {}
        for role in ("wrist", "ext1", "ext2"):
            serial = meta.get(f"{role}_cam_serial")
            if not serial:
                return ("__error__", f"no_{role}_cam_serial", ep_dir)

            cfg_path = os.path.join(s2m2_root, serial, "config.json")
            npz_path = os.path.join(s2m2_root, serial, "stereo_s2m2.npz")
            mp4_path = os.path.join(ep_dir, "recordings", "MP4", f"{serial}.mp4")

            if not (os.path.isfile(cfg_path) and os.path.isfile(npz_path) and os.path.isfile(mp4_path)):
                return ("__error__", f"missing_files_for_{role}", ep_dir)
            if os.path.getsize(npz_path) < MIN_NPZ_SIZE_BYTES:
                return ("__error__", f"npz_too_small_for_{role}", ep_dir)

            with open(cfg_path) as f:
                cfg = json.load(f)
            try:
                role_in_cfg = cfg["camera_role"]
                fx = float(cfg["calibration"]["left"]["fx"])
                baseline = float(cfg["calibration"]["stereo_baseline_m_used"])
            except (KeyError, TypeError, ValueError):
                return ("__error__", f"bad_config_for_{role}", ep_dir)

            if role_in_cfg != role:
                return ("__error__", f"role_mismatch_{role}_vs_{role_in_cfg}", ep_dir)

            cameras[role] = {
                "serial": str(serial),
                "fx": fx,
                "baseline": baseline,
                "mp4_path": mp4_path,
                "npz_path": npz_path,
                "config_path": cfg_path,
            }

        lang_entry = _AGG_ANN.get(uuid, {})
        lang_list = [
            lang_entry[k] for k in ("language_instruction1", "language_instruction2", "language_instruction3")
            if k in lang_entry and lang_entry[k]
        ]

        row = {
            "episode_uuid": uuid,
            "raw_dir": ep_dir,
            "lab": meta.get("lab"),
            "success": bool(meta.get("success", False)),
            "current_task": meta.get("current_task", ""),
            "trajectory_length": int(meta.get("trajectory_length", 0)),
            "cameras": cameras,
            "language_instructions": lang_list,
        }
        signal.alarm(0)
        return ("__row__", row)
    except _Timeout:
        signal.alarm(0)
        return ("__error__", "timeout", ep_dir)
    except Exception as e:
        signal.alarm(0)
        return ("__error__", repr(e), ep_dir)


def _atomic_write(payload_str, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tmp = output_path + ".tmp"
    with open(tmp, "w") as f:
        f.write(payload_str)
    os.replace(tmp, output_path)


def main():
    global RAW_ROOT, AGG_ANN_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", type=str, required=True,
                    help="root of the raw DROID download "
                         "(<raw_root>/<LAB>/{success,failure}/<DATE>/<TS>/)")
    ap.add_argument("--num_workers", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 16)))
    ap.add_argument("--output", type=str,
                    default="dataset_meta_info/droid_raw_ctrl/raw_index.jsonl")
    ap.add_argument("--checkpoint_every", type=int, default=10000)
    ap.add_argument("--val_every", type=int, default=100,
                    help="hash(uuid) %% val_every == val_every-1 -> val split (default 100)")
    args = ap.parse_args()
    RAW_ROOT = args.raw_root
    AGG_ANN_PATH = os.path.join(RAW_ROOT, "aggregated-annotations-030724.json")

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Loading aggregated annotations from {AGG_ANN_PATH}", flush=True)
    try:
        with open(AGG_ANN_PATH) as f:
            agg_ann = json.load(f)
        print(f"[{time.strftime('%H:%M:%S')}]   loaded {len(agg_ann)} language entries", flush=True)
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}]   FAILED ({e}); languages will be empty", flush=True)
        agg_ann = {}

    print(f"[{time.strftime('%H:%M:%S')}] Discovering episode dirs under {RAW_ROOT}", flush=True)
    ep_dirs = discover_episode_dirs()
    print(f"[{time.strftime('%H:%M:%S')}]   {len(ep_dirs)} candidate dirs ({time.time()-t0:.1f}s)", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] Pool: {args.num_workers} workers, timeout={PER_EPISODE_TIMEOUT_S}s/ep, ckpt every {args.checkpoint_every}", flush=True)

    rows = []
    errors_by_kind: dict[str, int] = {}
    done = 0
    t1 = time.time()
    pool_obj = {"pool": None}

    def _flush_checkpoint(reason: str):
        rows.sort(key=lambda r: r["episode_uuid"])
        out_lines = []
        meta = {
            "_meta": {
                "raw_root": RAW_ROOT,
                "n_ep_dirs_seen": len(ep_dirs),
                "n_episodes_indexed": len(rows),
                "n_errors": sum(errors_by_kind.values()),
                "errors_by_kind": dict(sorted(errors_by_kind.items(), key=lambda kv: -kv[1])),
                "val_every": args.val_every,
                "wall_seconds": round(time.time() - t0, 1),
                "checkpoint_reason": reason,
            },
        }
        out_lines.append(json.dumps(meta))
        for i, r in enumerate(rows):
            r = dict(r)  # don't mutate the in-memory list
            r["traj_id"] = i
            h = int(hashlib.md5(r["episode_uuid"].encode()).hexdigest(), 16)
            r["split"] = "val" if (h % args.val_every) == (args.val_every - 1) else "train"
            out_lines.append(json.dumps(r))
        _atomic_write("\n".join(out_lines) + "\n", args.output)

    def _on_term(signum, frame):
        print(f"[{time.strftime('%H:%M:%S')}] signal {signum}; flushing checkpoint", flush=True)
        _flush_checkpoint(f"signal_{signum}")
        if pool_obj["pool"] is not None:
            try:
                pool_obj["pool"].terminate()
            except Exception:
                pass
        os._exit(0)

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)
    signal.signal(signal.SIGUSR1, _on_term)

    with Pool(args.num_workers, initializer=_init_worker, initargs=(AGG_ANN_PATH,)) as pool:
        pool_obj["pool"] = pool
        for r in pool.imap_unordered(index_episode, ep_dirs, chunksize=16):
            done += 1
            tag = r[0]
            if tag == "__row__":
                rows.append(r[1])
            else:
                kind = r[1]
                errors_by_kind[kind] = errors_by_kind.get(kind, 0) + 1
                if errors_by_kind[kind] <= 5:
                    print(f"  [err:{kind}] {r[2]}", flush=True)
            if done % 5000 == 0:
                elapsed = time.time() - t1
                rate = done / max(elapsed, 1e-3)
                eta = (len(ep_dirs) - done) / max(rate, 1e-3)
                print(
                    f"[{time.strftime('%H:%M:%S')}]  {done}/{len(ep_dirs)} eps "
                    f"({rate:.1f} ep/s, {elapsed:.0f}s elapsed, ~{eta:.0f}s ETA, "
                    f"{len(rows)} indexed, {sum(errors_by_kind.values())} errors)",
                    flush=True,
                )
            if done % args.checkpoint_every == 0:
                _flush_checkpoint(f"periodic_at_{done}")
                print(f"[{time.strftime('%H:%M:%S')}]  checkpoint -> {args.output}", flush=True)

    print(
        f"\n[{time.strftime('%H:%M:%S')}] Done: {len(rows)} indexed, "
        f"{sum(errors_by_kind.values())} errors, total {time.time()-t0:.0f}s",
        flush=True,
    )
    print(f"  errors by kind: {dict(sorted(errors_by_kind.items(), key=lambda kv: -kv[1]))}", flush=True)
    _flush_checkpoint("completed")
    print(f"Wrote {args.output}", flush=True)

    # Print a brief summary
    n_val = sum(
        1 for r in rows
        if int(hashlib.md5(r["episode_uuid"].encode()).hexdigest(), 16) % args.val_every == args.val_every - 1
    )
    print(f"  split: train={len(rows) - n_val}, val={n_val}", flush=True)
    by_lab = {}
    n_with_lang = 0
    for r in rows:
        by_lab[r["lab"]] = by_lab.get(r["lab"], 0) + 1
        if r["language_instructions"]:
            n_with_lang += 1
    print(f"  with language annotations: {n_with_lang} / {len(rows)} ({100 * n_with_lang / max(len(rows), 1):.1f}%)", flush=True)
    print(f"  episodes per lab: {dict(sorted(by_lab.items()))}", flush=True)


if __name__ == "__main__":
    main()
