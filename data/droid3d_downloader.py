#!/usr/bin/env python3
"""Parallel, resumable downloader for datasets hosted on data.ciirc.cvut.cz.

Usage:
    read -rs TOKEN && export TOKEN               # paste the token, press Enter
    python3 droid3d_downloader.py --out ./droid-s2m2
    python3 droid3d_downloader.py --dry-run          # inspect layout, download nothing
    python3 droid3d_downloader.py --limit 100 --group-depth 4 --seed 0

The dataset is taken from the token unless --dataset is given. Python 3.8+,
standard library only. Interrupt and rerun at any time: completed files are
skipped and partial files resume. If the token expires mid-download, set a
new TOKEN and rerun.

Subsets: --limit N keeps N units, where a unit is a directory --group-depth
levels below the dataset root (0 = single files). Files above that depth,
such as a top-level README, are always kept. Units are chosen by the N
smallest seeded hashes, so the subset is a uniform random sample, identical
for everyone using the same seed, independent of manifest order, and nested:
the --limit 10 subset is contained in the --limit 100 subset.
"""
import argparse
import base64
import collections
import fnmatch
import hashlib
import heapq
import json
import os
import posixpath
import queue
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE_URL = "https://data.ciirc.cvut.cz"
CHUNK = 1 << 20          # 1 MiB
STAT_DEPTHS = (1, 2, 3, 4, 5)
WINDOWS = os.name == "nt"
WIN_BAD = re.compile(r'[<>:"|?*]')  # characters Windows forbids in file names
DIR_CAP = 1_000_000      # stop counting a depth's directories past this


class AuthError(Exception):
    """Token rejected: expired, revoked, or not valid for this dataset."""


def host_of(url):
    return urllib.parse.urlsplit(url).netloc.lower()


class SameHostRedirect(urllib.request.HTTPRedirectHandler):
    """Carry the bearer token across a redirect only if the host is unchanged."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        auth = req.unredirected_hdrs.get("Authorization")
        if new is not None and auth and host_of(newurl) == host_of(req.full_url):
            new.add_unredirected_header("Authorization", auth)
        return new


class State:
    def __init__(self, out):
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.counts = {"done": 0, "skipped": 0, "failed": 0}
        self.bytes = 0
        self.abort_reason = None
        self.failed_log = open(out / "failed.txt", "w", encoding="utf-8")

    def record(self, result, url):
        with self.lock:
            self.counts[result] += 1
            if result == "failed":
                self.failed_log.write(url + "\n")
                self.failed_log.flush()

    def add_bytes(self, n):
        with self.lock:
            self.bytes += n

    def abort(self, reason):
        with self.lock:
            if self.abort_reason is None:
                self.abort_reason = reason
        self.stop.set()


def token_claims(token):
    """Decode the JWT payload without verifying it (the server verifies)."""
    try:
        payload = token.split(".")[1]
        return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except Exception:
        return {}


def fetch(opener, url, dest, token, host, timeout, state=None, meter=None):
    """Download url to dest through dest.part. Returns 'skipped' or 'done'."""
    if dest.exists():
        return "skipped"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url)
    if host_of(url) == host:  # never send the token to another host
        req.add_unredirected_header("Authorization", "Bearer " + token)
    if offset:
        req.add_header("Range", "bytes=%d-" % offset)
    try:
        resp = opener.open(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise AuthError("HTTP %d" % e.code) from None
        if e.code == 416:  # stale .part; restart this file on retry
            part.unlink(missing_ok=True)
        raise
    with resp:
        if offset and resp.status != 206:  # server ignored Range; start over
            offset = 0
        length = resp.headers.get("Content-Length")
        expected = offset + int(length) if length is not None else None
        if meter is not None:
            meter.start(expected, offset)
        with open(part, "ab" if offset else "wb") as f:
            while True:
                buf = resp.read(CHUNK)
                if not buf:
                    break
                f.write(buf)
                if state is not None:
                    state.add_bytes(len(buf))
                if meter is not None:
                    meter.add(len(buf), buf.count(b"\n"))
    if expected is not None and part.stat().st_size != expected:
        raise IOError("truncated transfer, will resume")
    os.replace(part, dest)
    return "done"


def resolve(url, opts, base):
    """Return (absolute url, relative POSIX path), or (url, None) if unsafe."""
    simple = url.startswith(base) and "?" not in url and "#" not in url
    if not simple:  # relative or unusual URL: full parse (slow, rare)
        url = urllib.parse.urljoin(base, url)
    if "out" in opts:
        rel = posixpath.join(opts.get("dir", "").lstrip("/"), opts["out"])
    elif simple:  # fast path, equivalent to the branch below for these URLs
        rel = urllib.parse.unquote(url[len(base):])
    else:
        path = urllib.parse.unquote(urllib.parse.urlsplit(url).path)
        prefix = urllib.parse.urlsplit(base).path
        rel = path[len(prefix):] if path.startswith(prefix) else path.lstrip("/")
    rel = posixpath.normpath(rel.replace("\\", "/"))
    if rel.startswith("/") or rel == "." or rel.split("/")[0] == "..":
        return url, None
    return url, rel


def parse_manifest(path, base, warn=False, meter=None):
    """Yield (url, relpath). Accepts one URL per line or aria2c input format
    (URI line followed by indented out=/dir= option lines)."""
    def emit(url, opts):
        u, rel = resolve(url, opts, base)
        if rel is None and warn:
            print("Skipping unsafe path in manifest: %s" % u, file=sys.stderr)
        return u, rel

    if meter is not None:
        meter.start(os.path.getsize(path))
    pending = 0
    url, opts = None, {}
    with open(path, "rb") as f:
        for raw in f:
            if meter is not None:
                pending += len(raw)
                if pending >= 1 << 20:  # report once per MiB read
                    meter.add(pending)
                    pending = 0
            line = raw.decode("utf-8").rstrip("\r\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if line[0] in " \t":
                key, _, val = line.strip().partition("=")
                opts[key] = val
                continue
            if url is not None:
                u, rel = emit(url, opts)
                if rel is not None:
                    yield u, rel
            url, opts = line.split("\t")[0].strip(), {}
    if url is not None:
        u, rel = emit(url, opts)
        if rel is not None:
            yield u, rel
    if meter is not None:
        meter.add(pending)
        meter.close()


def group_key(rel, depth):
    """The unit a file belongs to: its directory `depth` levels below the root.
    With depth 0 every file is its own unit. Files above `depth` (e.g. a
    top-level README.md) belong to no unit (None) and are always kept."""
    if depth <= 0:
        return rel
    parts = rel.split("/")
    return "/".join(parts[:depth]) if len(parts) > depth else None


def unit_hash(seed, key):
    h = hashlib.blake2b(("%d\0%s" % (seed, key)).encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), "big")


def survey(entries, depth, limit, seed):
    """One pass: count files, units, directories per depth, and pick the
    `limit` units with the smallest seeded hashes (O(limit) memory)."""
    n_files = 0
    units = set() if depth > 0 else None
    dirs = {d: set() for d in STAT_DEPTHS}
    heap, chosen = [], set()  # heap holds (-hash, key): a max-heap on hash
    examples = []
    for _, rel in entries:
        n_files += 1
        if len(examples) < 5:
            examples.append(rel)
        key = group_key(rel, depth)
        if units is not None and key is not None:
            units.add(key)
        parts = rel.split("/")
        for d, s in dirs.items():
            if len(parts) > d and len(s) <= DIR_CAP:
                s.add("/".join(parts[:d]))
        if limit and key is not None and key not in chosen:
            h = unit_hash(seed, key)
            if len(heap) < limit:
                heapq.heappush(heap, (-h, key))
                chosen.add(key)
            elif h < -heap[0][0]:
                _, old = heapq.heapreplace(heap, (-h, key))
                chosen.discard(old)
                chosen.add(key)
    n_units = n_files if units is None else len(units)
    dir_counts = {d: len(s) for d, s in dirs.items()}
    return n_files, n_units, (chosen if limit else None), dir_counts, examples


def worker(q, opener, token, host, args, state):
    while True:
        item = q.get()
        if item is None:
            return
        url, dest = item
        if state.stop.is_set():
            continue
        result = "failed"
        for attempt in range(args.retries):
            try:
                result = fetch(opener, url, dest, token, host, args.timeout, state)
                break
            except AuthError as e:
                state.abort("token rejected (%s)" % e)
                result = None
                break
            except urllib.error.HTTPError as e:
                if e.code in (404, 410):
                    break
            except Exception:
                pass  # network error, timeout, truncation: retry with resume
            if state.stop.is_set():
                result = None
                break
            if attempt + 1 < args.retries:
                time.sleep(min(60, 2 ** attempt))
        if result is not None:
            state.record(result, url)


def fmt_bytes(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return ("%.0f %s" if unit == "B" else "%.1f %s") % (n, unit)
        n /= 1000.0


def fmt_time(s):
    s = int(s)
    return "%d:%02d:%02d" % (s // 3600, s % 3600 // 60, s % 60)


def bar_str(frac, width):
    enc = (getattr(sys.stdout, "encoding", None) or "").lower()
    if "utf" not in enc:
        fill = int(frac * width)
        return "#" * fill + "-" * (width - fill)
    blocks = " \u258f\u258e\u258d\u258c\u258b\u258a\u2589\u2588"
    full, rem = divmod(int(frac * width * 8), 8)
    return (blocks[8] * full + (blocks[rem] if full < width else "")).ljust(width)


def bar_line(prefix, frac, stats):
    """prefix, percentage, bar and stats on one line fitted to the terminal.
    frac None (size unknown) drops the percentage and bar."""
    cols = shutil.get_terminal_size((100, 20)).columns - 1
    head = prefix + ("%3.0f%% " % (100 * frac) if frac is not None else "")
    width = cols - len(head) - len(stats) - 4
    mid = "|%s|  " % bar_str(frac, width) if frac is not None and width >= 10 else ""
    return (head + mid + stats)[:cols].ljust(cols)


class Meter:
    """Progress for one sequential byte stream: the manifest download and the
    manifest scans. Redraws a bar on a terminal; when output is redirected,
    prints a line every 30 s and one when finished."""

    def __init__(self, label):
        self.label = label
        self.tty = sys.stdout.isatty()
        self.interval = 0.2 if self.tty else 30.0
        self.total, self.n, self.n0 = None, 0, 0
        self.lines = 0
        self.t0 = self.last = time.time()
        self.drawn = False

    def start(self, total, offset=0):
        self.total, self.n, self.n0, self.lines = total, offset, offset, 0
        self.t0 = self.last = time.time()
        if self.tty:
            self.draw()

    def add(self, k, lines=0):
        self.n += k
        self.lines += lines
        now = time.time()
        if now - self.last >= self.interval:
            self.last = now
            self.draw()

    def stats(self, finished):
        el = time.time() - self.t0
        rate = (self.n - self.n0) / el if el > 0 else 0.0
        size = fmt_bytes(self.n) + (" / " + fmt_bytes(self.total) if self.total else "")
        if finished:
            tail = "in " + fmt_time(el)
        elif self.total:
            tail = "ETA " + (fmt_time((self.total - self.n) / rate) if rate > 0 else "--:--:--")
        else:  # server sent no size: no percentage or ETA, so show elapsed time
            tail = fmt_time(el) + " elapsed"
        if not self.total and self.lines:
            size += "  %s lines" % format(self.lines, ",")
        frac = min(self.n / self.total, 1.0) if self.total else None
        return ("%s  %s/s  %s" % (size, fmt_bytes(rate), tail)).rstrip(), frac

    def draw(self, finished=False):
        stats, frac = self.stats(finished)
        if self.tty:
            sys.stdout.write("\r" + bar_line(self.label + "  ", frac, stats))
            sys.stdout.flush()
            self.drawn = True
        else:
            pct = "%3.0f%%  " % (100 * frac) if frac is not None else ""
            print("%s  %s%s" % (self.label, pct, stats), flush=True)

    def close(self, ok=True):
        if ok:
            self.draw(finished=True)
            if self.tty:
                sys.stdout.write("\n")
                sys.stdout.flush()
        elif self.tty and self.drawn:  # wipe the partial line before a retry or error
            sys.stdout.write("\r" + " " * (shutil.get_terminal_size((100, 20)).columns - 1) + "\r")
            sys.stdout.flush()


class Progress(threading.Thread):
    """Progress of the parallel file downloads. Bar on a terminal; a plain
    status line every 30 s when output is redirected (batch jobs, log files)."""

    WINDOW = 20.0  # seconds over which rates are measured

    def __init__(self, state, total):
        super().__init__(daemon=True)
        self.state, self.total = state, total
        self.tty = sys.stdout.isatty()
        self.interval = 0.5 if self.tty else 30.0
        self.halt = threading.Event()
        self.samples = collections.deque([(time.time(), 0, 0)])  # (time, bytes, downloaded files)

    def snapshot(self):
        with self.state.lock:
            return dict(self.state.counts), self.state.bytes

    def render(self):
        now = time.time()
        c, b = self.snapshot()
        moved = c["done"] + c["failed"]           # files actually transferred or given up
        n = moved + c["skipped"]
        self.samples.append((now, b, moved))
        while len(self.samples) > 2 and now - self.samples[0][0] > self.WINDOW:
            self.samples.popleft()
        t0, b0, m0 = self.samples[0]
        dt = now - t0
        rate = (b - b0) / dt if dt > 0 else 0.0
        frate = (moved - m0) / dt if dt > 0 else 0.0
        left = self.total - n
        eta = fmt_time(left / frate) if frate > 0 else ("0:00:00" if left == 0 else "--:--:--")
        frac = n / self.total if self.total else 1.0
        stats = "%d/%d files  %s  %s/s  ETA %s" % (n, self.total, fmt_bytes(b), fmt_bytes(rate), eta)
        extra = []
        if c["skipped"]:
            extra.append("skipped %d" % c["skipped"])
        if c["failed"]:
            extra.append("failed %d" % c["failed"])
        if extra:
            stats += "  (" + ", ".join(extra) + ")"
        if not self.tty:
            return "%3.0f%%  %s" % (100 * frac, stats)
        return bar_line("", frac, stats)

    def run(self):
        while not self.halt.wait(self.interval):
            line = self.render()
            if self.tty:
                sys.stdout.write("\r" + line)
                sys.stdout.flush()
            else:
                print(line, flush=True)

    def finish(self):
        self.halt.set()
        self.join()
        line = self.render()
        print(("\r" + line) if self.tty else line, flush=True)


def utc(ts):
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(ts))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", help="dataset name (default: the dataset named in the token)")
    ap.add_argument("--out", help="output directory (default: ./<dataset>)")
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel downloads (default 8; please stay at or below 16)")
    ap.add_argument("--retries", type=int, default=8, help="attempts per file (default 8)")
    ap.add_argument("--timeout", type=float, default=60, help="seconds per request (default 60)")
    sub = ap.add_argument_group("subsets")
    sub.add_argument("--include", action="append", metavar="GLOB",
                     help="only files whose path matches GLOB, e.g. 'val/*' or '*/depth/*'; "
                          "repeatable; '*' also matches '/'")
    sub.add_argument("--limit", type=int, metavar="N", help="keep only N units")
    sub.add_argument("--group-depth", type=int, default=0, metavar="D",
                     help="a unit is a directory D levels below the root (default 0 = single files); "
                          "files above depth D, such as a top-level README, are always kept")
    sub.add_argument("--seed", type=int, default=0, help="seed for choosing units (default 0)")
    sub.add_argument("--dry-run", action="store_true",
                     help="fetch the manifest, print layout and selection, download nothing")
    ap.add_argument("--base-url", default=BASE_URL, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.limit is not None and args.limit < 1:
        sys.exit("--limit must be at least 1")
    if args.group_depth < 0:
        sys.exit("--group-depth must be 0 or more")

    token = os.environ.get("TOKEN", "").strip()
    if not token:
        sys.exit("TOKEN is not set. Run:  read -rs TOKEN && export TOKEN")
    claims = token_claims(token)
    dataset = args.dataset or claims.get("dataset")
    if not dataset:
        sys.exit("The token does not name a dataset; pass --dataset NAME.")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", dataset):
        sys.exit("Invalid dataset name: %r" % dataset)
    if claims.get("dataset") and claims["dataset"] != dataset:
        sys.exit("This token is for dataset '%s', not '%s'." % (claims["dataset"], dataset))
    exp = claims.get("exp")
    if isinstance(exp, (int, float)):
        if exp <= time.time():
            sys.exit("This token expired at %s. Request a new one." % utc(exp))
        print("Dataset %s, token valid until %s." % (dataset, utc(exp)), flush=True)

    base = "%s/ds2/%s/" % (args.base_url.rstrip("/"), dataset)
    host = host_of(base)
    out = Path(args.out or dataset)
    out.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(SameHostRedirect)

    manifest = out / "manifest.txt"
    if manifest.exists():
        print("Using existing %s (delete it to fetch a fresh copy)." % manifest, flush=True)
    for attempt in range(args.retries):
        if manifest.exists():
            break
        meter = Meter("Downloading manifest")
        try:
            fetch(opener, base + "manifest.txt", manifest, token, host, args.timeout, meter=meter)
            meter.close()
            break
        except AuthError as e:
            meter.close(ok=False)
            sys.exit("Token rejected (%s). It may have expired or be for another dataset." % e)
        except Exception as e:
            meter.close(ok=False)
            if attempt + 1 == args.retries:
                sys.exit("Could not download the manifest: %s" % e)
            time.sleep(min(60, 2 ** attempt))

    includes = args.include or []

    def entries(warn=False, meter=None):
        for url, rel in parse_manifest(manifest, base, warn, meter):
            if not includes or any(fnmatch.fnmatchcase(rel, g) for g in includes):
                yield url, rel

    depth = args.group_depth
    n_files, n_units, chosen, dir_counts, examples = survey(
        entries(warn=True, meter=Meter("Scanning manifest")), depth, args.limit, args.seed)

    def keep(rel):
        if chosen is None:
            return True
        key = group_key(rel, depth)
        return key is None or key in chosen

    subset = bool(includes or args.limit)
    total = n_files
    if subset:  # second pass: count the selection and record it
        total, examples = 0, []
        with open(out / "selection.txt", "w", encoding="utf-8") as sel_log:
            for _, rel in entries(meter=Meter("Applying selection")):
                if keep(rel):
                    total += 1
                    sel_log.write(rel + "\n")
                    if len(examples) < 5:
                        examples.append(rel)

    print("Manifest: %d files%s." % (n_files, " match --include" if includes else ""))
    if args.limit:
        print("Selected %d of %d units at --group-depth %d (seed %d): %d files."
              % (len(chosen), n_units, depth, args.seed, total))
    if subset:
        print("Selected paths are listed in %s." % (out / "selection.txt"))

    if WINDOWS:
        bad = next((rel for _, rel in entries() if keep(rel) and WIN_BAD.search(rel)), None)
        if bad:
            sys.exit("These paths are not valid Windows file names, e.g.\n  %s\n"
                     "Download on Linux or macOS, or inside WSL." % bad)

    if args.dry_run:
        print("Directories per depth (pick --group-depth from these):")
        for d, c in dir_counts.items():
            print("  depth %d: %s" % (d, ">%d" % DIR_CAP if c > DIR_CAP else c))
        print("Example paths:")
        for rel in examples:
            print("  " + rel)
        print("Dry run: nothing downloaded.")
        return

    state = State(out)
    print("Downloading %d files to %s" % (total, out.resolve()), flush=True)
    q = queue.Queue(maxsize=args.workers * 4)
    threads = [threading.Thread(target=worker, args=(q, opener, token, host, args, state),
                                daemon=True) for _ in range(args.workers)]
    for t in threads:
        t.start()
    progress = Progress(state, total)
    progress.start()

    try:
        for url, rel in entries():
            if state.stop.is_set():
                break
            if keep(rel):
                q.put((url, out / rel))
        for _ in threads:
            q.put(None)
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted. Rerun the same command to resume.")

    progress.finish()
    state.stop.set()
    state.failed_log.close()
    c = state.counts
    print("Finished: done %d, skipped %d, failed %d of %d."
          % (c["done"], c["skipped"], c["failed"], total))
    if state.abort_reason:
        sys.exit("Stopped early: %s. Set a new TOKEN and rerun to resume." % state.abort_reason)
    if c["failed"]:
        sys.exit("Failed URLs are listed in %s. Rerun to retry them." % (out / "failed.txt"))


if __name__ == "__main__":
    main()
