#!/usr/bin/env python3
"""One-line health check of a training run from its log.

    python scripts/health.py runs/dqfd2.log [--min-speed 3500] [--q 50 200]

Mean speed over the last 20 log lines, NaN/inf anywhere in the last 200
lines, mean Q of the last 20 lines inside --q, free disk on the repo's
file system, whether the run's tmux session is alive. Prints one line,
exit code 1 if anything is off.
"""

import argparse
import os
import re
import shutil
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("log")
    p.add_argument("--session", default=None, help="tmux session (default: the log's name)")
    p.add_argument("--min-speed", type=float, default=3500)
    p.add_argument("--q", type=float, nargs=2, default=(-50, 200))
    p.add_argument("--min-disk-gb", type=float, default=10)
    args = p.parse_args()
    with open(args.log, errors="replace") as f:
        lines = f.readlines()[-200:]
    steps_lines = [line for line in lines if re.match(r"\s*\d+ steps ", line)][-20:]
    speeds = [float(m.group(1).replace(",", "")) for line in steps_lines
              if (m := re.search(r"([\d,]+) steps/s", line))]
    qs = [float(m.group(1)) for line in steps_lines if (m := re.search(r"  Q (\S+)", line))]
    nan = any(re.search(r"\bnan\b|\binf\b", line, re.I) for line in lines)
    disk = shutil.disk_usage(ROOT).free / 1e9
    session = args.session or os.path.splitext(os.path.basename(args.log))[0]
    alive = subprocess.run(["tmux", "has-session", "-t", session],
                           capture_output=True).returncode == 0
    speed = sum(speeds) / len(speeds) if speeds else None
    q = sum(qs) / len(qs) if qs else None
    last = re.match(r"\s*(\d+) steps", steps_lines[-1]).group(1) if steps_lines else "?"
    problems = []
    if speed is None or speed < args.min_speed:
        problems.append("slow")
    if nan:
        problems.append("NaN")
    if q is None or not args.q[0] <= q <= args.q[1]:
        problems.append("Q out of range")
    if disk < args.min_disk_gb:
        problems.append("disk")
    if not alive:
        problems.append("session gone")
    print(f"{time.strftime('%H:%M')} {session}: {last} steps, "
          f"{speed or 0:,.0f} steps/s, Q {q if q is not None else float('nan'):.1f}, "
          f"NaN {'yes' if nan else 'no'}, disk {disk:.0f} GB, tmux {'up' if alive else 'GONE'}"
          + (f"  PROBLEMS: {', '.join(problems)}" if problems else "  ok"))
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
