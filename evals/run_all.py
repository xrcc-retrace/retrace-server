"""One-command eval report — runs the harness end to end and aggregates gates.

Runs each eval as a subprocess (decoupled), streams its output, scrapes the
PASS/MARGINAL/FAIL gate lines, and writes a combined JSON report so a single run
yields pass rates without watching any video.

    python evals/run_all.py                       # offline: harvest + oracle + extractor
    python evals/run_all.py --reextract           # also re-run the live extractor prompt
    python evals/run_all.py --server http://localhost:8000   # + endpoint probe
    python evals/run_all.py --live                # + Live injected-event probe (tokens)
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

EVALS = Path(__file__).resolve().parent
REPORTS = EVALS / "reports"
ANSI = re.compile(r"\033\[[0-9;]*m")
GATE = re.compile(r"(PASS|MARGINAL|FAIL)\s+(.+?):\s+([\d.]+)%")


def _run(name: str, argv: list[str]) -> dict:
    print(f"\n\033[1m### {name}\033[0m  ({' '.join(argv)})", flush=True)
    proc = subprocess.run([sys.executable, *argv], capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    print(out)
    gates = [{"verdict": m.group(1), "metric": ANSI.sub("", m.group(2)).strip(),
              "rate": float(m.group(3))} for m in GATE.finditer(ANSI.sub("", out))]
    return {"name": name, "exit_code": proc.returncode, "gates": gates}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reextract", action="store_true")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--server", help="Also run the endpoint probe against this base URL")
    ap.add_argument("--live", action="store_true", help="Also run the Live probe (spends tokens)")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    sections = []
    sections.append(_run("harvest", [str(EVALS / "harvest_frames.py")]))

    oracle_argv = [str(EVALS / "eval_oracle.py")]
    if args.limit:
        oracle_argv += ["--limit", str(args.limit)]
    sections.append(_run("eval_oracle", oracle_argv))

    ext_argv = [str(EVALS / "eval_extractor.py")]
    if args.reextract:
        ext_argv.append("--reextract")
    if args.judge:
        ext_argv.append("--judge")
    if args.limit:
        ext_argv += ["--limit", str(args.limit)]
    sections.append(_run("eval_extractor", ext_argv))

    if args.server:
        sections.append(_run("probe_oracle_endpoint",
                             [str(EVALS / "probe_oracle_endpoint.py"), "--server", args.server]))
    if args.live:
        sections.append(_run("probe_oracle_live", [str(EVALS / "probe_oracle_live.py")]))

    # aggregate
    REPORTS.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS / f"report_{int(time.time())}.json"
    report_path.write_text(json.dumps({"sections": sections}, indent=2))

    print("\n\033[1m" + "=" * 68 + "\n SUMMARY\n" + "=" * 68 + "\033[0m")
    all_gates = [(s["name"], g) for s in sections for g in s["gates"]]
    worst = {"PASS": 0, "MARGINAL": 0, "FAIL": 0}
    for name, g in all_gates:
        worst[g["verdict"]] = worst.get(g["verdict"], 0) + 1
        color = {"PASS": "\033[92m", "MARGINAL": "\033[93m", "FAIL": "\033[91m"}[g["verdict"]]
        print(f"  {color}{g['verdict']:8}\033[0m [{name}] {g['metric']}: {g['rate']:.1f}%")
    print(f"\n  totals: \033[92m{worst['PASS']} pass\033[0m  "
          f"\033[93m{worst['MARGINAL']} marginal\033[0m  \033[91m{worst['FAIL']} fail\033[0m")
    print(f"  report: {report_path}")


if __name__ == "__main__":
    main()
