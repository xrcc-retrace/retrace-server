"""Build the labeled frame set the oracle/extractor evals run against.

For every procedure with clips on disk, cut and cache four frames per step —
start (0%), mid (50%), near_done (90%), done (true last frame) — under
evals/fixtures/frames/{procedure_id}/. `done` is byte-identical to the
reference the /verify endpoint uses; the fractional frames give positives and
negatives that are NOT the exact reference JPEG (so the oracle can't win by
matching identical bytes).

Optional higher-fidelity labels: drop real learner-attempt clips/images into
evals/fixtures/staged/{procedure_id}/ named step_{n}_{done,notdone,errorK}[_*].
Those are reported here and consumed by eval_oracle when present.

Zero manual labeling required — run it and it populates from what's on disk.

    python evals/harvest_frames.py [--procedure <id>]
"""

import argparse

import _common as c


def main() -> None:
    import shutil
    ap = argparse.ArgumentParser()
    ap.add_argument("--procedure", help="Harvest only this procedure id")
    ap.add_argument("--force", action="store_true",
                    help="Re-cut cached frames (use after re-extracting clips)")
    args = ap.parse_args()

    procs = c.procedures_with_clips()
    if args.procedure:
        procs = [p for p in procs if p["id"] == args.procedure]

    if args.force:
        for p in procs:
            shutil.rmtree(c.FRAMES_DIR / p["id"], ignore_errors=True)

    c.header(f"Harvesting frames — {len(procs)} procedure(s) with clips")
    total_steps = total_frames = staged_count = 0
    verif = {"reliable": 0, "marginal": 0, "none": 0, None: 0}

    for p in procs:
        steps_with_clips = 0
        for s in p["steps"]:
            clip = c.clip_path_for(p["id"], s["step_number"])
            if not clip.exists():
                continue
            steps_with_clips += 1
            total_steps += 1
            verif[s.get("visual_verifiability")] = verif.get(s.get("visual_verifiability"), 0) + 1
            frames = c.harvest_step_frames(p["id"], s["step_number"], clip)
            total_frames += len(frames)
            staged = c.staged_step_frames(p["id"], s["step_number"])
            staged_count += sum(len(v) for v in staged.values())
        title = (p.get("title") or "(untitled)")[:44]
        print(f"  {p['id'][:8]}  {title:46} {steps_with_clips} step(s)")

    print()
    print(f"  steps harvested : {total_steps}")
    print(f"  frames cached   : {total_frames}  (→ {c.FRAMES_DIR})")
    print(f"  staged extras   : {staged_count}  (evals/fixtures/staged/)")
    print(f"  verifiability   : "
          f"reliable={verif.get('reliable',0)} marginal={verif.get('marginal',0)} "
          f"none={verif.get('none',0)} unset={verif.get(None,0)}")
    if verif.get(None, 0):
        print("  note: 'unset' steps predate the migration in this DB snapshot; "
              "they are treated as 'reliable' downstream.")


if __name__ == "__main__":
    main()
