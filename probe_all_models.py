"""List EVERY model available to our API key across v1alpha, v1beta, v1.

Shows capabilities (generateContent, bidiGenerateContent, embedContent, etc.)
so we can see the full picture of what our key is provisioned for.
"""

import os
import sys
from pathlib import Path

env_path = Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from google import genai

API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not API_KEY:
    print("GEMINI_API_KEY not set")
    sys.exit(1)


def capabilities(m) -> list[str]:
    methods = getattr(m, "supported_actions", None) \
        or getattr(m, "supported_generation_methods", None) \
        or []
    return [str(x) for x in methods]


def main():
    for version in ("v1alpha", "v1beta", "v1"):
        print("=" * 80)
        print(f"models.list() on {version}")
        print("=" * 80)
        client = genai.Client(api_key=API_KEY, http_options={"api_version": version})
        try:
            models = sorted(client.models.list(), key=lambda m: m.name)
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            continue

        # Group by top-level capability
        by_cap: dict[str, list] = {
            "bidi (Live)": [],
            "generate": [],
            "embed": [],
            "other": [],
        }
        for m in models:
            caps = capabilities(m)
            joined = ",".join(caps).lower()
            if "bidigenerate" in joined or "bidi" in joined:
                by_cap["bidi (Live)"].append((m.name, caps))
            elif "generatecontent" in joined or "generate" in joined:
                by_cap["generate"].append((m.name, caps))
            elif "embed" in joined:
                by_cap["embed"].append((m.name, caps))
            else:
                by_cap["other"].append((m.name, caps))

        total = sum(len(v) for v in by_cap.values())
        print(f"Total: {total} models\n")
        for cap, items in by_cap.items():
            if not items:
                continue
            print(f"  {cap} ({len(items)}):")
            for name, caps in items:
                # Strip "models/" prefix to reduce noise
                short = name.replace("models/", "")
                print(f"    {short}")
            print()


if __name__ == "__main__":
    main()
