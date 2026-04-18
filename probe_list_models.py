"""List all models available to our API key, filter for Live-capable ones.

Authoritative source of truth for "which model IDs actually exist for us"
when probe_live_models.py's curated candidate list comes up empty.
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


def main():
    for version in ("v1alpha", "v1beta", "v1"):
        print("=" * 80)
        print(f"models.list() on {version}")
        print("=" * 80)
        client = genai.Client(api_key=API_KEY, http_options={"api_version": version})
        try:
            models = list(client.models.list())
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            continue

        # Filter: only models that support bidiGenerateContent — the Live endpoint.
        live_models = []
        for m in models:
            methods = getattr(m, "supported_actions", None) or \
                      getattr(m, "supported_generation_methods", None) or []
            if any("bidi" in str(mth).lower() or "live" in str(mth).lower() for mth in methods):
                live_models.append((m.name, methods))

        if live_models:
            print(f"\nLive-capable ({len(live_models)}):")
            for name, methods in sorted(live_models):
                print(f"  {name}")
                print(f"    methods: {methods}")
        else:
            print("\nNo Live-capable models found via supported_actions filter.")

        # Also list everything that has "live" or "native-audio" in the name,
        # in case the supported_actions field isn't populated consistently.
        name_matches = [m for m in models
                        if "live" in m.name.lower() or "native-audio" in m.name.lower()]
        if name_matches:
            print(f"\nName-match (live/native-audio in {len(name_matches)} model names):")
            for m in sorted(name_matches, key=lambda x: x.name):
                print(f"  {m.name}")
        print()


if __name__ == "__main__":
    main()
