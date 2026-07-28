#!/usr/bin/env python3
"""sync_registry.py — sync registry.json derived fields from model manifests.

`models/<id>/model.json` is the source of truth for a model. `registry.json` is
the index consumers actually download: it repeats the fields a client needs to
choose a model (kind, engine, languages, capabilities, platforms, requirements,
license) so that browsing the catalog costs one HTTP request instead of one per
model. Those repeated fields are a mirror, and a mirror that drifts is worse
than no mirror at all — so this script is the only thing allowed to write them.

For each model entry in registry.json:
  - Recomputes integrity.manifestSha256 from the local manifest file
  - Mirrors the fields in SYNCED_FIELDS from the manifest into the entry
  - Drops a mirrored field the manifest no longer declares, so a field removed
    upstream cannot survive in the index as stale metadata
  - Updates the top-level `updated` timestamp when anything changed

It deliberately does NOT download artifacts. Artifact hashes are computed by the
submitter and checked during review; CI stays offline and fast. See
docs/REGISTRY-GUIDE.md § "Integrity philosophy".

Usage:
  python3 scripts/sync_registry.py          # apply changes
  python3 scripts/sync_registry.py --check  # exit 1 if any field would change
"""
import argparse
import datetime
import hashlib
import json
import pathlib
import sys

REGISTRY = pathlib.Path("registry.json")
MODELS_DIR = pathlib.Path("models")
MANIFEST_FILE = "model.json"

# Fields copied verbatim from the manifest into the index entry. Everything a
# consumer needs to pick a model without fetching every manifest lives here:
# `platforms` (what the registry vouches for), `requirements` (what it costs),
# and `license` (whether the consumer may legally use it) are as load-bearing
# for that decision as the descriptive fields are.
SYNCED_FIELDS = (
    "name",
    "summary",
    "version",
    "kind",
    "engine",
    "languages",
    "capabilities",
    "platforms",
    "requirements",
    "license",
    "trustStatus",
)

# Where a newly mirrored field is inserted, so it lands next to the metadata it
# belongs with instead of at the end of the entry, after the URL and integrity
# block. A registry.json diff is read by a human reviewer; the anchors keep an
# entry in one stable, readable order no matter which field arrived when.
FIELD_ANCHOR = {
    "name": "id",
    "summary": "name",
    "version": "summary",
    "kind": "version",
    "engine": "kind",
    "languages": "engine",
    "capabilities": "languages",
    "platforms": "capabilities",
    "requirements": "platforms",
    "license": "requirements",
    "trustStatus": "license",
}


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def move_after(entry: dict, field: str, anchor: str) -> None:
    """Reorder `entry` so `field` sits directly after `anchor`."""
    if field not in entry or anchor not in entry:
        return
    value = entry.pop(field)
    reordered = {}
    for key, existing in entry.items():
        reordered[key] = existing
        if key == anchor:
            reordered[field] = value
    entry.clear()
    entry.update(reordered)


def dir_from_url(url: str) -> str | None:
    """Extract the model directory name from a manifest URL."""
    if not isinstance(url, str):
        return None
    if "/models/" in url and url.endswith("/" + MANIFEST_FILE):
        return url.split("/models/")[1].split("/")[0]
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if any field would change (use as CI gate)",
    )
    args = parser.parse_args()

    registry = json.loads(REGISTRY.read_text())
    models = registry.get("models")
    if not isinstance(models, dict):
        print("registry.json: missing or malformed 'models' object")
        return 1

    changed = False

    for model_id, entry in models.items():
        if not isinstance(entry, dict):
            print(f"  SKIP {model_id}: entry is not an object")
            continue

        dir_name = dir_from_url(entry.get("manifest", ""))
        if not dir_name:
            print(f"  SKIP {model_id}: cannot derive local dir from manifest URL")
            continue

        manifest_path = MODELS_DIR / dir_name / MANIFEST_FILE
        if not manifest_path.exists():
            print(f"  SKIP {model_id}: {manifest_path} not found locally")
            continue

        new_sha = sha256_file(manifest_path)
        old_sha = entry.get("integrity", {}).get("manifestSha256", "")
        if new_sha != old_sha:
            print(f"  {model_id}: manifestSha256 updated")
            entry.setdefault("integrity", {})["manifestSha256"] = new_sha
            changed = True

        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as e:
            print(f"  SKIP {model_id}: {manifest_path} failed to parse: {e}")
            continue
        if not isinstance(manifest, dict):
            print(f"  SKIP {model_id}: {manifest_path} is not a JSON object")
            continue

        for field in SYNCED_FIELDS:
            if field not in manifest:
                # A mirrored value whose source is gone describes nothing, and
                # validate_registry.py fails on it — so this tool has to prune
                # in that direction too, or removing a field from a manifest
                # leaves an error that nothing but a hand edit can clear.
                if entry.pop(field, None) is not None:
                    print(f"  {model_id}: '{field}' dropped (manifest no longer declares it)")
                    changed = True
                continue
            if manifest[field] != entry.get(field):
                print(f"  {model_id}: '{field}' synced from manifest")
                is_new = field not in entry
                entry[field] = manifest[field]
                if is_new and field in FIELD_ANCHOR:
                    move_after(entry, field, FIELD_ANCHOR[field])
                changed = True

    if changed:
        if args.check:
            print("\nFAIL: registry.json is out of sync. Run scripts/sync_registry.py to fix.")
            return 1
        registry["updated"] = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        REGISTRY.write_text(json.dumps(registry, indent=2) + "\n")
        print("\nregistry.json updated.")
    else:
        print("All derived fields are already up to date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
