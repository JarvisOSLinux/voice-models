#!/usr/bin/env python3
"""validate_registry.py — PR-gate validation for the voice-models registry.

Static checks (always run):
  - registry.json parses, is an object, and declares the index format `version`
    this validator understands plus a well-formed `updated` timestamp
  - every entry: map key == entry.id == models/<dir>/ name, and the id matches
    the `<engine>.<model-name>` pattern
  - required fields present and no others: an index entry carries its identity,
    its manifest URL, the mirror and the integrity block — nothing hand-written
  - kind / engine / trustStatus / platforms / requirements.accelerator are
    inside their enums
  - languages look like BCP-47 codes (or the literal "multilingual")
  - license is present with an spdx id and a url — a model nobody may legally
    use is not a catalog entry, it is a trap
  - the manifest URL is https and resolves to an existing models/<dir>/model.json,
    and the manifest's own `manifest` field agrees with the URL the index used
  - integrity.manifestSha256 is present and matches the manifest file bytes
  - every mirrored field in the index equals the manifest it was copied from —
    an index entry is a mirror, and a hand-widened mirror is how a consumer ends
    up trusting a claim no manifest ever made
  - source shape: exactly one of `artifacts` / `huggingface`, with the fields
    that type actually needs (per-item sha256 + sizeBytes, or repo + revision)
  - no orphan directory: every models/<dir>/ is referenced by an entry

Warnings (reported, never fatal):
  - a capability outside the documented v1 vocabulary (the vocabulary is meant
    to grow; an unknown value is a review prompt, not a defect)
  - an stt/wake entry whose manifest declares no audio.sampleRateHz

Trust-promotion gate (when --base is given):
  - compares trustStatus per entry against the base registry.json
  - promotion to 'official' (or an entry added directly as 'official') fails
    unless --approval-label-present is passed. This is the rule that stops a
    submitter from self-assigning the official tier.
  - a changed license or a widened `platforms` list on an existing entry is
    reported as a warning: both change what a consumer is being told they may
    do, on a model they may already have deployed.
  - a changed artifact url/sha256 (or huggingface repo/revision) on an existing
    entry is reported the same way: that pair decides which bytes a consumer
    downloads and runs, so it needs the "a human read this" signal above all.
    Precise when --base-manifests is given; a softer "the manifest changed"
    prompt from the recorded manifestSha256 when it is not.

Malformed input is a validation error, never a traceback and never a pass: this
gate runs on submissions from strangers, so every value is type-checked before
it is used.

Usage:
  python3 scripts/validate_registry.py
  python3 scripts/validate_registry.py --base base_registry.json
  python3 scripts/validate_registry.py --base base_registry.json --base-manifests base_manifests/
  python3 scripts/validate_registry.py --base base_registry.json --approval-label-present
"""
import argparse
import json
import pathlib
import re
import sys

from sync_registry import (
    ALLOWED_ENTRY_FIELDS,
    INDEX_FORMAT_VERSION,
    MANIFEST_FILE,
    MODELS_DIR,
    REGISTRY,
    SYNCED_FIELDS,
    TIMESTAMP_PATTERN,
    dir_from_url,
    sha256_file,
)

ALLOWED_KINDS = {"stt", "tts", "wake"}

# The engines this registry knows how to describe. Unknown engine is an ERROR,
# not a warning: `engine` decides how a consumer unpacks and loads the model, so
# an entry naming an engine no consumer implements is undeliverable. Growth is
# by PR — add the engine here and document its packaging in the guide.
ALLOWED_ENGINES = {"vosk", "faster-whisper", "whisper-cpp", "piper", "openwakeword"}

ALLOWED_TRUST = {"community", "official", "deprecated", "removed"}
ALLOWED_PLATFORMS = {"linux", "darwin", "windows"}
ALLOWED_ACCELERATORS = {"cpu", "cuda", "any"}
ALLOWED_ARTIFACT_FORMATS = {"zip", "tar.gz", "onnx", "tflite", "json", "bin"}
ALLOWED_SOURCE_TYPES = {"artifacts", "huggingface"}

# Documented v1 capability vocabulary. Unknown values warn rather than fail —
# unlike `engine`, a capability nobody recognizes degrades to "not advertised",
# it does not break the download.
KNOWN_CAPABILITIES = {
    "streaming",
    "word-timestamps",
    "punctuation",
    "speaker-diarization",
    "voice-cloning",
    "multi-speaker",
    "gpu-acceleration",
}

REQUIRED_REQUIREMENTS = ("diskMb", "ramMb", "accelerator")

# Present in the index entry: its identity, where its manifest lives, and every
# mirrored field. The manifest carries all of those plus the fetch instructions
# (`upstream`, `source`) — the mirrored ones are checked by validate_mirror, so
# they are not repeated here.
REQUIRED_ENTRY_FIELDS = ("id", "manifest") + SYNCED_FIELDS
REQUIRED_MANIFEST_FIELDS = ("id", "manifest", "upstream", "source")

# Top-level keys of the index document. Anything else is a warning: a typo like
# `schemaVersion` is invisible to a consumer reading `version`, and it should be
# visible to a reviewer instead of silently doing nothing.
KNOWN_TOP_LEVEL_FIELDS = ("version", "updated", "models")

# `<engine>.<model-name>`, both halves from the same alphabet. The engine half
# has to admit a hyphen, because two of the engines this registry vouches for —
# `faster-whisper` and `whisper-cpp` — carry one in their names.
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*(\.[a-z0-9][a-z0-9-]*)+$")
LANGUAGE_PATTERN = re.compile(r"^[a-z]{2,3}(-[a-z0-9]{2,8})*$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
HF_REPO_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")

# Numeric audio fields, checked when present. `audio` is intentionally open —
# engines describe frames differently — so only the shared vocabulary is typed.
AUDIO_INT_FIELDS = ("sampleRateHz", "channels", "sampleWidthBits", "frameSamples", "frameMs")


def annotate(level: str, msg: str) -> None:
    # GitHub Actions annotation; harmless plain text when run locally.
    print(f"::{level}::{msg}" if level in ("error", "warning") else msg)


def check_enum(where: str, container: dict, field: str, allowed: set, errors: list) -> None:
    """Check a scalar enum field, tolerating any JSON type without crashing."""
    if field not in container:
        return
    value = container[field]
    # isinstance first: a list or object is unhashable, so a bare set lookup
    # would abort the whole gate with a traceback instead of reporting this
    # entry and carrying on through the rest of the registry.
    if not isinstance(value, str) or value not in allowed:
        errors.append(f"{where}: {field} {value!r} not in {sorted(allowed)}")


def check_string_list(
    where: str,
    container: dict,
    field: str,
    errors: list,
    *,
    pattern: re.Pattern | None = None,
    allowed: set | None = None,
    unknown: str = "error",
    warnings: list | None = None,
    extra_values: set = frozenset(),
) -> None:
    """Check a non-empty array of lowercase strings against a pattern or enum."""
    if field not in container:
        return
    values = container[field]
    if not isinstance(values, list) or not values:
        errors.append(f"{where}: '{field}' must be a non-empty array")
        return

    for value in values:
        if not isinstance(value, str) or not value:
            errors.append(f"{where}: {field} entry {value!r} must be a non-empty string")
            continue
        if value in extra_values:
            continue
        if allowed is not None and value not in allowed:
            # Both branches spell the message out in full rather than sharing a
            # variable: selftest_registry.py audits every reported string
            # against a case that asserts on it, and it can only read literals.
            if unknown == "warning" and warnings is not None:
                warnings.append(
                    f"{where}: {field} {value!r} not in {sorted(allowed)} — documented "
                    f"vocabulary only; add it to docs/model-reference.md if it is real"
                )
            else:
                errors.append(f"{where}: {field} {value!r} not in {sorted(allowed)}")
            continue
        if pattern is not None and not pattern.match(value):
            errors.append(f"{where}: {field} {value!r} is not a well-formed value")


def validate_requirements(where: str, entry: dict, errors: list) -> None:
    """Check the approximate cost block a consumer budgets against.

    These numbers are documented as approximate, but they still have to be
    numbers: a consumer that cannot compare diskMb across entries cannot choose
    a model for a constrained device, which is most of why this registry exists.
    """
    if "requirements" not in entry:
        return
    requirements = entry["requirements"]
    if not isinstance(requirements, dict):
        errors.append(f"{where}: 'requirements' must be an object")
        return

    for field in REQUIRED_REQUIREMENTS:
        if field not in requirements:
            errors.append(f"{where}: requirements is missing '{field}'")

    for field in ("diskMb", "ramMb"):
        if field not in requirements:
            continue
        value = requirements[field]
        # bool is an int in Python; JSON true would otherwise pass as 1.
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append(f"{where}: requirements.{field} {value!r} must be a positive integer")

    check_enum(f"{where}: requirements", requirements, "accelerator", ALLOWED_ACCELERATORS, errors)


def validate_license(where: str, entry: dict, errors: list) -> None:
    """Check the license block.

    Required, and required to be specific. Speech models — TTS voices above all
    — routinely ship under dataset licenses with real restrictions, and a
    catalog that omits them hands its consumers a legal problem disguised as a
    convenience.
    """
    if "license" not in entry:
        return
    license_block = entry["license"]
    if not isinstance(license_block, dict):
        errors.append(f"{where}: 'license' must be an object with 'spdx' and 'url'")
        return

    for field in ("spdx", "url"):
        value = license_block.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(
                f"{where}: license.{field} is required and must be a non-empty string "
                f"(use a LicenseRef- id when upstream declares no SPDX identifier)"
            )

    if "notes" in license_block:
        notes = license_block["notes"]
        if not isinstance(notes, str) or not notes.strip():
            errors.append(f"{where}: license.notes must be a non-empty string when present")


def validate_source(where: str, manifest: dict, errors: list) -> None:
    """Check the fetch instructions — exactly one source type, fully specified.

    `artifacts` is the verifiable form: every item carries a sha256 the consumer
    checks after download. `huggingface` hands the download to the engine, which
    means no artifact hash is possible — so `revision` is mandatory there, as
    the only thing pinning what was actually vetted.
    """
    if "source" not in manifest:
        return
    source = manifest["source"]
    if not isinstance(source, dict):
        errors.append(f"{where}: 'source' must be an object")
        return

    source_type = source.get("type")
    if not isinstance(source_type, str) or source_type not in ALLOWED_SOURCE_TYPES:
        errors.append(f"{where}: source.type {source_type!r} not in {sorted(ALLOWED_SOURCE_TYPES)}")
        return

    # Exactly one shape: a source carrying the other type's keys is ambiguous
    # about where the bytes come from, and a consumer would have to guess.
    foreign = {"artifacts": ("repo", "revision"), "huggingface": ("items",)}[source_type]
    for field in foreign:
        if field in source:
            errors.append(
                f"{where}: source.type is '{source_type}' but carries '{field}' — "
                f"an entry declares exactly one source shape"
            )

    if source_type == "huggingface":
        repo = source.get("repo")
        if not isinstance(repo, str) or not HF_REPO_PATTERN.match(repo):
            errors.append(f"{where}: source.repo {repo!r} must look like 'org/name'")
        revision = source.get("revision")
        if not isinstance(revision, str) or not revision.strip():
            errors.append(
                f"{where}: source.revision is required for a huggingface source — "
                f"the engine downloads these weights itself, so the revision is the "
                f"only thing pinning what was vetted"
            )
        engine_managed = source.get("engineManaged")
        if not isinstance(engine_managed, bool):
            errors.append(f"{where}: source.engineManaged must be true or false")
        return

    items = source.get("items")
    if not isinstance(items, list) or not items:
        errors.append(f"{where}: source.items must be a non-empty array")
        return

    for position, item in enumerate(items):
        at = f"{where}: source.items[{position}]"
        if not isinstance(item, dict):
            errors.append(f"{at}: must be an object")
            continue

        url = item.get("url")
        if not isinstance(url, str) or not url.startswith("https://"):
            errors.append(f"{at}: url {url!r} must be an https:// URL")

        digest = item.get("sha256")
        if not isinstance(digest, str) or not SHA256_PATTERN.match(digest):
            errors.append(
                f"{at}: sha256 {digest!r} must be 64 lowercase hex characters — "
                f"this is the only thing a consumer can verify the download against"
            )

        size = item.get("sizeBytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            errors.append(f"{at}: sizeBytes {size!r} must be a positive integer")

        check_enum(at, item, "format", ALLOWED_ARTIFACT_FORMATS, errors)
        if "format" not in item:
            errors.append(f"{at}: missing 'format'")

        if not isinstance(item.get("unpack"), bool):
            errors.append(f"{at}: unpack must be true or false")


def validate_upstream(where: str, manifest: dict, errors: list) -> None:
    if "upstream" not in manifest:
        return
    upstream = manifest["upstream"]
    if not isinstance(upstream, dict):
        errors.append(f"{where}: 'upstream' must be an object with a 'project' URL")
        return
    for field in ("project", "homepage"):
        if field == "homepage" and field not in upstream:
            continue
        value = upstream.get(field)
        if not isinstance(value, str) or not value.startswith("https://"):
            errors.append(f"{where}: upstream.{field} {value!r} must be an https:// URL")


def validate_audio(where: str, manifest: dict, entry: dict, errors: list, warnings: list) -> None:
    """Check the audio contract a consumer has to feed the model.

    Missing audio metadata on an stt/wake model is a warning, not an error: it
    is recoverable (the engine's own defaults usually apply) but it is exactly
    the field whose absence makes a consumer guess a sample rate, and a wrong
    guess is silent garbage rather than a crash.
    """
    if "audio" not in manifest:
        if entry.get("kind") in ("stt", "wake"):
            warnings.append(
                f"{where}: kind '{entry.get('kind')}' with no 'audio' block — "
                f"a consumer has to guess the input sample rate"
            )
        return

    audio = manifest["audio"]
    if not isinstance(audio, dict):
        errors.append(f"{where}: 'audio' must be an object")
        return

    for field in AUDIO_INT_FIELDS:
        if field not in audio:
            continue
        value = audio[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append(f"{where}: audio.{field} {value!r} must be a positive integer")

    if "sampleRateHz" not in audio:
        errors.append(f"{where}: audio block present but 'sampleRateHz' missing")

    if "sampleFormat" in audio:
        sample_format = audio["sampleFormat"]
        if not isinstance(sample_format, str) or not sample_format:
            errors.append(f"{where}: audio.sampleFormat must be a non-empty string")


def validate_mirror(where: str, manifest_path: pathlib.Path, manifest: dict, entry: dict, errors: list) -> None:
    """Check every mirrored field against the manifest it was copied from.

    The index is what consumers read; the manifest is what reviewers read. If
    those two disagree, the review happened against a document nobody fetches.
    """
    for field in SYNCED_FIELDS:
        if field not in manifest:
            errors.append(
                f"{where}: manifest {manifest_path} declares no '{field}' — the index "
                f"mirrors it (add it to the manifest and run sync_registry.py)"
            )
        elif manifest[field] is None:
            # Named here rather than left to the typed check on the mirrored
            # copy, so the error points at the manifest — the file a
            # contributor is allowed to edit — instead of at registry.json,
            # which CONTRIBUTING.md tells them never to touch.
            errors.append(
                f"{where}: manifest {manifest_path} declares '{field}' as null — a "
                f"mirrored field must carry a value"
            )
        elif manifest[field] != entry.get(field):
            errors.append(
                f"{where}: '{field}' {entry.get(field)!r} does not match manifest "
                f"{manifest[field]!r} — run sync_registry.py"
            )


def validate_entry_fields(where: str, model_id: str, entry: dict, errors: list, warnings: list) -> None:
    for field in REQUIRED_ENTRY_FIELDS:
        if field not in entry:
            errors.append(f"{where}: missing required field '{field}'")

    # The mirror invariant runs in both directions: validate_mirror catches a
    # mirrored field that disagrees with its manifest, and this catches one no
    # manifest declares at all. Without it an entry can carry a whole rogue
    # `source` block — new fetch instructions, straight past review.
    for field in sorted(set(entry) - set(ALLOWED_ENTRY_FIELDS)):
        errors.append(
            f"{where}: index-only field '{field}' — an index entry mirrors the manifest; "
            f"declare it in models/<id>/{MANIFEST_FILE} (and add it to SYNCED_FIELDS) "
            f"rather than hand-writing it into registry.json"
        )

    # Key presence, not value-is-not-None: JSON null is not absence, and
    # treating it as absence let `"id": null` satisfy both this check and the
    # required-fields loop above. Absence is already reported there, so gating
    # on presence adds no duplicate error.
    if "id" in entry and entry["id"] != model_id:
        errors.append(f"{where}: entry.id {entry['id']!r} != map key '{model_id}'")

    if not ID_PATTERN.match(model_id):
        errors.append(
            f"{where}: id '{model_id}' must match {ID_PATTERN.pattern} "
            f"(lowercase dotted, e.g. 'piper.en-us-lessac-medium')"
        )

    for field in ("name", "summary", "version"):
        value = entry.get(field)
        if field in entry and (not isinstance(value, str) or not value.strip()):
            errors.append(f"{where}: '{field}' must be a non-empty string")

    check_enum(where, entry, "kind", ALLOWED_KINDS, errors)
    check_enum(where, entry, "engine", ALLOWED_ENGINES, errors)
    check_enum(where, entry, "trustStatus", ALLOWED_TRUST, errors)

    check_string_list(
        where, entry, "languages", errors, pattern=LANGUAGE_PATTERN, extra_values={"multilingual"}
    )
    check_string_list(
        where,
        entry,
        "capabilities",
        errors,
        allowed=KNOWN_CAPABILITIES,
        unknown="warning",
        warnings=warnings,
    )
    check_string_list(where, entry, "platforms", errors, allowed=ALLOWED_PLATFORMS)
    if "platforms" not in entry:
        errors.append(
            f"{where}: missing 'platforms' — every entry must declare the platforms "
            f"it was actually vetted on (add the field to the manifest and run "
            f"sync_registry.py)"
        )

    validate_requirements(where, entry, errors)
    validate_license(where, entry, errors)


def validate_top_level(registry: dict, errors: list, warnings: list) -> None:
    """Check the two fields that describe the index document itself.

    `version` exists so a consumer can branch on how to parse the document. An
    index with no version, or a version this validator does not know, is one a
    consumer cannot safely parse — and neither is something a submission should
    be able to introduce, so a format bump is a deliberate edit to both
    INDEX_FORMAT_VERSION and registry.json.
    """
    version = registry.get("version")
    if not isinstance(version, str) or not version.strip():
        errors.append(
            f"registry.json: 'version' {version!r} must be a non-empty string — it is the "
            f"index format version a consumer branches on (run sync_registry.py)"
        )
    elif version != INDEX_FORMAT_VERSION:
        errors.append(
            f"registry.json: 'version' is {version!r} but this validator implements "
            f"{INDEX_FORMAT_VERSION!r} — a format bump changes both, in one deliberate PR"
        )

    updated = registry.get("updated")
    if not isinstance(updated, str) or not TIMESTAMP_PATTERN.match(updated):
        errors.append(
            f"registry.json: 'updated' {updated!r} must be a UTC timestamp shaped "
            f"YYYY-MM-DDTHH:MM:SSZ — run sync_registry.py"
        )

    for field in sorted(set(registry) - set(KNOWN_TOP_LEVEL_FIELDS)):
        warnings.append(
            f"registry.json: unknown top-level field '{field}' — nothing reads it; "
            f"the index carries {list(KNOWN_TOP_LEVEL_FIELDS)}"
        )


def validate_static(registry: dict, errors: list, warnings: list) -> None:
    models = registry.get("models")
    if not isinstance(models, dict):
        errors.append("registry.json: missing or malformed 'models' object")
        return

    for model_id, entry in models.items():
        where = f"models['{model_id}']"

        if not isinstance(entry, dict):
            errors.append(f"{where}: entry must be an object")
            continue

        validate_entry_fields(where, model_id, entry, errors, warnings)

        manifest_url = entry.get("manifest", "")
        # The one URL a consumer dereferences to obtain the reviewed document,
        # so it gets the scheme requirement artifact URLs already have. No host
        # is pinned: a fork served from elsewhere is a valid registry.
        if not isinstance(manifest_url, str) or not manifest_url.startswith("https://"):
            errors.append(
                f"{where}: manifest {manifest_url!r} must be an https:// URL — this is the "
                f"URL a consumer dereferences to get the reviewed manifest"
            )
        dir_name = dir_from_url(manifest_url)
        if not dir_name:
            errors.append(
                f"{where}: cannot derive a local dir from manifest URL {manifest_url!r} — "
                f"it must end in '/models/<id>/{MANIFEST_FILE}'"
            )
            continue

        if dir_name != model_id:
            errors.append(
                f"{where}: manifest URL points at models/{dir_name}/ but the entry id is "
                f"'{model_id}' — map key, entry.id and directory name must agree"
            )

        manifest_path = MODELS_DIR / dir_name / MANIFEST_FILE
        if not manifest_path.exists():
            errors.append(f"{where}: manifest {manifest_path} not found")
            continue

        integrity = entry.get("integrity")
        if not isinstance(integrity, dict):
            errors.append(f"{where}: 'integrity' must be an object — run sync_registry.py")
            integrity = {}
        recorded = integrity.get("manifestSha256", "")
        try:
            actual = sha256_file(manifest_path)
        except OSError as e:
            # Anything that exists but is not a readable regular file — a
            # directory of that name is representable in a PR. Reported and
            # skipped, so the run still lists everything else it found.
            errors.append(f"{where}: manifest {manifest_path} is not a readable file: {e}")
            continue
        if not recorded:
            errors.append(f"{where}: integrity.manifestSha256 missing — run sync_registry.py")
        elif not isinstance(recorded, str) or recorded != actual:
            errors.append(
                f"{where}: integrity.manifestSha256 stale "
                f"(recorded {str(recorded)[:12]}…, actual {actual[:12]}…) — run sync_registry.py"
            )

        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
            errors.append(f"{where}: manifest {manifest_path} failed to parse: {e}")
            continue
        if not isinstance(manifest, dict):
            errors.append(f"{where}: manifest {manifest_path} is not a JSON object")
            continue

        for field in REQUIRED_MANIFEST_FIELDS:
            if field not in manifest:
                errors.append(f"{where}: manifest {manifest_path} missing required field '{field}'")

        if "id" in manifest and manifest["id"] != model_id:
            errors.append(
                f"{where}: manifest id {manifest['id']!r} != '{model_id}' — the manifest "
                f"in models/{model_id}/ describes a different model"
            )

        # The manifest states its own canonical URL, so a manifest copied out of
        # another entry's directory is caught even when everything else lines up.
        if "manifest" in manifest and manifest["manifest"] != manifest_url:
            errors.append(
                f"{where}: manifest declares its URL as {manifest['manifest']!r} but the "
                f"index fetches it from {manifest_url!r}"
            )

        validate_mirror(where, manifest_path, manifest, entry, errors)
        validate_upstream(where, manifest, errors)
        validate_source(where, manifest, errors)
        validate_audio(where, manifest, entry, errors, warnings)


def validate_no_orphan_dirs(registry: dict, errors: list) -> None:
    """Flag any models/<dir>/ not referenced by a registry entry.

    Deleting an entry but leaving its directory is the one removal mistake the
    entry-driven checks above cannot see — this closes that gap.
    """
    if not MODELS_DIR.is_dir():
        return

    models = registry.get("models")
    referenced = {
        dir_from_url(entry.get("manifest", ""))
        for entry in (models.values() if isinstance(models, dict) else [])
        if isinstance(entry, dict)
    }
    referenced.discard(None)

    for child in sorted(MODELS_DIR.iterdir()):
        if child.is_dir() and child.name not in referenced:
            errors.append(
                f"models/{child.name}/: orphan directory — no registry entry references "
                f"it (remove the directory, or add its entry)"
            )


def load_json(path: pathlib.Path):
    """Read a JSON document, or None if it is missing or unreadable."""
    try:
        return json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def recorded_manifest_sha(entry: dict):
    integrity = entry.get("integrity")
    return integrity.get("manifestSha256") if isinstance(integrity, dict) else None


def fetch_pins(manifest) -> dict:
    """The values that decide which bytes a consumer ends up running.

    Keyed by where they live in the manifest, so a diff against the base
    branch's copy names the item that changed. For `artifacts` that is the
    (url, sha256) pair per item; for `huggingface` it is (repo, revision), the
    only provenance pin that source type has.
    """
    if not isinstance(manifest, dict):
        return {}
    source = manifest.get("source")
    if not isinstance(source, dict):
        return {}

    if source.get("type") == "artifacts":
        items = source.get("items")
        if not isinstance(items, list):
            return {}
        return {
            f"source.items[{position}]": (item.get("url"), item.get("sha256"))
            for position, item in enumerate(items)
            if isinstance(item, dict)
        }

    if source.get("type") == "huggingface":
        return {"source": (source.get("repo"), source.get("revision"))}

    return {}


def review_changed_fetch_instructions(
    model_id: str, entry: dict, base_entry: dict, base_manifests: pathlib.Path | None
) -> None:
    """Annotate a change to the bytes a consumer would fetch for an entry.

    A swapped `url` + `sha256` is the change that most needs a human to look:
    it redirects every future consumer of an already-deployed model at
    different bytes, and the registry cannot verify a digest itself. This is
    the analogue of mcp-registry's setup-script annotation.

    Precise when the base manifests are available; otherwise the recorded
    manifestSha256 still says "something in this manifest changed", which is
    enough to ask a reviewer to re-read `source`.
    """
    head_manifest = None
    dir_name = dir_from_url(entry.get("manifest", ""))
    if dir_name:
        head_manifest = load_json(MODELS_DIR / dir_name / MANIFEST_FILE)

    base_manifest = None
    # An id is validated against ID_PATTERN elsewhere, but this reads a path
    # built from it, so it does not get to contain a separator either way.
    if base_manifests is not None and "/" not in model_id and model_id not in (".", ".."):
        base_manifest = load_json(base_manifests / f"{model_id}.json")

    if base_manifest is None or head_manifest is None:
        head_sha = recorded_manifest_sha(entry)
        base_sha = recorded_manifest_sha(base_entry)
        if head_sha and base_sha and head_sha != base_sha:
            annotate(
                "warning",
                f"{model_id}: manifest changed — re-read its 'source' block before merge; "
                f"a changed artifact url/sha256 is what decides which bytes a consumer runs",
            )
        return

    head_pins = fetch_pins(head_manifest)
    base_pins = fetch_pins(base_manifest)
    digest_changed = False

    for label, head_value in head_pins.items():
        base_value = base_pins.get(label)
        if base_value == head_value:
            continue
        if base_value is not None and head_value[1] != base_value[1]:
            digest_changed = True
        what = "repo/revision" if label == "source" else "url/sha256"
        annotate(
            "warning",
            f"{model_id}: {label} {what} changed — confirm the digest against a fresh "
            f"download before merge (CONTRIBUTING.md § 'Do not edit a hash without "
            f"re-downloading')",
        )

    if digest_changed and entry.get("version") == base_entry.get("version"):
        annotate(
            "warning",
            f"{model_id}: the artifacts changed but 'version' is still "
            f"{entry.get('version')!r} — a real upstream release bumps the version, so a "
            f"silent digest change on an unchanged version needs an explicit explanation",
        )


def validate_promotions(
    registry: dict,
    base: dict,
    approval: bool,
    errors: list,
    base_manifests: pathlib.Path | None = None,
) -> None:
    base_models = base.get("models", {}) if isinstance(base, dict) else {}
    if not isinstance(base_models, dict):
        base_models = {}
    models = registry.get("models")
    if not isinstance(models, dict):
        return

    promotions = []

    for model_id, entry in models.items():
        if not isinstance(entry, dict):
            continue
        base_entry = base_models.get(model_id)
        if not isinstance(base_entry, dict):
            base_entry = None

        head_trust = entry.get("trustStatus")
        base_trust = base_entry.get("trustStatus") if base_entry else None
        if head_trust == "official" and base_trust != "official":
            promotions.append(model_id)

        if base_entry is None:
            continue

        # A relicense reaches users who already deployed the model, so it needs
        # the same "a human read this" signal a code change would get.
        head_license = entry.get("license")
        base_license = base_entry.get("license")
        if isinstance(head_license, dict) and isinstance(base_license, dict):
            if head_license.get("spdx") != base_license.get("spdx"):
                annotate(
                    "warning",
                    f"{model_id}: license changed "
                    f"{base_license.get('spdx')!r} -> {head_license.get('spdx')!r} — "
                    f"confirm the new terms against upstream before merge",
                )

        # `platforms` records what this registry actually exercised, so widening
        # it is a new claim about work someone did, not a metadata tidy-up.
        head_platforms = entry.get("platforms")
        base_platforms = base_entry.get("platforms")
        if isinstance(head_platforms, list) and isinstance(base_platforms, list):
            added = [
                p for p in head_platforms if isinstance(p, str) and p not in base_platforms
            ]
            if added:
                annotate(
                    "warning",
                    f"{model_id}: 'platforms' widened by {added} — confirm the model was "
                    f"exercised there, since this field records vetting, not upstream's claim",
                )

        review_changed_fetch_instructions(model_id, entry, base_entry, base_manifests)

    if promotions:
        listed = ", ".join(promotions)
        if approval:
            annotate("notice", f"trustStatus→official approved by maintainer label for: {listed}")
        else:
            errors.append(
                "trustStatus raised to 'official' without maintainer approval for: "
                f"{listed}. A maintainer must apply the 'trust-approved' label "
                "(see docs/REGISTRY-GUIDE.md § Trust model)."
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", help="Path to the base branch's registry.json for diff checks")
    parser.add_argument(
        "--base-manifests",
        help=(
            "Directory of the base branch's manifests as <model-id>.json, so a changed "
            "artifact url/sha256 can be named precisely rather than inferred from the hash"
        ),
    )
    parser.add_argument(
        "--approval-label-present",
        action="store_true",
        help="Set when the PR carries the maintainer 'trust-approved' label",
    )
    args = parser.parse_args()

    errors: list = []
    warnings: list = []

    try:
        registry = json.loads(REGISTRY.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        annotate("error", f"registry.json failed to parse: {e}")
        return 1
    if not isinstance(registry, dict):
        annotate("error", "registry.json: top level must be a JSON object")
        return 1

    validate_top_level(registry, errors, warnings)
    validate_static(registry, errors, warnings)
    validate_no_orphan_dirs(registry, errors)

    if args.base:
        base = load_json(pathlib.Path(args.base))
        if not isinstance(base, dict):
            base = {}
        base_manifests = pathlib.Path(args.base_manifests) if args.base_manifests else None
        validate_promotions(
            registry, base, args.approval_label_present, errors, base_manifests
        )

    for warn in warnings:
        annotate("warning", warn)

    for err in errors:
        annotate("error", err)

    if errors:
        print(f"\nFAIL: {len(errors)} validation error(s).")
        return 1
    print("OK: registry validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
