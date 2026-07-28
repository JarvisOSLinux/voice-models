#!/usr/bin/env python3
"""selftest_registry.py — prove the registry gates actually fire.

This repo has no test suite, and a validator that has quietly stopped firing
looks exactly like a registry with nothing wrong. Each case below builds a
throwaway registry in a temp directory, runs the real sync and validate entry
points against it, and asserts on what they report — including the cases that
must stay silent, so a false positive is caught as readily as a missed one.

Covers: the sync mirror (fill, prune, anchor order, --check idempotency), the
enum gates (kind/engine/trustStatus/platforms/accelerator/format), the
requirement/license/source/audio shapes, identity agreement between map key,
entry, manifest and directory, orphan directories, malformed input reaching a
type-checked path instead of a traceback, and the trust-promotion gate.

Offline, stdlib only, writes nothing outside its temp directory.

Usage:
  python3 scripts/selftest_registry.py
"""
import contextlib
import copy
import hashlib
import io
import json
import os
import pathlib
import sys
import tempfile

import sync_registry
import validate_registry

MODEL_ID = "piper.demo-voice"
MANIFEST_URL = f"https://example.invalid/models/{MODEL_ID}/model.json"

ARTIFACT = {
    "url": "https://example.invalid/voice.tar.gz",
    "sha256": "a" * 64,
    "sizeBytes": 1024,
    "format": "tar.gz",
    "unpack": True,
}

HF_SOURCE = {
    "type": "huggingface",
    "repo": "Systran/faster-whisper-small",
    "revision": "536005ba43e2eda7e5b4ed5f7f8dcfb1b4d0f8dc",
    "engineManaged": True,
}

# The order sync's FIELD_ANCHOR chain should produce, whichever order the fields
# happen to arrive in. A reviewer reads these diffs; drift here is a real defect.
EXPECTED_ENTRY_ORDER = [
    "id",
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
    "manifest",
    "integrity",
]


def manifest(**extra):
    doc = {
        "id": MODEL_ID,
        "name": "Demo Voice",
        "summary": "Fixture model for the registry self-test",
        "version": "1.0.0",
        "kind": "tts",
        "engine": "piper",
        "languages": ["en-us"],
        "capabilities": ["streaming"],
        "platforms": ["linux"],
        "requirements": {"diskMb": 61, "ramMb": 191, "accelerator": "cpu"},
        "license": {"spdx": "MIT", "url": "https://example.invalid/LICENSE"},
        "trustStatus": "community",
        "manifest": MANIFEST_URL,
        "upstream": {"project": "https://example.invalid/demo"},
        "source": {"type": "artifacts", "items": [copy.deepcopy(ARTIFACT)]},
        "audio": {"sampleRateHz": 22050, "channels": 1},
    }
    doc.update(extra)
    return {k: v for k, v in doc.items() if v is not DROP}


class _Drop:
    def __repr__(self):
        return "<drop>"


# Sentinel for manifest(field=DROP): removes a key the default fixture sets.
DROP = _Drop()


@contextlib.contextmanager
def fixture(doc):
    """Build a one-model registry in a temp dir and run the scripts inside it."""
    saved_cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="voice-models-selftest-") as tmp:
        root = pathlib.Path(tmp)
        (root / "models" / MODEL_ID).mkdir(parents=True)
        write_manifest(root, doc)

        # The entry a contributor hand-writes: identity, trust tier and the
        # manifest URL. Everything else — including the integrity block — is
        # sync_registry.py's job, exactly as CONTRIBUTING describes it.
        (root / "registry.json").write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "updated": "2026-01-01T00:00:00Z",
                    "models": {
                        MODEL_ID: {
                            "id": MODEL_ID,
                            "name": "Demo Voice",
                            "summary": "Fixture model for the registry self-test",
                            "version": "1.0.0",
                            "kind": "tts",
                            "engine": "piper",
                            "trustStatus": "community",
                            "manifest": MANIFEST_URL,
                        }
                    },
                },
                indent=2,
            )
            + "\n"
        )
        os.chdir(root)
        try:
            yield root
        finally:
            os.chdir(saved_cwd)


def run(entry_point, argv):
    """Call a script's main() with argv, capturing its exit code and output."""
    buffer = io.StringIO()
    saved_argv = sys.argv
    sys.argv = argv
    try:
        with contextlib.redirect_stdout(buffer):
            code = entry_point()
    except SystemExit as exc:
        code = exc.code
    finally:
        sys.argv = saved_argv
    return code or 0, buffer.getvalue()


def sync(*args):
    return run(sync_registry.main, ["sync_registry.py", *args])


def validate(*args):
    return run(validate_registry.main, ["validate_registry.py", *args])


def read_registry(root):
    return json.loads((root / "registry.json").read_text())


def read_entry(root):
    return read_registry(root)["models"][MODEL_ID]


def write_entry(root, entry):
    registry = read_registry(root)
    registry["models"][MODEL_ID] = entry
    (root / "registry.json").write_text(json.dumps(registry, indent=2) + "\n")


def write_manifest(root, doc):
    (root / "models" / MODEL_ID / "model.json").write_text(json.dumps(doc, indent=2) + "\n")


def write_base(root, mutate):
    """Snapshot registry.json as a `--base` file with `mutate` applied to the entry."""
    base = read_registry(root)
    mutate(base["models"][MODEL_ID])
    path = root / "base_registry.json"
    path.write_text(json.dumps(base, indent=2) + "\n")
    return str(path)


def expect_manifest_error(doc, needle, description):
    """Sync a fixture, then assert validate fails with `needle` in its output."""
    with fixture(doc):
        sync()
        code, out = validate()
        check(code == 1 and needle in out, f"{description}: {needle!r}")
        check("Traceback" not in out, f"{description}: reported, not crashed")


CASES = []
FAILURES = []
running = "?"


def case(fn):
    CASES.append(fn)
    return fn


def check(condition, description):
    print(f"    {'ok  ' if condition else 'FAIL'}  {description}")
    if not condition:
        FAILURES.append(f"{running}: {description}")


@case
def sync_mirrors_the_manifest_and_is_idempotent():
    doc = manifest()
    with fixture(doc) as root:
        code, _ = sync("--check")
        check(code == 1, "--check fails while the mirror and hash are missing")

        code, out = sync()
        check(code == 0, "sync succeeds")
        check("manifestSha256 updated" in out, "sync reports the new manifest hash")
        check("'license' synced from manifest" in out, "sync reports each mirrored field")

        entry = read_entry(root)
        expected = hashlib.sha256(
            (root / "models" / MODEL_ID / "model.json").read_bytes()
        ).hexdigest()
        check(
            entry["integrity"]["manifestSha256"] == expected,
            "manifestSha256 matches the manifest bytes",
        )
        check(entry["platforms"] == ["linux"], "platforms is mirrored into the index")
        check(entry["license"]["spdx"] == "MIT", "license is mirrored into the index")
        check(
            entry["requirements"]["accelerator"] == "cpu",
            "requirements is mirrored into the index",
        )
        check(
            read_registry(root)["updated"] != "2026-01-01T00:00:00Z",
            "the top-level 'updated' timestamp is bumped",
        )

        code, out = sync("--check")
        check(code == 0, "--check passes once synced")
        check("already up to date" in out, "a second sync reports nothing to do")

        code, out = validate()
        check(code == 0, "a fully synced fixture validates")
        check(
            "::error::" not in out and "::warning::" not in out,
            "nothing is reported for a clean entry",
        )


@case
def mirrored_fields_land_in_anchor_order():
    with fixture(manifest()) as root:
        sync()
        check(
            list(read_entry(root).keys()) == EXPECTED_ENTRY_ORDER,
            "newly mirrored fields are inserted at their anchors, not appended",
        )


@case
def a_field_removed_from_the_manifest_is_pruned():
    with fixture(manifest()) as root:
        sync()
        check("capabilities" in read_entry(root), "the mirrored field is there to begin with")

        write_manifest(root, manifest(capabilities=DROP))
        code, _ = sync("--check")
        check(code == 1, "--check sees the mirror the manifest no longer backs")

        code, out = sync()
        check(
            code == 0 and "'capabilities' dropped" in out,
            "sync reports dropping the unbacked mirror",
        )
        check("capabilities" not in read_entry(root), "the stale value is gone from the index")

        code, out = validate()
        check(code == 1, "an entry whose manifest dropped a mirrored field fails the gate")
        check("declares no 'capabilities'" in out, "the error names the missing manifest field")


@case
def unknown_engine_is_an_error():
    expect_manifest_error(
        manifest(engine="coqui"),
        "engine 'coqui' not in",
        "an engine outside the vouched set fails the gate",
    )


@case
def unknown_capability_is_a_warning_not_an_error():
    with fixture(manifest(capabilities=["streaming", "emotion-control"])):
        sync()
        code, out = validate()
        check(code == 0, "an unknown capability does not fail the gate")
        check(
            "::warning::" in out and "emotion-control" in out,
            "the warning names the undocumented capability",
        )
        check("::error::" not in out, "nothing else is reported as an error")


@case
def kind_and_trust_status_enums_fire():
    expect_manifest_error(
        manifest(kind="asr"), "kind 'asr' not in", "an out-of-enum kind fails the gate"
    )
    expect_manifest_error(
        manifest(trustStatus="verified"),
        "trustStatus 'verified' not in",
        "an out-of-enum trustStatus fails the gate",
    )


@case
def entry_platforms_gate_fires():
    for mutate, expect in [
        (lambda e: e.pop("platforms"), "missing 'platforms'"),
        (lambda e: e.update(platforms=[]), "non-empty array"),
        (lambda e: e.update(platforms="linux"), "non-empty array"),
        (lambda e: e.update(platforms=["Linux"]), "not in ['darwin', 'linux', 'windows']"),
        (lambda e: e.update(platforms=["linux", "darwin"]), "does not match manifest"),
    ]:
        with fixture(manifest()) as root:
            sync()
            entry = read_entry(root)
            mutate(entry)
            write_entry(root, entry)
            code, out = validate()
            check(code == 1 and expect in out, f"entry platforms gate fires: {expect!r}")


@case
def a_nested_platform_value_is_an_error_not_a_traceback():
    with fixture(manifest(platforms=["linux", ["windows"]])):
        sync()
        code, out = validate()
        check(code == 1, "an unhashable platform value fails the gate instead of crashing it")
        check("['windows']" in out, "the nested value is named")


@case
def languages_gate_fires():
    expect_manifest_error(
        manifest(languages=["EN"]),
        "not a well-formed value",
        "an upper-case language tag fails the gate",
    )
    expect_manifest_error(
        manifest(languages=[]), "non-empty array", "an empty languages list fails the gate"
    )
    with fixture(manifest(languages=["multilingual"])):
        sync()
        code, out = validate()
        check(code == 0, "'multilingual' is accepted alongside BCP-47 codes")
        check("::error::" not in out, "nothing is reported for the multilingual literal")


@case
def requirements_gate_fires():
    for mutation, expect, description in [
        ({"requirements": {"ramMb": 191, "accelerator": "cpu"}}, "missing 'diskMb'", "a missing key"),
        (
            {"requirements": {"diskMb": "61", "ramMb": 191, "accelerator": "cpu"}},
            "must be a positive integer",
            "a stringified size",
        ),
        (
            {"requirements": {"diskMb": 61, "ramMb": True, "accelerator": "cpu"}},
            "must be a positive integer",
            "a boolean masquerading as an int",
        ),
        (
            {"requirements": {"diskMb": 61, "ramMb": 0, "accelerator": "cpu"}},
            "must be a positive integer",
            "a zero footprint",
        ),
        (
            {"requirements": {"diskMb": 61, "ramMb": 191, "accelerator": "npu"}},
            "accelerator 'npu' not in",
            "an out-of-enum accelerator",
        ),
        ({"requirements": "small"}, "must be an object", "a scalar requirements block"),
    ]:
        expect_manifest_error(
            manifest(**mutation), expect, f"requirements gate fires on {description}"
        )


@case
def license_gate_fires():
    expect_manifest_error(
        manifest(license=DROP),
        "declares no 'license'",
        "a model with no license at all fails the gate",
    )
    expect_manifest_error(
        manifest(license={"url": "https://example.invalid/LICENSE"}),
        "license.spdx is required",
        "a license with no spdx id fails the gate",
    )
    expect_manifest_error(
        manifest(license={"spdx": "MIT"}),
        "license.url is required",
        "a license with no url fails the gate",
    )
    expect_manifest_error(
        manifest(license="MIT"), "must be an object", "a scalar license fails the gate"
    )
    expect_manifest_error(
        manifest(license={"spdx": "  ", "url": "https://example.invalid/LICENSE"}),
        "license.spdx is required",
        "a blank spdx id fails the gate",
    )

    with fixture(
        manifest(
            license={
                "spdx": "LicenseRef-Blizzard2013-Lessac",
                "url": "https://example.invalid/LICENSE",
                "notes": "Review the dataset license before commercial use.",
            }
        )
    ):
        sync()
        code, out = validate()
        check(code == 0, "a LicenseRef- id with notes is accepted")
        check("::error::" not in out, "nothing is reported for a non-SPDX upstream license")


@case
def source_gate_fires():
    for source, expect, description in [
        ({"type": "torrent", "items": []}, "source.type 'torrent' not in", "an unknown source type"),
        ({"items": [copy.deepcopy(ARTIFACT)]}, "source.type None not in", "a source with no type"),
        ({"type": "artifacts", "items": []}, "must be a non-empty array", "no artifacts at all"),
        (
            {"type": "artifacts", "items": [copy.deepcopy(ARTIFACT)], "repo": "org/name"},
            "an entry declares exactly one source shape",
            "an artifacts source carrying huggingface keys",
        ),
        (
            {**HF_SOURCE, "items": [copy.deepcopy(ARTIFACT)]},
            "an entry declares exactly one source shape",
            "a huggingface source carrying artifact items",
        ),
        (
            {**HF_SOURCE, "revision": ""},
            "source.revision is required",
            "a huggingface source with no pinned revision",
        ),
        (
            {**HF_SOURCE, "repo": "faster-whisper-small"},
            "must look like 'org/name'",
            "a huggingface repo with no org",
        ),
        (
            {k: v for k, v in HF_SOURCE.items() if k != "engineManaged"},
            "engineManaged must be true or false",
            "a huggingface source that does not say who downloads",
        ),
        ("https://example.invalid/voice.tar.gz", "must be an object", "a bare URL as the source"),
    ]:
        expect_manifest_error(manifest(source=source), expect, f"source gate fires on {description}")

    for mutation, expect, description in [
        ({"sha256": "notahash"}, "must be 64 lowercase hex", "an artifact with no usable hash"),
        ({"sha256": "A" * 64}, "must be 64 lowercase hex", "an upper-case hash"),
        ({"sizeBytes": "1024"}, "sizeBytes '1024' must be a positive integer", "a stringified size"),
        ({"format": "tarball"}, "format 'tarball' not in", "an unknown artifact format"),
        ({"url": "http://example.invalid/voice.tar.gz"}, "must be an https", "a plaintext URL"),
        ({"unpack": "yes"}, "unpack must be true or false", "a stringified unpack flag"),
    ]:
        item = {**copy.deepcopy(ARTIFACT), **mutation}
        expect_manifest_error(
            manifest(source={"type": "artifacts", "items": [item]}),
            expect,
            f"artifact gate fires on {description}",
        )

    for dropped, expect in [("format", "missing 'format'"), ("unpack", "unpack must be true")]:
        item = {k: v for k, v in copy.deepcopy(ARTIFACT).items() if k != dropped}
        expect_manifest_error(
            manifest(source={"type": "artifacts", "items": [item]}),
            expect,
            f"artifact gate fires on a missing '{dropped}'",
        )

    with fixture(manifest(kind="stt", engine="faster-whisper", source=HF_SOURCE)):
        sync()
        code, out = validate()
        check(code == 0, "a well-formed engine-managed huggingface source validates")
        check("::error::" not in out, "nothing is reported for a pinned HF revision")


@case
def audio_gate_fires():
    with fixture(manifest(kind="stt", engine="vosk", audio=DROP)):
        sync()
        code, out = validate()
        check(code == 0, "an stt model with no audio block does not fail the gate")
        check(
            "::warning::" in out and "guess the input sample rate" in out,
            "the warning says what a consumer is left to guess",
        )

    expect_manifest_error(
        manifest(audio={"channels": 1}),
        "'sampleRateHz' missing",
        "an audio block with no sample rate fails the gate",
    )
    expect_manifest_error(
        manifest(audio={"sampleRateHz": "16000"}),
        "must be a positive integer",
        "a stringified sample rate fails the gate",
    )


@case
def identity_must_agree_across_key_entry_manifest_and_dir():
    with fixture(manifest()) as root:
        sync()
        entry = read_entry(root)
        entry["id"] = "piper.something-else"
        write_entry(root, entry)
        code, out = validate()
        check(code == 1 and "!= map key" in out, "entry.id disagreeing with the map key fails")

    with fixture(manifest(id="piper.something-else")) as root:
        sync()
        code, out = validate()
        check(
            code == 1 and "describes a different model" in out,
            "a manifest whose id is not its directory fails",
        )

    with fixture(manifest()) as root:
        sync()
        entry = read_entry(root)
        entry["manifest"] = "https://example.invalid/models/piper.other/model.json"
        write_entry(root, entry)
        code, out = validate()
        check(
            code == 1 and "must agree" in out,
            "a manifest URL pointing at another directory fails",
        )

    with fixture(manifest()) as root:
        sync()
        entry = read_entry(root)
        entry["manifest"] = "https://example.invalid/models/piper.demo-voice/manifest.json"
        write_entry(root, entry)
        code, out = validate()
        check(
            code == 1 and "cannot derive a local dir" in out,
            "a manifest URL that is not a model.json fails",
        )

    with fixture(manifest()) as root:
        sync()
        write_manifest(root, manifest(manifest="https://elsewhere.invalid/registry/model.json"))
        code, out = validate()
        check(
            code == 1 and "declares its URL as" in out,
            "a manifest whose self-declared URL is not where the index fetches it fails",
        )


@case
def a_malformed_id_is_an_error():
    with fixture(manifest()) as root:
        sync()
        registry = read_registry(root)
        entry = registry["models"].pop(MODEL_ID)
        entry["id"] = "Piper.Demo_Voice"
        registry["models"]["Piper.Demo_Voice"] = entry
        (root / "registry.json").write_text(json.dumps(registry, indent=2) + "\n")
        code, out = validate()
        check(code == 1, "an id outside the naming pattern fails the gate")
        check("must match" in out, "the error shows the required id pattern")


@case
def an_orphan_model_directory_is_an_error():
    with fixture(manifest()) as root:
        sync()
        orphan = root / "models" / "vosk.left-behind"
        orphan.mkdir()
        (orphan / "model.json").write_text("{}\n")
        code, out = validate()
        check(code == 1, "a directory no entry references fails the gate")
        check(
            "orphan directory" in out and "vosk.left-behind" in out,
            "the error names the orphan directory",
        )


@case
def a_stale_manifest_hash_is_an_error():
    with fixture(manifest()) as root:
        sync()
        # Touch only a non-mirrored field, so the hash is the sole thing that
        # goes stale and the error cannot be attributed to anything else.
        write_manifest(root, manifest(notes="Re-verified on 2026-07-28."))
        code, out = validate()
        check(code == 1, "a manifest edited without re-syncing fails the gate")
        check("manifestSha256 stale" in out, "the error names the stale hash")

        code, _ = sync()
        code, out = validate()
        check(code == 0, "the two gates agree once sync has run")


@case
def a_malformed_registry_is_reported_not_crashed():
    with fixture(manifest()) as root:
        sync()
        registry = read_registry(root)
        registry["models"] = []
        (root / "registry.json").write_text(json.dumps(registry, indent=2) + "\n")
        code, out = validate()
        check(code == 1 and "malformed 'models'" in out, "a non-object 'models' fails the gate")

        code, out = sync()
        check(code == 1 and "malformed 'models'" in out, "sync refuses it too, without crashing")

    with fixture(manifest()) as root:
        sync()
        write_entry(root, "piper.demo-voice")
        code, out = validate()
        check(
            code == 1 and "entry must be an object" in out,
            "a scalar entry is reported instead of crashing the gate",
        )

    with fixture(manifest()) as root:
        sync()
        (root / "registry.json").write_text("{not json\n")
        code, out = validate()
        check(code == 1 and "failed to parse" in out, "unparseable JSON fails the gate cleanly")


@case
def the_promotion_gate_needs_a_maintainer_label():
    with fixture(manifest(trustStatus="official")) as root:
        sync()
        base = write_base(root, lambda entry: entry.update(trustStatus="community"))

        code, out = validate("--base", base)
        check(code == 1, "promotion to 'official' fails without the maintainer label")
        check("trust-approved" in out, "the error names the label a maintainer must apply")

        code, out = validate("--base", base, "--approval-label-present")
        check(code == 0, "the same promotion passes with the label")
        check("approved by maintainer label" in out, "the approved promotion is annotated")

        base = write_base(root, lambda entry: None)
        code, out = validate("--base", base)
        check(code == 0, "an entry that was already official is not re-gated")
        check("trust-approved" not in out, "no promotion is reported for an unchanged tier")

    with fixture(manifest(trustStatus="official")) as root:
        sync()
        base = write_base(root, lambda entry: None)
        base_doc = json.loads(pathlib.Path(base).read_text())
        base_doc["models"] = {}
        pathlib.Path(base).write_text(json.dumps(base_doc, indent=2) + "\n")
        code, out = validate("--base", base)
        check(
            code == 1 and "trust-approved" in out,
            "a brand-new entry added straight as 'official' is gated too",
        )


@case
def license_and_platform_changes_are_flagged_for_review():
    with fixture(manifest()) as root:
        sync()

        base = write_base(root, lambda entry: entry["license"].update(spdx="Apache-2.0"))
        code, out = validate("--base", base)
        check(code == 0, "a license change does not fail the gate")
        check(
            "license changed" in out and "'Apache-2.0' -> 'MIT'" in out,
            "the warning names both licenses",
        )

        base = write_base(root, lambda entry: entry.update(platforms=["linux", "darwin"]))
        code, out = validate("--base", base)
        check("widened" not in out, "narrowing 'platforms' is not flagged as a new claim")

        base = write_base(root, lambda entry: entry.update(platforms=[]))
        code, out = validate("--base", base)
        check(
            "'platforms' widened by ['linux']" in out,
            "a widened vetting claim is annotated for review",
        )

        base = write_base(root, lambda entry: None)
        code, out = validate("--base", base)
        check(
            "::warning::" not in out,
            "an unchanged entry is not re-flagged",
        )


def main() -> int:
    global running

    for fn in CASES:
        running = fn.__name__
        print(f"  {running}")
        fn()

    if FAILURES:
        print(f"\nFAIL: {len(FAILURES)} self-test assertion(s) failed.")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print(f"\nOK: registry self-test passed ({len(CASES)} cases).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
