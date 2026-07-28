# Registry Guide

How `voice-models` is put together, what guarantees it makes, and how a consumer
resolves an entry into files on disk.

For the per-field spec of a manifest, see
[`model-reference.md`](model-reference.md). This document is about the registry
as a whole.

---

## Table of contents

1. [What this registry is](#what-this-registry-is)
2. [Two files, one source of truth](#two-files-one-source-of-truth)
3. [registry.json format](#registryjson-format)
4. [Integrity philosophy](#integrity-philosophy)
5. [Consuming the registry: fetch and verify](#consuming-the-registry-fetch-and-verify)
6. [Trust model](#trust-model)
7. [Validation and tooling](#validation-and-tooling)
8. [Hosting and forking](#hosting-and-forking)

---

## What this registry is

A catalog of **local speech models** — speech-to-text, text-to-speech, and
wake-word — that any project can consume.

It exists because local speech has no equivalent of the two access mechanisms
that LLMs converged on. With an LLM you either run a local runtime or you speak
an OpenAI-shaped API, and one abstraction covers both. Speech models have no such
convergence: every engine packages its weights differently, and each one is
discovered a different way.

| Engine | What "the model" actually is | Where it comes from |
|--------|------------------------------|---------------------|
| Vosk | A zip that unpacks to a model directory | A vendor download page |
| faster-whisper | CTranslate2 weights | Hugging Face, fetched by the engine |
| Piper | An ONNX voice + its `.onnx.json` config | Release assets, historically several different repos |
| whisper.cpp | A single GGML/GGUF file | Release assets or a converted upload |
| openWakeWord | An ONNX classifier + two shared preprocessor models | Release assets, some fetched at runtime |

So a project that wants to offer "pick a voice" has to hard-code five different
acquisition strategies, and has nowhere to look up the things it actually needs
to decide: **what exists, what can it do, what does it cost, where does it
legally come from, and how do I fetch and verify it.** This registry answers
those five questions in one JSON document.

It is deliberately **project-agnostic**. There are no consumer-specific fields in
the schema, no assumptions about a consumer's configuration system, and no
required runtime. A consumer can be a Python daemon, a Rust CLI, or a shell
script with `curl` and `sha256sum` — the [walkthrough
below](#consuming-the-registry-fetch-and-verify) is the last of those.

---

## Two files, one source of truth

```
models/<id>/model.json    the manifest — source of truth for one model
registry.json             the index — everything, in one request
```

The index repeats the fields a consumer needs in order to *choose*: `kind`,
`engine`, `languages`, `capabilities`, `platforms`, `requirements`, `license`,
`trustStatus`, plus the display metadata. That repetition is the whole point —
browsing the catalog costs one HTTP request instead of one per model, which
matters for a consumer on a slow link deciding between forty voices.

Repetition also means the two can drift, and a drifted mirror is worse than no
mirror: the reviewer reads the manifest, the consumer reads the index, and if
they disagree the review happened against a document nobody fetches. So:

- **`scripts/sync_registry.py` is the only thing that writes a mirrored field.**
  It copies from manifest to index, prunes an index field the manifest no longer
  declares, and recomputes `integrity.manifestSha256`.
- **`scripts/validate_registry.py` fails any PR where a mirrored field differs
  from its manifest**, in either direction, along with any stale hash.
- CI runs `sync_registry.py --check`, so a PR that edits a manifest without
  re-syncing cannot merge.

Never hand-edit a mirrored field in `registry.json`. Edit the manifest and run
sync.

---

## registry.json format

```json
{
  "version": "1.0",
  "updated": "2026-07-28T00:00:00Z",
  "models": {
    "piper.en-us-lessac-medium": {
      "id": "piper.en-us-lessac-medium",
      "name": "Piper Lessac (medium, en-US)",
      "summary": "Single-speaker U.S. English VITS neural text-to-speech voice …",
      "version": "0.0.2",
      "kind": "tts",
      "engine": "piper",
      "languages": ["en-us"],
      "capabilities": ["streaming"],
      "platforms": ["linux"],
      "requirements": { "diskMb": 61, "ramMb": 191, "accelerator": "cpu" },
      "license": { "spdx": "LicenseRef-Blizzard2013-Lessac", "url": "…", "notes": "…" },
      "trustStatus": "community",
      "manifest": "https://raw.githubusercontent.com/JarvisOSLinux/voice-models/main/models/piper.en-us-lessac-medium/model.json",
      "integrity": { "manifestSha256": "…" }
    }
  }
}
```

| Top-level field | Description |
|-----------------|-------------|
| `version` | Format version of the index itself. `"1.0"`. |
| `updated` | UTC timestamp, bumped by `sync_registry.py` whenever anything changed. |
| `models` | Map of model id → entry. The key is the id. |

Entry fields are documented in [`model-reference.md`](model-reference.md);
everything except `manifest` and `integrity` is mirrored from the manifest.

Entries are written in one fixed field order, and `sync_registry.py` inserts a
newly mirrored field at its anchor rather than appending it. This is purely for
the humans: a registry diff is read during review, and a field that lands next to
the metadata it belongs with is a field a reviewer can check.

---

## Integrity philosophy

Three different things need verifying, and they are verified in three different
places. Being explicit about which is which is what keeps CI fast and the
guarantees honest.

### 1. The manifest — verified by the registry, in CI

`integrity.manifestSha256` is the SHA-256 of the manifest file's bytes. It binds
the index entry to the exact manifest a reviewer approved, so a consumer that
fetches the manifest separately can confirm it got the reviewed document and not
a later edit. `sync_registry.py` computes it; `validate_registry.py` fails a
stale one. It is never edited by hand.

### 2. Artifacts — verified by the *consumer*, at download time

Every item in an `artifacts` source carries a required `sha256`. **The registry
does not download artifacts, and CI never will.** The hashes are computed by the
submitter on the machine where they verified the model, and checked by a reviewer
against upstream during review.

This is a deliberate trade, and the reasoning is worth stating plainly:

- **CI stays offline and fast.** A gate that downloads models is a gate that
  takes twenty minutes, burns bandwidth on every push, and fails when an upstream
  host has a bad afternoon. A registry whose CI is flaky gets merged around.
- **A hash computed in CI would verify nothing anyway.** If CI downloads the file
  and records its hash, the recorded hash is by construction whatever was served
  that day — including a compromised or silently re-uploaded asset. The hash is
  only meaningful because a *human* asserted "this is the file I ran, and here is
  its digest."
- **The value lands where the risk is.** The consumer is the one running the
  bytes. It downloads, hashes, compares, and refuses on mismatch. That check is
  where a swapped release asset or a corrupted transfer is actually caught, and
  it works offline, forever, without the registry being reachable.

So the validator checks that a `sha256` is *present and well-formed* (64
lowercase hex) and that `sizeBytes` is a positive integer. It does not and cannot
check that the digest matches reality — that is the reviewer's job at merge time
and the consumer's job at fetch time.

### 3. Engine-managed weights — pinned, not hashed

A `huggingface` source has no per-file hash, because the consumer never handles
the individual files: `faster-whisper` resolves and caches them internally. What
the registry can pin is **which** revision was vetted, so `revision` is mandatory
there. It is the entire provenance story for that source type, which is why the
validator rejects an empty one even though the engine would happily default to
`main`.

### What none of this covers

The registry vouches for identity and provenance: *this file, from this project,
under this license, exercised on these platforms.* It does not sandbox a model,
audit weights for backdoors, or make any claim about output quality. A verified
hash means you got the bytes the submitter got. It does not mean those bytes are
benign — that is what `trustStatus` and human review are for.

---

## Consuming the registry: fetch and verify

Four steps: resolve the entry, verify the manifest against the index, fetch and
verify the artifacts, point the engine at the result.

### The whole thing in plain `curl` + `sha256sum`

Nothing but a shell. This fetches the Piper Lessac voice and verifies it:

```bash
set -euo pipefail

REGISTRY=https://raw.githubusercontent.com/JarvisOSLinux/voice-models/main/registry.json
MODEL=piper.en-us-lessac-medium

# 1. Get the index.
curl -fsSL "$REGISTRY" -o registry.json

# 2. Fetch the manifest and check it against the hash the index recorded.
#    (The two values below are read out of registry.json — with jq, or by eye.)
MANIFEST_URL=https://raw.githubusercontent.com/JarvisOSLinux/voice-models/main/models/$MODEL/model.json
MANIFEST_SHA=$(python3 -c 'import json;print(json.load(open("registry.json"))["models"]["'"$MODEL"'"]["integrity"]["manifestSha256"])')

curl -fsSL "$MANIFEST_URL" -o model.json
echo "$MANIFEST_SHA  model.json" | sha256sum --check --strict
# model.json: OK

# 3. Fetch the artifact and verify it against the manifest's sha256.
curl -fL -o voice-en-us-lessac-medium.tar.gz \
  https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-us-lessac-medium.tar.gz

echo "d45ca50084c0558eb9581cd7d26938043bc8853513da47c63b94d95a2367a5c9  voice-en-us-lessac-medium.tar.gz" \
  | sha256sum --check --strict
# voice-en-us-lessac-medium.tar.gz: OK

# 4. Unpack ("unpack": true) and hand the result to the engine.
tar -xzf voice-en-us-lessac-medium.tar.gz
echo 'Hello from the registry.' | piper -m en-us-lessac-medium.onnx -f out.wav
```

`sha256sum --check --strict` exits non-zero on a mismatch, so under `set -e` a
swapped artifact stops the script instead of being fed to the engine.

### Driving it from the manifest, for any entry

The literals above come out of the manifest, so the same flow generalizes:

```bash
MODEL=openwakeword.hey-jarvis
BASE=https://raw.githubusercontent.com/JarvisOSLinux/voice-models/main

curl -fsSL "$BASE/models/$MODEL/model.json" -o model.json

# Every artifact, verified, into ./models/<id>/
mkdir -p "models/$MODEL" && cd "models/$MODEL"
jq -r '.source.items[] | "\(.sha256)  \(.url)"' ../../model.json |
while read -r sha url; do
  curl -fL -o "${url##*/}" "$url"
  echo "$sha  ${url##*/}" | sha256sum --check --strict
done
```

Then read `.audio` for the input contract (`sampleRateHz`, `frameSamples`) and
`.notes` for anything the engine needs to be told — for this entry, that all three
ONNX files must be passed explicitly or openWakeWord fetches its own preprocessor
at runtime.

### What a consumer should do with the metadata

| Field | Use it to |
|-------|-----------|
| `kind` + `languages` | Filter the catalog down to candidates. |
| `platforms` | Skip entries this registry never exercised on the host OS. |
| `requirements` | Rule out models that will not fit the device. |
| `license` | Decide whether the deployment may legally use it — surface `notes` to the user. |
| `trustStatus` | Refuse `removed`, warn on `deprecated`, treat `community` as unreviewed. |
| `audio` | Configure the capture/playback pipeline without guessing a sample rate. |

A consumer should **cache `registry.json`** and re-fetch on a schedule rather than
per query, and should treat a manifest whose hash does not match the index as a
hard failure, not a warning.

---

## Trust model

Same tiers and the same promotion rule as the sibling
[`mcp-registry`](https://github.com/JarvisOSLinux/mcp-registry), for the same
reason: a submitter must not be able to grant themselves the label that means "a
maintainer checked this."

| Tier | Meaning | How it is set |
|------|---------|---------------|
| `community` | Validated, not maintainer-vetted. | Default for every new entry. |
| `official` | A maintainer verified the entry end to end. | Requires the `trust-approved` PR label. |
| `deprecated` | Superseded; still fetchable. | Ordinary PR. |
| `removed` | Revoked; consumers should refuse it. | Ordinary PR. |

The gate is mechanical. `validate_registry.py --base <base registry.json>`
compares `trustStatus` per entry against the base branch, and any entry raised to
`official` — including one added directly as `official` — fails the run unless
`--approval-label-present` is passed. CI passes that flag only when the PR carries
the `trust-approved` label, and re-runs on `labeled`/`unlabeled` so removing the
label re-blocks the merge.

Two changes are annotated for review without failing:

- **a changed `license.spdx`** — a relicense reaches everyone who already deployed
  the model;
- **a widened `platforms` list** — that field records what this registry actually
  exercised, so adding a platform is a claim about work someone did, not a
  metadata tidy-up.

**Revocation is two stages.** First set `trustStatus: "removed"`: consumers refuse
new installs, existing ones keep working, and the entry and its provenance stay
intact. Only later delete the entry *and* its `models/<id>/` directory. The
orphan-directory check blocks a half-done removal in either direction.

---

## Validation and tooling

```bash
python3 scripts/sync_registry.py           # mirror manifests into the index, recompute hashes
python3 scripts/sync_registry.py --check   # exit 1 if anything would change (CI gate)
python3 scripts/validate_registry.py       # the PR gate
python3 scripts/selftest_registry.py       # prove the gate still fires
```

Stdlib only, no dependencies, no network.

`validate_registry.py` checks identity agreement (map key / entry id / manifest
id / directory name / self-declared URL), every enum, the mirror against each
manifest, the integrity hash, the shape of `requirements` / `license` / `source` /
`audio`, and orphan directories. Malformed input is always a validation error and
never a traceback: the gate runs on submissions from strangers, so every value is
type-checked before it is used, and a nested list where a string belongs is
reported rather than aborting the run.

`selftest_registry.py` is the reason to trust the above. This repo has no test
suite, and a validator that has quietly stopped firing looks exactly like a
registry with nothing wrong — so the self-test builds throwaway registries in a
temp directory, breaks them twenty different ways, and asserts the real entry
points report exactly what they should, including the cases that must stay silent.
CI runs it **before** the validator, so a broken gate fails the PR instead of
waving it through.

---

## Hosting and forking

The registry is consumed as two raw URLs:

```
https://raw.githubusercontent.com/JarvisOSLinux/voice-models/main/registry.json
https://raw.githubusercontent.com/JarvisOSLinux/voice-models/main/models/<id>/model.json
```

There is no server. A fork is a valid registry: change the `manifest` URLs in the
manifests and the index to point at your fork's raw URLs, re-run
`sync_registry.py`, and point your consumer at your `registry.json`. The
self-declared `manifest` field is what makes that rewrite checkable — the
validator fails if a manifest claims a URL the index does not fetch it from, so a
half-rewritten fork is caught rather than silently serving another registry's
documents.

Consumers that support multiple sources should treat an entry's id as unique
per-source, not globally: two registries may both carry
`piper.en-us-lessac-medium` with different hashes, and the one you fetched from is
the one whose review you are relying on.
