# voice-models

**A catalog of local speech models — STT, TTS, and wake-word — that any project
can consume.**

One JSON index answers the five questions a project actually has about local
speech: *what models exist, what can they do, what do they cost, where do they
legally come from, and how do I fetch and verify one.*

Nothing here is specific to one consumer. There are no project-specific fields in
the schema, no required runtime, and no assumptions about how you configure
anything. A consumer can be a Python daemon, a Rust CLI, or a shell script with
`curl` and `sha256sum`.

---

## Why this exists

Large language models converged on two access mechanisms — run a local runtime,
or speak an OpenAI-shaped API — so one abstraction covers the field. **Local
speech never converged.** Every engine packages its models differently and every
model is discovered a different way:

| Engine | What "the model" is | Where it comes from |
|--------|---------------------|---------------------|
| Vosk | A zip that unpacks to a model directory | A vendor download page |
| faster-whisper | CTranslate2 weights | Hugging Face, fetched by the engine |
| Piper | An ONNX voice + its `.onnx.json` config | Release assets, historically across several repos |
| whisper.cpp | A single GGML/GGUF file | Release assets or a converted upload |
| openWakeWord | An ONNX classifier + two shared preprocessor models | Release assets, classifier and preprocessor published separately |

So every project that wants to offer "pick a voice" writes five acquisition
strategies, hard-codes a handful of URLs, and still has nowhere to look up
whether a given voice may legally be used in a commercial product, or whether it
fits on a Raspberry Pi. This registry is that lookup — a single index, with
verifiable provenance, and a license field on every entry.

---

## Catalog

| ID | Kind | Engine | Languages | Disk | License | Platforms | Trust |
|----|------|--------|-----------|------|---------|-----------|-------|
| [`piper.en-us-lessac-medium`](models/piper.en-us-lessac-medium/model.json) | tts | piper | `en-us` | 61 MB | `LicenseRef-Blizzard2013-Lessac` ⚠️ | linux | community |
| [`openwakeword.hey-jarvis`](models/openwakeword.hey-jarvis/model.json) | wake | openwakeword | `en` | 4 MB | `CC-BY-NC-SA-4.0` ⚠️ | linux | community |

⚠️ **Read the `license.notes` before deploying either of these.** The Piper voice
carries the Blizzard 2013 Lessac dataset license, for which upstream declares no
SPDX identifier and states the voices are intended for personal use and
text-to-speech research. The openWakeWord model is CC-BY-NC-SA-4.0 —
non-commercial — even though the repository that ships it is Apache-2.0. That
distinction is exactly the kind of thing this registry exists to record.

`platforms` lists what the registry **actually exercised**, not what upstream
claims. Both entries were run end to end on Linux x86_64 — synthesizing audio and
scoring frames respectively — which is what earns them `["linux"]`. Nobody has run
them on macOS or Windows, so those do not appear.

More engines are wanted: Vosk, faster-whisper and whisper.cpp are supported by
the schema and validator today and are waiting on entries whose facts someone can
verify firsthand. See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Quick start

Fetch a model, verify it, hand it to the engine — with nothing but a shell:

```bash
set -euo pipefail

BASE=https://raw.githubusercontent.com/JarvisOSLinux/voice-models/main

# 1. Read the catalog.
curl -fsSL "$BASE/registry.json" -o registry.json

# 2. Fetch the artifact named by the entry.
curl -fL -O https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-us-lessac-medium.tar.gz

# 3. Verify it against the sha256 the registry recorded.
echo "d45ca50084c0558eb9581cd7d26938043bc8853513da47c63b94d95a2367a5c9  voice-en-us-lessac-medium.tar.gz" \
  | sha256sum --check --strict
# voice-en-us-lessac-medium.tar.gz: OK

# 4. Unpack and run.
tar -xzf voice-en-us-lessac-medium.tar.gz
echo 'Hello from the registry.' | piper -m en-us-lessac-medium.onnx -f out.wav
```

Step 3 is the point. Every artifact in the registry carries a required `sha256`
that the **consumer** verifies after download, so a swapped release asset or a
corrupted transfer is caught before the bytes reach an engine — offline, forever,
without the registry being reachable.

The full walkthrough, including verifying the manifest itself against the index
and driving the loop from `model.json` for any entry, is in
[`docs/REGISTRY-GUIDE.md`](docs/REGISTRY-GUIDE.md#consuming-the-registry-fetch-and-verify).

---

## Structure

```
registry.json                    The index: every model, in one request
models/<id>/model.json           Per-model manifest — the source of truth
scripts/
  sync_registry.py               Mirror manifests into the index + recompute hashes (--check for CI)
  validate_registry.py           PR gate: schema, enums, mirrors, hashes, orphans, trust promotion
  selftest_registry.py           Offline self-test proving the gates still fire
docs/
  model-reference.md             Full field reference
  REGISTRY-GUIDE.md              Format spec, integrity philosophy, consumer walkthrough
CONTRIBUTING.md                  How to submit a model
```

The manifest is the source of truth; the index repeats the fields a consumer
needs in order to *choose* a model, so browsing costs one request instead of one
per model. That mirror is written **only** by `sync_registry.py`, and
`validate_registry.py` fails any PR where the two disagree.

---

## Trust and verification

Every entry carries a `trustStatus`: `community` (validated, not maintainer-vetted
— the default), `official` (a maintainer verified it end to end), `deprecated`,
or `removed`. Promotion to `official` requires a maintainer's `trust-approved` PR
label; the validator fails a self-promotion.

What the registry vouches for is **identity and provenance** — this file, from
this project, under this license, exercised on these platforms. It does not
sandbox models, audit weights, or make claims about output quality. A verified
hash means you got the bytes the submitter got.

CI runs `selftest_registry.py` before the validator, so a gate that has quietly
stopped firing fails the PR instead of waving it through. Everything is stdlib
Python, offline: the registry never downloads a model, because a hash computed in
CI would be whatever the server served that day and would verify nothing. The
reasoning is spelled out in
[`docs/REGISTRY-GUIDE.md`](docs/REGISTRY-GUIDE.md#integrity-philosophy).

---

## Licensing

**AGPL-3.0 applies to this registry** — the scripts, the docs, the metadata in
this repository. It does **not** apply to the models it catalogs, and the registry
does not relicense anything: each entry carries its own `license` block naming
the terms upstream actually publishes, with the restrictions spelled out in
`license.notes`. Read that field before deploying a model. Several of the best
local speech models are non-commercial or research-only, and the license is
frequently not the one on the repository that hosts them.

---

## Related

- **[Project-JARVIS](https://github.com/JarvisOSLinux/Project-JARVIS)** — a
  reference consumer. It is one example of a project that needs this, not the
  audience: nothing in the schema knows it exists.
- **[mcp-registry](https://github.com/JarvisOSLinux/mcp-registry)** — the sibling
  catalog, for MCP servers rather than speech models. Same layout, same trust
  tiers, same promotion gate; this registry's tooling is adapted from it.
