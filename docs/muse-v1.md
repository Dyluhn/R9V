# Muse Glimmer 30B R9V V1

This page covers the experimental single-R9700 Muse profile. Its exact model
package and benchmark identities are frozen and public, but it is not currently
a runnable user release because the curated native runtime source has not been
published.

## Inspect the profile from a clean clone

The supported public workflow includes catalog inspection and model download:

```bash
git clone --recursive https://github.com/Dyluhn/R9V.git
cd R9V

./r9v show muse
./r9v validate muse
./r9v fetch muse --model-dir /path/to/MODEL_DIR
./r9v verify muse --model-dir /path/to/MODEL_DIR -- --hash
```

The profile requires one Radeon AI PRO R9700 (`gfx1201`) with 32 GiB VRAM and
at least 64 GiB host RAM. The frozen model is a 24,554,611,392-byte GGUF with
SHA256
`f4870ff4ac316c1dbf50a55501f4c00e16070336fc40e119ff1167e43382856a`.

## Fail-closed stages

- `./r9v build muse` fails until the minimal native engine source is curated,
  license-audited, and published.
- `./r9v run muse --model-dir /path/to/MODEL_DIR` is not a supported user path.
  The frozen executable and `gfx1201` code object are research proof artifacts,
  not a published chat runtime or OpenAI-compatible server.

Do not substitute similarly named binaries or code objects. The accepted proof
artifact hashes and current limitations are recorded in the
[qualification report](../profiles/muse-glimmer-30b/v1-r9700/QUALIFICATION.md).
The speed comparison and its non-identical benchmark protocols are recorded in
the [benchmark report](../profiles/muse-glimmer-30b/v1-r9700/BENCHMARKS.md).

## Gates before a user release

The 10-file package, including license, attribution, projector, and DFlash
sidecar, was remotely size- and SHA256-verified at revision
`093f8ced7a8e2308b0f597084ebdbfa5f6614f75`. The runtime must still be rebuilt
from published source, enforce the recorded model/binary/code-object hashes,
and provide a tokenizer-aware interface. The profile remains `experimental`
until that complete clean-clone path passes.
