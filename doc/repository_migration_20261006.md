# Independent project migration, 2026-10-06

Active development now lives at
[jyohukuchan/rocm_exl3_forme](https://github.com/jyohukuchan/rocm_exl3_forme),
a public standalone repository (`fork: false`, default branch `main`).
The [old fork](https://github.com/jyohukuchan/rocm_exl3) remains available with
a migration notice and its existing history and upstream PR relationship.

## Preserved history and licensing

The two published source branches were copied without rewriting commits:

| Ref | Source revision at cutover |
|---|---|
| `main` | `1bc73608448da6d56b8b2e1846903f74097bffd8` |
| `codex/r9700-preflight-go` | `7f00a294f1bc3c2155aa87754a779ee8bfd50cad` |

The original source had no tags or release assets. Its main cutover commit is
an ancestor of the independent project's main branch. Published branch heads
were verified identical immediately after mirroring. Unpublished local branches
and existing worktrees remain local; their heads were preserved.

The original MIT license text and Turboderp copyright remain byte-for-byte
after removing the single added 2026 copyright line for jyohukuchan's own
contributions. Vendor licenses and inference implementation are unchanged.
README preserves the ExLlamaV3 and CarouselAether lineage and describes the
current project's independently maintained scope. Package metadata retains
the `exllamav3` package name/original author and adds the project maintainer and
new repository URLs.

GitHub push protection initially misidentified the 32-character architecture
identifier `Mistral3ForConditionalGeneration` as a Mistral API key in five old
commit locations. Every location was checked and contained that same public
class name. A specific `false_positive` classification was accepted, allowing
the original history to be pushed without rewriting it. No repository-wide
security setting was disabled.

## Local environment

The working checkout remains `/home/homelab1/coding-local/rocm_exl3` so Docker
mounts, user services, native libraries and existing worktrees keep their paths.
Remotes now are:

| Remote | Repository |
|---|---|
| `origin` | `jyohukuchan/rocm_exl3_forme` |
| `legacy-fork` | `jyohukuchan/rocm_exl3` |
| `upstream` | `CarouselAether/rocm_exl3` |

Main and the previously tracked published branch now track the corresponding
`origin` branches. Existing Qwen API health was verified after migration;
inference source, requirements and native build setup did not change.

## Published JEV model

[jyohukuchan/JEV-27B-VL-exl3-4bpw](https://huggingface.co/jyohukuchan/JEV-27B-VL-exl3-4bpw)
now links to the independent engine. Model metadata revision
`dde678d36be0a4fe188e8dbd969a7660460cbd5d` changes only five small files:
README, NOTICE, code copyright, engine provenance and the file manifest.
All 19 safetensors retain their original SHA256 values; model weights remain
Apache-2.0. All 44 release files and anonymous public metadata access were
reverified; see [HF verification](jev_hf_release/hf-verification.json).

The validated engine commit `7675fa26cdf52fbe5a90060a69c60c752855fd8f`
is reachable unchanged in the independent repository. The original HF model
revision and old GitHub commit links also remain available.
