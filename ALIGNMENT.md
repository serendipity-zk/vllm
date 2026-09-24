# ServingStudio instrumentation on upstream vLLM

`servingstudio-alignment` is the one maintained ServingStudio line of this fork.
It sits on official vLLM `upstream/main`; the current upstream base is main
`04730e82700d0c9b957cda188dd1bbc8073e1fcd` (2026-09-24). That is the newest
main commit with a published `cu130` precompiled wheel that contains
`vllm/models/deepseek_v41`, so DeepSeek-V4.1 can be served. ServingStudio Sim
pins the line at `alignment/profiler/vllm`.

The line was previously based on `8369affa5428ca29a30f378d660fbffbad12e240`
(v0.28.1rc0). The rebase onto `04730e8` kept one commit per instrumentation
commit, except `Bind AsyncLLM.profiler on every profiler backend`. That commit
is dropped because upstream `c3ec0d29f5` binds `AsyncLLM.profiler` itself.

It converges two earlier lines:

- `moesim-profile` (base `967c5c3`, v0.22.1rc0), the line Sim pinned for the
  GLM-5.2 captures. It stays on the remote because recorded captures name it.
- `glm53-dflash2-alignment` (this branch's start, `79838a5a18`), the v0.28
  port for GLM-5.3 DFlash2, kept as `backup/glm53-dflash2-alignment-20260923`.

## Scope

The eight source commits, in order, are `7c025011f`, `24b6c79a8`, `08fe031f0`,
`15092c51b`, `4492fff9a`, `91547cfa0`, `0b8edfb56`, and `556aaf730`.
They preserve request TTFT/TPOT, API timing, EngineCore cadence and speculative
progress, bulk token/routing dumps, and target/draft EPLB provenance.

Migration adaptations preserve upstream's `capture_iteration_details` and
scheduler prefill throttling. Token-in/token-out serving now lives in
`vllm/entrypoints/scale_out/token_in_token_out/`.

Model Runner V2 additionally emits indexed `vllm_iteration(N): <phase>` scopes
for preprocess, target forward, postprocess, sampling, draft, bookkeeping, and
EPLB. The draft scope covers the complete speculator proposal, including the
DFlash2 selector. Warmup/dummy calls do not acquire real iteration IDs. The
index travels with `ExecuteModelState` from target execution to sampling.
Bookkeeping may have multiple disjoint scopes in one iteration.

`VLLM_NVTX_SCOPES_FOR_PROFILING=1` enables the scopes. Enable
`--enable-logging-iteration-details` for schema-v4
`VibeSimAlignmentIteration` records. Existing
`VLLM_VIBESIM_TOKEN_TRACE_*` and `VLLM_VIBESIM_ROUTING_TRACE_*` switches apply
to both model runners. Dumps can synchronize/copy GPU data; collect them in
the separate evidence pass, not in timing captures. The token dump uses actual
query boundaries, including when adaptive verification trims scheduled tokens.

## Environment

Use this checkout's `.venv`; never copy another checkout's editable environment
or native extensions. The installation used:

```bash
uv venv --python 3.12 .venv
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_COMMIT=04730e82700d0c9b957cda188dd1bbc8073e1fcd \
VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 \
uv pip install --python .venv/bin/python -e .
uv pip install --python .venv/bin/python -r requirements/lint.txt \
  pytest pytest-asyncio nvtx ruff tblib
uv pip check --python .venv/bin/python
```

`VLLM_PRECOMPILED_WHEEL_COMMIT` must be the upstream base SHA,
`04730e82700d0c9b957cda188dd1bbc8073e1fcd`. If the upstream base changes,
update the wheel commit explicitly. Import and
GPU validation must resolve Python and native extensions under this worktree.
An import check alone does not qualify the driver or a serving run.

ServingStudio's profiler runs `alignment/profiler/vllm/.venv/bin/python` by
default. Run pre-commit checks explicitly in this checkout; do not replace
shared Git hooks with a hook that points at this environment.

The wheel saved in `../glm53-dflash2-artifacts/` is the `cu130` wheel for the
earlier `8369aff` base. It does not match `04730e8`, so do not use it with this
base. If that metadata endpoint is
temporarily unavailable, set `VLLM_PRECOMPILED_WHEEL_LOCATION` to the path in
`vllm-wheel-path.txt`. `native-wheel.json` records its SHA256 and provenance.

## Frozen models

| Role | Repository | Revision |
| --- | --- | --- |
| Target | `zai-org/GLM-5.3` | `aca966e4e02791568aa6a4ced368624b3d897f42` |
| Draft | `incoai/GLM-5.3-DFlash2` | `425aa615ce320caac34400208b30808c8f14f76c` |

The snapshots live under `/raid/hf/hub/models--<owner>--<model>/snapshots/<revision>`.
The target is the original roughly 756 GB checkpoint, not GLM-5.3-Flash or an
NVFP4 conversion. The roughly 4.92 GB BF16 draft has six layers, hidden size
6144, block size 8, selector rank 256, and selector top-k 16. It proposes seven
draft tokens per block.

Download logs and completion manifests live in
`../glm53-dflash2-artifacts/`. A model's manifest is written only after every
repository file exists and its size matches Hub metadata. The download script
can resume the fixed revisions using the shared cache.

For a future serving validation, select Model Runner V2 and pass the draft
snapshot with `--speculative-config`:

```json
{"method":"dflash","model":"/raid/hf/hub/models--incoai--GLM-5.3-DFlash2/snapshots/425aa615ce320caac34400208b30808c8f14f76c","num_speculative_tokens":7}
```

Use `VLLM_USE_V2_MODEL_RUNNER=1`. Select topology, memory budget, and attention
backend against the actual available GPUs. The model card demonstrates SGLang;
checkpoint compatibility in a full vLLM run still requires validation.

## Routed-experts capture

ServingStudio's `token_corpus` pass serves with `--enable-return-routed-experts`
and reads every generated token's routes, body layers and MTP layer. At
`04730e8`, upstream serves routes through the AuxOutput connector
(`vllm/distributed/aux_output_connector/`). The connector requires Model Runner
V2, a MoE generate model, and prefix caching. It rejects adaptive speculative
verification, PP > 1, DCP/PCP > 1, and KV connectors. Upstream also captures
inside monolithic TRT-LLM kernels and refuses kernels it cannot capture. This
branch adds:

- capture slots for an MTP drafter's layers (method `mtp` only). The
  single-module MTP speculator is bound when the V2 runner creates the
  connector;
- the draft prefill's routes only; later draft passes cover one row per request
  and are undone;
- the drafter's slots written into the connector's pending snapshot after
  propose. `AsyncOutput.copy_aux_output` then starts the step's deferred D2H
  copy.

V1 cannot return routes at this base. The routing dump
(`VLLM_VIBESIM_ROUTING_TRACE_*`) still works on both runners. On V1, and on V2
without the connector, the dump's env var binds a private capturer that returns
nothing on responses. The dump skips the drafter's slots.

`EPLBConfig.rearrange=false` keeps EPLB a load recorder for the routing passes.
It allocates no transfer buffer and never moves experts.

### Audit of `moesim-profile` commits

| `moesim-profile` commit | Here |
| --- | --- |
| 6 upstream bugfix cherry-picks (ROCm, Docker, CPU, FastAPI) | in upstream by v0.28 |
| 7 `feat(alignment)` instrumentation commits | ported by the v0.28 line |
| `9a0675d` speculative progress | ported as `ce0ad5378d` |
| `2c2f73f` target/draft expert-load roles | ported (`model_role`, `max_forwards_per_step`) |
| `829f25c`, `7fa4e7c`, `5552bbd`, `2cbd2d9`, `133dc06` prefer modular kernels while capturing | not needed: upstream captures inside monolithic kernels and refuses the rest by name |
| `4b0cab6` capture-layer sizing, `routed_experts_prompt_start` | sizing ported; `prompt_start` is upstream |
| `5316893` slots for the MTP drafter only | ported |
| `4652f20` keep the first draft pass's routes | ported to the V2 speculator |
| `b986f76` refuse the unpadded drafter | not applicable: V2 has no unpadded drafter path, and V1 returns no routes |
| `828da2e` EPLB `rearrange: false` | ported |

## Maintenance and validation

Fetch official upstream, record its SHA, and replay this branch's instrumentation
in an isolated worktree. Keep the wheel base and Python base identical. Preserve
upstream fixes when resolving conflicts; audit both runner implementations.

Focused tests include:

```bash
.venv/bin/python -m pytest -q \
  tests/v1/worker/test_alignment_trace.py \
  tests/v1/worker/test_gpu_model_runner_v2_eplb.py \
  tests/model_executor/test_routed_experts_capture.py \
  tests/distributed/test_eplb_utils.py \
  tests/v1/engine/test_iteration_logging.py \
  tests/entrypoints/openai/completion/test_alignment_api_timing.py \
  tests/entrypoints/scale_out/token_in_token_out/test_generate_stream.py
.venv/bin/python -m pytest -q tests/v1/engine/test_output_processor.py \
  -k alignment_request_timing
```

The final migration result and commands belong in the task artifacts. Do not
claim GPU/NSYS or full GLM-5.3 serving qualification from unit-test results.
