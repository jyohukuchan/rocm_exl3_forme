# R9700 VLM projection and MoE dispatch

The gfx12 WMMA wrapper remains unimplemented. A cooperative EXL3 GEMM that
reaches it traps rather than producing an answer. The existing individual
`LinearEXL3` guard used reconstruct plus GEMM for multi-row gfx12 input, but
pairwise Q/G, K/V, gate/up bundles and C++ block paths could bypass that guard.
MoE's fused-buffer guard also left the BC single-expert prefill route enabled.

Projection dispatch now consults the loaded linears' device capability. On
gfx12, multi-row bundled projections and multi-row BC attention/GDN/MLP use
the individual guarded projections. Sliced bundles always need cooperative
GEMM, so they decline even at one row. Ordinary single-row GEMV/MGEMV remains
enabled. MoE prefill disables BC expert GEMM while retaining the existing
per-token ROCm MGEMV decode proxy. Modules on other devices retain their prior
dispatch. This is a functional dispatch fix, not a native gfx12 WMMA port.

## Verification on 2026-10-08

R9700 only (gfx1201, PCI 0000:07:00.0). Qwen3.5-35B-A3B EXL3 4.09bpw,
immutable revision39e6392b9fb84ad2216fd3362793cf80f7d409fc. No second full model
was resident. The isolated checks loaded only the selected matrices/layer.

| Check | Result |
| --- | --- |
| Raw K/V MGEMM, H2048/N512/K5/MCG, 16 rows | HSA exception, including forced shape1 |
| K/V ordinary one-row MGEMV | finite; bitwise equal to separate projections; FP32-matmul max error0.00022134 |
| Guarded attention dispatch, 16 rows | finite; equal to separate projections; FP32 max error0.00029032 |
| Routed MoE layer0, 1 row | FP32 max error7.94e-7; relative-L2 versus individual-expert path9.89e-8 |
| Routed MoE layer0, 16 rows | FP32 max error1.11e-6; relative-L2 versus individual-expert path3.98e-4 |
| Six fixed Minecraft PNGs, default dispatch | all requests completed without diagnostic unfusing flags |

The same six PNGs, prompt, temperature0, context65536, KV5/4 and pixel budget
524288 were used for the previous fully unfused reference. Mean response time
excluding the first request was7.89s there and4.13s with guarded dispatch.
Five different warmed requests are a screening measurement, not a repeated
throughput benchmark. The recognition score remained19/25 fields; faster
dispatch did not establish better game understanding or task completion.

CPU tests cover normal/SWA attention declining an unsupported multi-row
bundle, preserving single-row GEMV capability, and declining sliced bundles.
The related runtime/dispatch suite passed24 tests.

Reproduce the isolated checks with `rocm_tools.check_mgemm_projection`
(`--dispatch` tests module dispatch; omitting it deliberately forces the raw
kernel) and `rocm_tools.check_moe_dispatch`. Both compare finite output against
FP32 matmul over reconstructed effective quantized weights. These checks do
not prove every model, activation, shard, graph mode or input shape correct.
