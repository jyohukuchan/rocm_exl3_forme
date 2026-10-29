# r4d (vllm-mxfp4 / libr4d) 探索報告 — 27B prefill 高速化への適用

日付: 2026-10-30。測定 GPU: R9700 (gfx1201)。作業ブランチ: `feat/r4d-explore`。
対象: Qwen3.8-27B (full attention: 24 q / 4 kv / head_dim 256 = GQA 6, causal)、
GDN (16 k-head / 48 v-head / dim 128 / chunk 64)。目的: prefill 高速化
(特に長 context ~1M を見据えた coding agent 用)。

## 結論 (先に)

1. **r4d prefill attention は買い**。自前 fp16 Triton (retune 後 BM128/BN64/S1)
   に対して層あたり **1.5–2.8 倍** (現行 default 比では 2.3–4.0 倍)。
   f16 leg + bf16 KV が無精度リスクで 1.5–1.6x、fp8 KV + PV8 leg が
   小さな精度コストで 2.0x、両 leg で 2.6–2.8x。
2. **fp16-best retune (BM128/BN64/S1) を先に採用** — 1.5x、リスクゼロ、計測済み
   (Round-4)。r4d の効果はこの上に乗る。
3. **GDN は chunk_scan + kkt のアダプタ利用が次点** (FLA 3/5 カーネル置換、
   約 1/3 バイト、推定 2–2.5x)。ただし v0.5.0/base-pin の scan は
   decay span > 160 で不正確という既知 issue (#4) があり、exact 版の gate が必須。
   LAZY スナップショットは上流で revert 済み (多 turn 会話を破壊) のため不採用。
4. **ライセンス: 両 repo とも無し** (後述)。コード複写は不可。
   自前実装は文書化された技法からの clean-room のみ。

e2e prefill 投影 (現行 default-fp16 基準、内訳は課題文の実測シェアを使用。
8K: linear 51% / GDN 39% / attn 9%、128K: attn 69%、1M: attn 95%):

| 構成 | 8K | 128K | 1M |
|---|---|---|---|
| attn のみ (fp8KV+両leg, ~4x) | 1.07x | 2.07x | 3.48x |
| + GDN (~2.5x, 推定) | **1.45x** | **~2.4x** | ~3.5x (attn 支配) |

注意: GDN 値は推定 (未実測)。8K では linear が 51% で頭打ちになるため、
さらなる短文脈 prefill 向上には GEMM 側 (項目 5) が必要。

## 1. ライセンス確認結果

- 両 repo に LICENSE / COPYING / NOTICE ファイルなし。
  `grep -ri licen/copyright/SPDX` は libr4d で 5 件ヒットするが全て
  「limit」等の部分一致で、権利表示ではない。vllm-mxfp4 の .md には 0 件。
- GitHub license API は vllm-mxfp4 に対して 404 (認識できるライセンスなし)。
- Codeberg API の libr4d メタデータに license 欄なし。
- **結論: 無許諾 (all rights reserved 扱い)。rocm_exl3 (公開 repo) への
  コード複写・patch 流用は不可。** ローカルのビルド・計測・読解は問題なし。
  自前実装は公開文書の技法説明に基づく clean-room で行うこと。
  (参考: Quark MXFP4 チェックポイント自体は Apache-2.0 と README に記載があるが、
  今回のカーネル/patch 群とは別物)

## 2. 外部コンポーネント表

pin 実態の訂正: 課題文は「v0.5.0 が pin」とあるが、vllm-mxfp4 の `R4D_PIN` は
`b9e42ab` であり、rx パッチはその上に当てる運用である。
`rx10` は **v0.5.0 には clean apply 不可**
(build.sh の objs-loop 化、registry/module の drift のため 3 ファイルで失敗)、
**b9e42ab には clean apply 可** (全ファイル check 通過)。
本報告のビルド・計測は `b9e42ab + rx10` (本家と同一構成) である。
v0.5.0 = b9e42ab + dflash カーネル + dev-script 削除 (mxfp4-gemm は v0.5.0 後)。

| # | コンポーネント | 内容 | 上流の実測ゲイン | 制約 | EXL3 への適用性 |
|---|---|---|---|---|---|
| A1 | r4d prefill attention f16 legs (`attn_prefill_h256_gqa6_{bf16,f}8kv`) | 転置 S^T=K·Q^T、lane-private softmax、LAZY 参照 max rescale、MSKIP、FOLDQ、CONTIG ds_read_b128、prefetch+sched barrier、DOT2 rowsum、KWIDE、LDSB。bf16 Q、bf16/fp8 KV | serve で AITER unified 比 prefill attention kernel **1.65x**、e2e prefill +14.6% @64K (attn 34%)、+4.1% @16K。`R4D_ATTN=1` で 260K prefill +37.8% (対 AITER) | head_dim 256、GQA 6、paged block 16、causal、bf16 Q のみ。sliding window / sink / softcap / alibi / per-(token,head) scale 不可。fp16 との rel は 3.7e-3 (本報告実測) | **◎ そのまま呼べる**。27B 形状は完全一致。loader + KV アダプタのみ |
| A2 | fp8 QK/PV legs (`R4D_ATTN_FP8`=1 O_QK8 / 2 O_PV8 / 3 両方、rx10) | QK の fp8 WMMA 化 (LDS 半減→WGP 常駐 2 倍)、PV の P-fp8 化。fp8 KV 専用。O_QK8 は QK 2x rate、O_PV8 は PV 2x | kernel 96.2→150.5 TF (+56%、hot 8k chunk)、e2e prefill +10.8% @106K。ppl 8.3708→8.3707、GSM8K paired p=1.000 (end-task 無料の判定) | fp8 KV 必須。kernel-oracle 誤差は 1.7e-3→1.7e-2/2.6e-2/3.1e-2 (row relRMSE)。**本報告実測の max-rel: mode1 0.52 / mode2 0.045 / mode3 0.51** (fp8 KV 量子化ノイズ 0.033 を含む) | ○。PV8 (mode2) は誤差小で 2.0x。QK8 系は end-task gate (ppl/GSM8K or KLD/needle) が必須 |
| G1 | r4d GDN chunk_scan + kkt_solve (+conv_prep) | FLA の recompute_w_u + chunk_fwd_h + chunk_fwd_o を 1 kernel 化 (state を WMMA accumulator 常駐)。conv_prep 融合で conv 出力 34MB/層が HBM に出ない | 「同じ計算で約 1/3 バイト」。serve TP=1 行に GDN prefill 改善が含まれる (+20–24% の一部、他変更と混在) | k/v dim 128、chunk 64、conv width 4。fp32/bf16/f16 state 版あり。**v0.5.0/base-pin の scan は in-chunk decay span > 160 で不正確 (libr4d #4)**。exact 版 (radiance_gdn2、無 split) が opt-in で存在 | ○だが gate 付き。chunk_scan+kkt のみならアダプタ薄い。conv_prep 融合は vLLM glue 依存で工数大 |
| G2 | GDN decode 系 (recurrent/fused/lazy update) | prefill 対象外のため本報告は未評価 | decode step 系の行に記載 (省略) | **LAZY は revert 済み。多 turn 破壊のため不採用確定** | × (prefill 主眼)。decode は別途 |
| M1 | MXFP4/W4A8×FP8 A-tiled GEMM (`radiance_mxfp4_fp8.hip`, BM128/BN64/BK64, 8 wave) | fp8 A × e2m1 W (e4m3 LUT + E8M0 fold)。手書き fp8 WMMA (Triton は fp8 dot を出せず bf16 に upconvert するため)。fragment-order + NT decode | MoE lane prefill GEMM +25–28%、decode 不変。ParoQuant 帯で prefill +15–18% | MXFP4 (e2m1+E8M0, group 32) / W4A8 専用。EXL3 trellis (~3bpw) とは形式が別 | △。転用は「in-kernel weight decode→WMMA operand 直結」の設計思想のみ。新 kernel が要る (工数大)。短文脈 prefill (linear 51%) の将来札 |
| M2 | skinny bf16 GEMM (`gemm_bf16_nt_m64`, M≤64) | 小 M の split-K GEMM。MoE gate/b-a 等 | 2.4–5.8x (該当形状)。ただし drafter ULP 感応の課税あり (`_CFG_TAXED` 扱い) | (N,K) 毎の実測 config が要る | △。decode 向き。prefill 主眼では優先度低 |
| P1/P2 | `patch_moe_prefill_attn.py` / `patch_unified_attention_lds.py` | 2D prefill tile (BLOCK_M 128 / warps 8 / stage 1) + LDS fit clamp | prefill attention 10.8x (16k, 対 decode-shaped 設定)、LDS OOM 回避 | vLLM/AITER 専用 patch | △。思想のみ: 自前も BM128 再調整で 1.5x を既得 (Round-4)。LDS 64KiB clamp は自前実装時のチェック項目 |

## 3. R9700 実測 (本報告)

環境: 2nd container (`rocm-exl3-phase2-env:tested`、R9700 のみ可視、
torch 2.12+rocm7.2 / Triton 3.7)。r4d.so は scratch で
`b9e42ab + rx10` を `GFX_ARCH=gfx1201` でビルド (1.9MB、registry 検証済み)。
Hessian ジョブ (24.5GB 占有) と同居のため A/B 交互×3 round の中央値を報告
(spread も記載)。自前 Triton 側は `/src` (=本 worktree) の
`paged_attn_triton_prefill` を full-window fp16 PAGE=256 cache + append で呼ぶ
(production 同等)。r4d 側は ctypes 直呼び (R4DArgs + stream)。
Q/K/V は同一 fp32 マスタから生成 (r4d 用 bf16、Triton 用 fp16、参照は fp32)。
fp8 KV は per-tensor scale (vLLM production 式)。

注意: r4d の Q と出力は **bf16 固定** (fp16 バッファで受けるとゴミになる。
実測で踏んだ)。自前 fp16 パイプラインとの接続には Q の bf16 化と
出力の扱い (bf16 のまま次層へ or 戻し) が要る。コストは prefill 1 回分で微小。

### 3.1 attention 正確性 (Tk=2048, Tq=256。chunked fp32 手計算参照)

| 比較 | maxabs | rel (max/参照amax) |
|---|---|---|
| 自前 fp16-def vs 参照 | 3.05e-05 | 5.3e-04 |
| r4d bf16kv (f16 legs) vs 参照 | 2.14e-04 | 3.7e-03 |
| r4d fp8kv (f16 legs) vs 参照 | 1.90e-03 | 3.3e-02 (fp8 KV 量子化ノイズ) |
| r4d bf16kv vs 自前 fp16-def | 2.14e-04 | 3.7e-03 |
| r4d fp8kv mode1 (O_QK8) vs 参照 | 2.99e-02 | 5.2e-01 |
| r4d fp8kv mode2 (O_PV8) vs 参照 | 2.60e-03 | 4.5e-02 |
| r4d fp8kv mode3 (両方) vs 参照 | 2.96e-02 | 5.1e-01 |

mode0/2 は量子化ノイズの範囲。mode1/3 (QK8) は kernel 誤差が 15 倍大きく、
上流は end-task (ppl/GSM8K) で無料と判定しているが、EXL3 側でも gate が要る。
(副産物: ROCm torch SDPA の GQA+causal は参照として壊れている (rel 1.5) ため、
本測定の参照は手計算。Round-4 と同所見。)

### 3.2 attention 速度

層あたり ms。med [spread]、括弧は fp16-best 比。
fp16-def/best は自前 Triton。r4d-fp8kv の leg は R4D_ATTN_FP8 で切替。

f16 legs (mode0):

| Tk | Tq | r4d-fp8kv | r4d-bf16kv | fp16-def | fp16-best |
|---|---|---|---|---|---|
| 8192 | 2048 | 3.23 (0.55x) | 3.77 (0.65x) | 8.65 (1.49x) | 5.82 (1.00x) |
| 8192 | 8192 | 7.74 (0.59x) | 8.78 (0.67x) | 20.40 (1.56x) | 13.10 (1.00x) |
| 32768 | 2048 | 13.49 (0.56x) | 15.51 (0.64x) | 38.02 (1.57x) | 24.26 (1.00x) |
| 32768 | 8192 | 48.62 (0.56x) | 56.32 (0.64x) | 136.32 (1.56x) | 87.57 (1.00x) |
| 131072 | 2048 | 55.88 (0.57x) | 64.01 (0.65x) | 158.05 (1.60x) | 98.56 (1.00x) |
| 131072 | 8192 | 219.02 (0.54x) | 251.10 (0.62x) | 624.60 (1.53x) | 407.19 (1.00x) |

fp8 legs (r4d-fp8kv のみ。bf16kv/def/best は mode 不変のため省略):

| Tk | Tq | mode1 O_QK8 | mode2 O_PV8 | mode3 両方 |
|---|---|---|---|---|
| 8192 | 2048 | 3.01 (0.51x) | 2.91 (0.49x) | 2.17 (0.36x) |
| 8192 | 8192 | 7.17 (0.53x) | 6.88 (0.51x) | 5.22 (0.39x) |
| 32768 | 2048 | 12.72 (0.52x) | 12.14 (0.49x) | 8.99 (0.37x) |
| 32768 | 8192 | 45.66 (0.52x) | 43.58 (0.49x) | 32.32 (0.37x) |
| 131072 | 2048 | 51.93 (0.52x) | 50.21 (0.51x) | 37.01 (0.37x) |
| 131072 | 8192 | 201.86 (0.50x) | 194.25 (0.48x) | 144.13 (0.35x) |

読み: fp16-best 比で mode0-fp8kv **1.7–1.85x**、bf16kv **1.5–1.6x**、
mode1 **1.9–2.0x**、mode2 **2.0–2.1x**、mode3 **2.6–2.8x**。
現行 default 比ではさらに ~1.55x 掛かる (mode3 ≈ 4.0–4.2x)。
自前 fp16 値は doc の既報 (5.9/13.3/24.5/87.7/98.7/404) と一致し、
測定系は較正されている。spread は 0.05–0.7ms (競合下でも安定)。

FLOP 参考 (因果、層あたり ≈ 4·Tq·Tk·D·Hq。128K/8K = 25.8 PFLOP):
fp16-best 63 TFLOPS、r4d mode0-fp8kv 118 TF、mode3 179 TF。
上流の 96→150 TF (hot 8k chunk) とは形状が違うが、比の傾向は一致する。

### 3.3 GDN (自前 FLA chunk が prize の定量化。r4d 側は未実測)

`chunk_gated_delta_rule` (vendored fla) を B=1、q/k 16head / v 48head /
dim 128、prefill 条件で計測 (層あたり):

| T | fla-chunk-gdn |
|---|---|
| 2048 | 1.32ms |
| 8192 | 5.40ms |
| 32768 | 29.44ms |

131072 は Hessian 同居の VRAM (空き ~7GB に対し中間 3GB + 入力 2.7GB) で
OOM のため未測定。ほぼ線形に伸びる。r4d 側の直接計測は「conv_prep 融合が
vLLM glue 依存で重い」ため見送り、chunk_scan+kkt のアダプタ利用を
推定 2–2.5x (約 1/3 バイトの主張から。conv_prep 融合込みなら上振れ) とする。
**ただし issue #4 (span > 160 で不正確) の exact gate が前提。**

## 4. EXL3 への統合パス

### (a) r4d.so を loader 経由で呼ぶ (推奨。工数 S: 数日)

- ビルドは本報告の手順で再現可能 (pin b9e42ab + rx10、hipcc、pybind11 のみ。
  `./build.sh` の中身は読了済み。setup/docker は使っていない)。
- Python loader は `radiance_r4d_attn.py` の run-routing 部分だけを
  自前 cache 向けに書き直す (~200 行)。C ABI は安定 (R4DArgs + stream)。
- KV アダプタ: r4d は block 16・K/V pack (HND)・bf16/fp8 を要求し、
  自前は PAGE 256・fp16 (or i8f8)。当面は prefill 毎に window を変換
  (128K で ~1GB traffic、数 ms。kernel 400ms に対し無視可)。
  本命は書き込み時量子化 (fp8 K/V + scale。i8f8 の K-int8/V-fp8 と親和)。
- Q は bf16 化 (fp16→bf16、安価)。出力 bf16 の扱いを決める (次層が fp16 のため
  戻し変換 or bf16 伝播。戻しでも kernel 比で微小)。
- フォールバックは現行 Triton (形状外・非 causal 等は辞退)。
- 精度 gate: bf16kv+f16legs は rel 3.7e-3 で無審査可の水準。
  fp8kv (mode0/2) は KLD/needle (27B E2E。VRAM 空き待ち) を通す。
  mode1/3 (QK8) は上流同様 ppl + GSM8K 相当の end-task gate が必須。

### (b) 技法の自前 Triton/HIP への移植 (非推奨。工数 M: 数週間)

- LAZY rescale、転置 S^T、FOLDQ、fp8 legs の移植。
- Round-2/3/4 の実測が示す通り、Triton lowering は fp8 WMMA の 2x を
  出せず (1.3x、traffic 分のみ)、HIP 直書きは multi-wave + pipeline 化という
  本格 FA 最適化の書き直しが要る。到達点は r4d との parity が天井。
- 結論: 同じ物を二度書くのではなく (a) で呼ぶ。

### (c) GEMM: trellis→operand 直結デコード (研究枠。工数 L)

- r4d の W4A8/MXFP4×FP8 は「重み decode を WMMA operand 生成に融合」
  (e2m1→e4m3 LUT、E8M0 の accumulator 側適用、A-tiled、LDS staging)。
- EXL3 prefill GEMM は trellis 復号→fp16→hgemm。復号を WMMA fragment 生成に
  融合できれば 8K prefill の linear 51% に効く。ただし新 kernel で形式も別、
  ゲイン未知。attn/GDN の後に研究 spike で。
- 副次知見: `tl.dot_scaled` は gfx1201 で fp8 を出さない (upconvert+変換コスト)
  ため、fp8 を活かす GEMM は手書き WMMA が前提。

## 5. ranked 推奨 (採用順)

1. **fp16-best retune の採用** (BM128/BN64/S1): 1.5x、リスクゼロ、測定済み。
   dispatch 既定の変更のみ。r4d と独立に効く。
2. **r4d prefill attention loader (a)**: bf16kv + f16 legs から
   (1.5–1.6x vs best / 2.3–2.5x vs default、精度無審査級)。
   次に fp8kv + PV8 (mode2、2.0x / 3.1x、KLD gate 後)。
   mode3 (2.7x / 4.1x) は end-task gate 後に選択。
   27B prefill への寄与は §6 の表の通り (1M で ~3.5x が天井)。
3. **GDN chunk_scan + kkt アダプタ** (推定 2–2.5x、8K prefill +15–20%):
   exact 性 gate (span>160 の synthetic + 実文脈) を先に。conv_prep 融合は後段。
4. **GEMM 研究 spike** (8K の linear 51% 用): trellis 融合の試作と計測。
   優先度は 2・3 の後。
5. やらないこと: LAZY (revert 済み)、decode 帯への適用 (本件対象外)、
   Triton への技法移植 (b)、コード複写 (license)。

## 6. 残課題・制約メモ

- 27B E2E (KLD/needle) が VRAM 空き待ち (13.6GB vs 空き 9.7GB):
  Hessian 終了後に `bench_r4d_attn.py` の正確性節 + `dense_kld_dump.py` 方式で
  通すこと。fp8 legs の採用判断はこれ待ち。
- 1M での kernel 実測は不可 (KV だけで層あたり GB 級)。128K までの線形性と
  シェアからの投影に留める。
- r4d decode kernel (q_len·GQA ≤ 64) は未評価。prefill 主眼のため対象外。
- GDN exact gate: span>160 を踏む synthetic (gate 極値) + 長文脈実測で
  radiance_gdn2 相当の無-split 版が必要か判断する。
- 本測定は Hessian と同居 (≤2 プロセス遵守、VRAM 小)。spread 0.05–0.7ms。
- 2nd container (`rocm-exl3-r4d`) は本報告用に起動したまま。
  不要になれば `docker rm -f rocm-exl3-r4d` で削除可 (image 共有、無状態)。
  prebuilt ext のコピーは `/home/homelab1/coding-local/r4d-scratch/prebuilt/`。

## 7. 再現手順

```bash
# scratch 展開 (clone は触らない)
cd /home/homelab1/coding-local/external/libr4d
git archive b9e42ab | tar -x -C /home/homelab1/coding-local/r4d-scratch/base-pin
cd /home/homelab1/coding-local/r4d-scratch/base-pin
git apply /home/homelab1/coding-local/external/vllm-mxfp4/radiance_r4d_extras_rx10.patch
# container (R9700 のみ): image は rocm-exl3-phase2-env:tested と同一
#   --device=/dev/kfd --device=/dev/dri/renderD130, ROCR_VISIBLE_DEVICES=GPU-a8e9...,
#   PYTORCH_ROCM_ARCH=gfx1201, mount: worktree→/src, r4d-scratch→/scratch
pip install pybind11   # container 内
cd /scratch/base-pin && GFX_ARCH=gfx1201 ./build.sh   # → r4d.so
cd /scratch && R4D_ATTN_FP8={0,1,2,3} python3 bench_r4d_attn.py
cd /scratch && python3 bench_gdn_fla.py --t 2048 8192 32768
```

bench は `eval/prof/bench_r4d_attn.py`, `eval/prof/bench_gdn_fla.py`
(本ブランチに同梱。`R4D_SO` env で .so 位置を指定可)。
`import exllamav3` には事前ビルド ext が要る — w8a8 側の dual-arch .so
(89MB) のコピーで解決した (bench 内で `/scratch/prebuilt` を優先)。

