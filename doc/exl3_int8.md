# EXL3-INT8 仕様 (固定グリッド)

shelf Issue #10「EXL3-INT8の実装」の仕様書。この文書を正本とし、shelf はこれに合わせる。
実装は `feat/exl3-int8-fixed-grid` (rocm_exl3)。

## 目的

EXL3 は dequant すると既定で FP16 に展開されるため、W8A8 の INT8 GEMM を使えない。
EXL3-INT8 は、展開先を最初から INT8 にした実行形態である。

**artifact は変更しない。** 既存の mul1 EXL3 を再量子化せず、固定の INT8 グリッドで直接展開する。
`s0` の保存も、モデル側の目印も要らない。

## 仕様

### 1. 対象

- codebook が mul1 の EXL3 tensor。cb0 (legacy) と mcg は対象外 (値の分布が違い、この式が使えない)。
- 対象外の tensor、行数が少ない呼び出し、N が小さい線形は FP16 のまま実行する。

### 2. 重みの INT8 展開

mul1 の復号値は、byte 和 `B` (0〜1020) から `(887·B − 452608) · 2^-17` で決まる。
標準偏差はほぼ 1 で、最大は約 ±3.45 である。

```text
S_FIX = 4 · 887 · 2^-17 = 0.027069091796875   (fp32 で厳密に表せる)
q     = (B − 509) >> 2                         (算術右シフト。範囲は −128..127)
重み  = q · S_FIX
```

- `q` は復号値を `S_FIX` で四捨五入した値と、全ての `B` (0〜1020) で一致する。
  fp16 に丸めた復号値を丸めても同じ。tie までの余裕は 0.067 (q 単位)。
- 範囲がちょうど int8 に収まるので clamp は不要。
- 再現: `tests/test_exl3_int8_fixed_grid.py`。実モデルの復号器の出力とも全要素一致を確認した。
- グリッドは全 tensor で共通。tensor ごと・チャネルごとの scale は持たない。

### 3. 活性の INT8 化

1. 入力に Hadamard (`suh`) をかける (通常の EXL3 と同じ回転基底)。
2. token ごとに対称量子化する。`sa = amax(行) / 127`、`a8 = round(x / sa)` を [−128, 127] に clamp。

### 4. GEMM と epilogue

1. `int8 × int8 → int32` を厳密に累積する。
2. `y = c · sa · S_FIX` を **fp32** で計算する。
3. 出力側 Hadamard は **元の `svh`** を使う。scale を fp16 の `svh` に畳み込まない。
4. bias を足す。

手順 2〜3 は変更不可。scale を fp16 の `svh` に畳むと、丸め誤差が 48 層で複利になり、
macro KLD が +0.058 悪化した (`doc/w8a8_int8_prefill.md` の「svh8 事件」)。

## 検証結果

Qwen3-30B-A3B、通常の 3.05bpw EXL3 ((i) モデル)、R9700、21 ケース、BF16 参照に対する macro KLD。
fp16 コントロールは 0.05212 (V620 の以前の測定 0.0524 と一致)。

| 構成 | macro KLD | Δ | 備考 |
|---|---|---|---|
| attention の重みのみ | 0.05272 | +0.00060 | |
| attention の活性のみ | 0.05201 | −0.00011 | |
| attention の W8A8 | 0.05351 | +0.00138 | |
| lm_head の W8A8 | 0.05272 | +0.00059 | |
| attention + lm_head の W8A8 | 0.05417 | +0.00205 | 上の 2 つの和とほぼ一致 |
| expert の重みのみ (エミュレーション) | 0.05374 | +0.00162 | 丸めなしの経路確認は +0.00041 |

- expert の固定グリッド分は +0.0012。ケース別の標準誤差は 0.0011 で、統計的には有意でない。
- 全体の見積もりは約 +0.003 (KLD の約 6%)。全部を同時に INT8 にした実測はない。
- 想定した重みの丸め誤差だけの影響 (約 +0.0002) より大きい。理由は未解明。
- 判断: 現状の測定ではノイズと区別できない水準であり、許容として仕様を確定した。

## Qwen3.8 27B の GPU と KV キャッシュの確認

`qwen38-27b-exl3-4bpw` (mul1、dense + GDN、64 層) を `Generator` (paged cache) で貪欲生成し、
INT8 (`EXL3_INT8_PREFILL=1 EXL3_INT8_FIXED_GRID=1 EXL3_INT8_MIN_ROWS=256`) と通常を比べた。
プロンプト 2048 トークン、生成 96 トークン、`eval/exl3_int8_check.py --gen`。
INT8 経路は全層 (attention、GDN の in_proj、out_proj、MLP、lm_head) で engage した。

| GPU | KV キャッシュ | 結果 |
|---|---|---|
| R9700 (gfx1201) | fp16 | 完走。通常と先頭 380/409 文字一致 |
| R9700 (gfx1201) | 量子化 K5/V4 | 完走。通常と先頭 280/386 文字一致 |
| V620 (gfx1030) | fp16 | 完走。分岐は 149/386 文字 |
| V620 (gfx1030) | 量子化 K5/V4 | 完走。分岐は 255/399 文字 |

どれも生成は自然な文で、分岐は貪欲生成が数値差で別の語を選んだもの。ロジットの KL は 27B では未測定。
KV の量子化は INT8 と無関係で、通常の EXL3 でも同じ KV 設定が動く。

この確認の過程で、量子化 KV キャッシュが `AttnArgs.sanity_check` の `if`/`elif` の誤りで
27B に限らず常に落ちる不具合 (i8f8 キャッシュ追加時に混入) を見つけて直した。

## 使い方 (rocm_exl3、評価用)

| 環境変数 | 意味 |
|---|---|
| `EXL3_INT8_PREFILL=1` | INT8 経路を有効にする |
| `EXL3_INT8_FIXED_GRID=1` | `s0` の無い mul1 モデルに固定グリッドを与える |
| `EXL3_INT8_MIN_ROWS=0` | 行数の下限を外す (KLD 評価用。既定は 1024) |
| `EXL3_INT8_SCOPE=head\|nohead` | lm_head のみ / lm_head 以外のみ |
| `EXL3_INT8_ABLATE=w\|a` | 重みのみ / 活性のみ |
| `EXL3_INT8_EMU_W_EXPERTS=1` | expert を重みのみ丸めのエミュレーションで実行 (速度は出ない) |
| `EXL3_INT8_EMU_NOROUND=1` | 上のエミュレーションで丸めを外す (経路の確認用) |

評価スクリプトは `eval/kl_q30b/` (`compare_kld.py` は GPU 版)。

## 未検証・未実装

- **expert の INT8 実行は未実装**。既存の MoE INT8 経路は、全 expert の `suh` が一致するモデル
  (`--share-input-su`) を前提にし、通常のモデルでは使えない。expert の活性 INT8 化の影響も未測定。
- 全部を同時に INT8 にした KLD の実測、別モデル (Flash Next など) での確認は無い。
- ノイズフロアは未測定 (21 ケースのみ)。
- 速度は未評価。復号は FP16 を経由して丸めており、q を直接出すカーネルは無い。
  `q = (B − 509) >> 2` は整数演算だけなので、shelf ではカーネル内で直接生成できる。
- cb0 と mcg は未対応。

## shelf への引き継ぎ

- 上の「仕様」の 1〜4 をそのまま実装する。artifact の形式変更は要らない。
- shelf には EXL3 (mul1) の loader がまだ無い。homework の「exl3(mul1)の実装」が前提になる。
