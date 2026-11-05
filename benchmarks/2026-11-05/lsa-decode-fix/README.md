# LSA decode修正の限定比較とV620×2の2M容量確認

この測定はTP共有index対応より前の版を対象とする。現在のTP=2＋MTP構成は
[1.75Miプロファイル](../../../doc/qwen38_27b_lsa_tp2_mtp_175mi.md)を参照。

Treeのdecodeを公式既定のFlat検索へ合わせ、FP16および通常のbit量子化KVではQ/K/V投影から出力投影までをHIP graphで実行するよう修正した。クラスタ更新・ページ検査はgraph外で行う。GPUを分割した際のTriton device guardも修正した。

実モデル8Kの生成では、階層探索eager比で1.30倍、Flat検索のgraph無効比で1.11倍を観測した。V620×2・K4/V4のレイヤー分割では2,097,664 token分のKV/indexと最大長decode用作業領域を確保し、短入力の生成まで成功した。2M全文のprefill・品質は測定していない。

## 速度

V620×1、Qwen3.8-27B EXL3 3.05 bpw/head 6 bit、K5/V4、power mode auto、OMP 16。全条件でTreeのprefill設定を維持し、decode経路だけを変更した。`descend_eager`は従来の階層探索を現コード上で再現した比較であり、バケット再利用の改善は全条件に共通する。

|decode経路|8K実モデル token/s|合成64K履歴のattentionブロック ms|
|---|---:|---:|
|Tree探索・graph無効|13.54|13.467|
|Flat検索・graph無効|15.84|1.437|
|Flat検索・HIP graph有効|17.66|1.000|

実モデルは同じ8K入力をprefix再利用し、各経路1回warmup＋1回本測定、64 token固定生成。prefixヒットは全方式7,936 tokenで一致する。生成速度は最初と最後に観測したtokenの間の63 token/秒で、旧レポートのcold-prefix測定とは直接比較しない。番号取得はwarmupを含む全6リクエストで成功した。graph再実行は2,048回を記録した。

64Kの数値は、実モデルの重みを使うattention一層に合成KV/index履歴を与え、同じ位置・同じ入力で32回×3組実行した中央値。モデル全体の64K性能ではない。graph有効とFlat eagerの出力一致を検査した（atol/rtol 3e-3）。全リクエストの品質や統計的な一般性能を示す試験ではない。

## 2Mの容量

キャッシュ容量は2,097,664 token（2,097,152入力＋512出力相当）。Flat LSA、V620×2、レイヤー分割で各GPUにfull-attention cacheを8層ずつ配置した。

- K4/V4のKV/index合計は45.26 GiB。別途、最大長の論理バケット表＋graph固定表として各GPUに1.25 GiBを保持した。
- その状態で8K入力・16 token生成を実行し、番号取得に成功。両GPU合計でgraphを256回再実行した。
- 終了時のPyTorch割当はGPU 0が29.14 GiB、GPU 1が29.21 GiB。予約量は29.81/29.93 GiB、CUDAが報告する空きは1.26/1.17 GiBだった。
- 成功時は既存のallocator設定で`expandable_segments`を無効にした。この環境の既定の有効設定では、約30 GiB空いているGPUでも約1 GiBの割当が失敗した。K5/V4の既定allocatorでの確保も失敗したため、K5/V4が使えるとは確認していない。

これはメモリ確保と短入力の動作確認であり、2M全文の処理成功・速度・精度の確認ではない。標準RoPEは262,144 tokenで、[公式mapの設定](https://github.com/Prefrontal-AI/latent-sparse-attention/blob/main/src/lsa/assets/models/qwen38_27b.yaml)はYaRN factor 40に拡張されている。容量確認では標準RoPEのまま短入力を使い、モデル設定は書き換えていない。

容量の結果はレイヤー分割の場合。TPではLSA indexがrankごとに複製されるため、同じメモリ量とはならない。また、今回graph化したのは通常のレイヤー分割経路で、TP collectives・LoRA・特殊KV形式はFlat検索のeager fallbackを維持する。

## 再現

```bash
python rocm_tools/lsa_decode_benchmark.py \
  --model /path/to/qwen38-27b-exl3-305-int8grid \
  --projections /path/to/qwen38_27b.pt --output /path/to/decode.json

# V620だけを2枚とも可視にした環境で実行
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
python rocm_tools/lsa_capacity_probe.py \
  --model /path/to/qwen38-27b-exl3-305-int8grid \
  --projections /path/to/qwen38_27b.pt --kv 4 4 \
  --cache-size 2097664 --output /path/to/capacity.json
```

関連テストはV620×2で32件成功、R9700で31件成功・1件skip（2枚目のGPU用ケース）。ページの変更・バケット境界・batch 2・履歴上書き・current deviceと異なるGPUでのgraphを確認した。

[速度の元JSON](v620.json)、[容量成功の元JSON](lsa-capacity-2m-k4v4-nativealloc.json)、[通常allocatorのK4失敗](lsa-capacity-2m-k4v4.json)、[K5失敗](lsa-capacity-2m-k5v4.json)、[環境とソース](runtime.json)、[V620テスト](lsa-graph-tests-v620.txt)、[R9700テスト](lsa-graph-tests-r9700.txt)。
