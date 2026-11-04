# LSA速度・長文比較の測定条件

対象はこのEXL3移植版のQwen3.8-27B全体であり、attention一層だけの
カーネル倍率ではない。通常attention、D-LSA Flat、D-LSA Treeを比較する。
このモデルは64層のうち16層がfull attention、48層がGated DeltaNetで、
LSAの対象はfull attention層である。MLP等も含むモデル全体の所要時間を測る。

## 固定する条件

- モデルは `qwen38-27b-exl3-305-int8grid`。EXL3 3.05 bpw、head 6 bit。
- KVは全方式でK5/V4、batch 1、prefill chunk 4096。RoPEはモデル標準のまま。
- Flatは公開mapと既定値（top-k 4096、top-p 1024、window 512、sink 4）。
- Treeは同じmapと予算に、top-level 16、node size 4、node block 4096、
  beam 2048、start nodes 32768、FP8 node centroidを追加する。
- R9700×1は8K/32K/64K/128K、および262,080入力＋64生成を測る。
  最後のケースもモデル標準の262,144 token以内に収まる。
- V620×1は8K/32K/64K。8〜128Kの比較ではキャッシュ容量131,584 token、
  約256Kの追加比較では262,656 tokenを全方式で共通に確保する。
- GPU power modeは既存の`auto`を維持する。各実行前後の温度・電力・
  クロック等のsysfsスナップショットと、使用環境をJSONに保存する。
- Threadripper 3995WXの同一ホストでGPU別のプロセスを並行実行した。
  各プロセスは`OMP_NUM_THREADS=16`で、CPU affinityは固定していない。

各方式・各長さでウォームアップを1回行い、続く2回を本測定とする。
初期コンパイルやgraph captureの時間を本測定に混ぜない。各リクエスト前に
ページテーブルとrecurrent prefix stashをクリアし、実際の`cached_tokens=0`を
検査する。方式間で入力token列のSHA256が一致することも検査する。

通常attentionの既存最適化は有効なままで、LSAにも通常の選択時と同じ
実行経路を使う。従って、graph等も含めた「現在ユーザーが利用する実装」
としての差であり、同一カーネル上の選択アルゴリズムだけを比較する試験ではない。

## 速度・メモリ

生成はstop tokenを64 tokenまで抑制し、全実行で64 tokenに固定する。
番号回答後の継続文は速度測定用であり、その文章品質は評価しない。

prefill速度は`入力token数 / Job.time_prefill`、生成速度は`Generator.iterate`後に
`Job.new_tokens`の増加を最初と最後に観測した間の`63 token / 観測時間`を使う。
HTTPやクライアントまでの配信速度ではない。参考としてengineの生成時間と
速度、wall TTFTも残す。各tokenに独自のGPU同期は挿入せず、リクエストの
前後だけ同期する。メモリは確保済みモデル・キャッシュを含むPyTorchの
割当ピークで、予約ピークとpaged-cache自体の容量も別に記録する。

表示値は本測定2回の中央値。「ばらつき」はその2入力の最大値と最小値の差を
中央値で割ったもので、時間変動だけでなく入力による差も含む。
小さい差を統計的に有意な高速化と断定しない。
約256Kの追加比較ではtoken生成を観測するたびに時刻も保存し、区間の中央値・
95パーセンタイル・最大値と、最初の8 tokenを除いた速度も算出する。
この記録はCPUの時刻取得とリスト追加だけで、GPU同期は増やさない。
64 tokenの平均速度にはクラスタや階層の更新が含まれるため、短い生成窓だけを
定常的な1 tokenの速度と解釈しない。

## 長文での情報取得・logits

固定seedで多数の異なるレコードと6桁のdistractor codeを生成する。
求めるレコード`TARGET-ALPHA`とその番号を、文書の50%（ウォームアップ）、
10%・90%（本測定）に挿入し、文書末尾でその番号を質問する。
精度集計はこの3配置を対象に、出力の最初の6桁番号が正しいかで判定する。
これは限定したretrieval試験であり、RULER全体や日本語・コード理解の
網羅的な長文品質評価ではない。

最初の回答位置のlogitsをGPU上で1回だけcloneし、測定終了後にCPUへ転送する。
同じ入力の通常attentionをteacherとして、actual vocabulary全体でforward KLと
top-1一致を計算する。パディング語彙は除く。1位置の差であり、文書全体の
perplexityや全生成位置のKLではない。KLの単位はnats。
この1回のcloneは全方式の計測に含まれる。

## 再現

通常のROCm環境で実行する。両GPUを同じプロセスで比較せず、各GPUに
割り当てた実行環境で1つずつ起動する。

```bash
python rocm_tools/lsa_benchmark.py \
  --model /path/to/qwen38-27b-exl3-305-int8grid \
  --projections /path/to/qwen38_27b.pt \
  --lengths 8192 32768 65536 131072 --cache-size 131584 \
  --repeats 2 --new-tokens 64 --output /path/to/results/r9700.json
```

V620では`--lengths 8192 32768 65536`を使う。約256Kの追加測定は
`--lengths 262080 --cache-size 262656`を指定する。各リクエスト後にJSONを
atomicに保存し、完走したレポートだけを集計する。

```bash
python rocm_tools/lsa_benchmark_report.py \
  /path/to/results/r9700.json /path/to/results/r9700-256k.json \
  /path/to/results/v620.json --output-dir /path/to/summary
```

集計結果はCSV、JSON、比較ページとして出力する。KLのteacher logitsの
`.pt`は測定先に保持し、Gitには集計JSONと元の測定JSONを保存する。

## Treeキー不具合の修正と再測定

初回の128K Tree測定で、親の管理キーが子孫のキー全体を繰り返し含み、
階層が増えるとハッシュ計算量が急増する不具合を発見した。4,096子×4,096子の
再現キーではhash 1回が約0.6秒だった。ノードを固定長のslot-incarnation IDで
識別する修正後は同型のキーで約44μ秒となり、ID再利用の衝突回帰テストと
公式Tree参照との一致テストを追加・実行した。group signatureの参照も
子ごとに繰り返さずgroupごとに一度だけにし、参照回数を検査する回帰テストを加えた。

影響を受けたTree測定は中断し、修正後に全長さをウォームアップから測り直す。
通常attentionとFlatはこのTreeキー処理を使わず、ソースも変えていないため、
完走済みのペアを保持する。`--resume --modes tree`はモデル、map、GPU、KV、
容量、chunk、出力長、試行数が同じであることを確認し、Treeだけ置き換える。
各方式のソースcommitとハーネスSHA256、再開履歴をJSONに保存する。

## 大きなソート作業領域の修正

約256KのFlat初回では、query行をまとめたstable argsortの大きな一時領域の
割り当てに失敗した。新しいプロセスでも同じ箇所で再現し、エラー時には
13 GiB程度の空きVRAMとmemory fraction 1.0が報告されていた。
実メモリ不足の閾値とは区別し、失敗元の例外スタックを保存した。

Flatでは独立したquery行を16M score要素以下のbatchへ自動分割する。
Treeでも候補IDのソートを既存の有界batch内で実行する。
historyの切捨て、key budgetの削減、コンテキスト長の使用制限は追加しない。
スコア式、causal visibility、同値時のID順を維持することをCPU/GPUテストで
確認し、両GPUで29テストを通した。修正後のLSA測定を全長さで取り直し、
通常attentionのデータは処理経路が変わらないため保持する。

## dense prefixのKV展開範囲の修正

約256KのTreeでは、まだhistoryが空の最初のchunkでもキャッシュ全容量を
K/V展開する経路が残っており、大きな一時領域の割り当てに失敗した。
LSAでも通常のQSA経路と同様に実際のhistory長を渡し、未使用のキャッシュまで
展開しないように修正した。保持するhistoryやattentionの対象は変わらない。
ページを並べ替えた量子化キャッシュを使うprefill回帰テストを追加した。

最終runtime修正は`84dfb5b`。両GPUで関連30テストを通し、Flat/Treeを
全長さでウォームアップから再測定した。通常attentionは変更を通らないため
完走データを保持する。元JSONの方式別ソース情報と`runtime.json`、テストログを
結果ディレクトリに保存する。

## 実測結果

[2026-11-04のV620/R9700比較](../benchmarks/2026-11-04/lsa-performance/README.md)
