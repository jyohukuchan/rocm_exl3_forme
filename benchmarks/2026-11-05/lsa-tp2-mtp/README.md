# Qwen3.8-27B 4bpw・TP=2・MTP・1.75Mi

V620×2、batch 1、モデル本体TP=2、MTPはGPU 0、K4/V4を使用する。
4bpwモデルは公開EXL3を取得し、safetensorsのSHA256を配布元のLFS metadataと照合した。
モデル本体4.00bpw、出力head 6bit、MTP 4bit。自前の再量子化は行っていない。

## 実装と配置

LSAのglobal-query indexを層ごとに1 rankだけで保持し、Q/K投影の部分和を集約、
選択した論理token番号をbroadcastする。各rankのKVは従来どおりheadで分割する。
indexを二重保持する必要がなくなり、FP32精度を維持したままTPのメモリ使用量を減らした。

TP allocatorはMTPを読み込んだ後の空きメモリからindex ownerを配置する。
今回MTP側はindex 6層、他方は10層、両GPUとも全16 attention層のKV headを2つずつ保持した。
MTPのembedding/headは本体から共有する。

`--draft-device`でMTPのGPUを選択できる。
任意の`--draft-attention-window`も追加し、今回は65,536を指定した。
これはMTPのattention計算範囲だけを制限し、本体のcontextや検証を変更しない。
MTPのKV pool自体は全長を確保する。省略時はMTPもfull attentionであり、必須条件にはしていない。

TP decodeはFlat選択と各rankのKV readをHIP graph化した。
collectiveはgraph外、graphのpool/capture streamは同一TP worker内で共有する。

## 検証記録

- 8K入力・通常RoPE・MTP full attentionの生成が成功した（`lsa-tp2-mtp-small.json`）。
- 全長cacheを確保し、YaRN/任意MTP windowで8K入力を生成できた（`lsa-tp2-mtp-175mi-capacity.json`）。
- 実モデルと全長cache、合成履歴で末尾のq1/q5/q1/q5を実行し、本体/MTPの有限値・state巻き戻し・graph再実行を確認した（`lsa-tp2-mtp-175mi-memory.json`）。これは全文prefillや品質の証明には含めない。
- 関連テスト38件が成功した（`lsa-tp-unit-tests-pool-sharing.txt`）。

実入力1,835,008 tokenの全文prefillから生成まで成功し、終了コードは0だった。
cache容量は1,835,520 token、prefix再利用は0 token。
中央の番号「739182」を正しく回答し、生成logitsも有限値だった。

- prefill: 8,993.48秒（約149.9分、204.04 token/s）。モデル本体とMTPのprefillを含む。
- 生成: 7 token、8.16秒。初回HIP graph構築を含む短い機能試験であり、定常decode速度としては扱わない。
- MTP: 採用2、棄却18。提案20 token中の採用は10%。この1件ではMTP有無の速度比較は行っていない。
- graph再実行: index選択80回、local attention 160回。両TP rankで実行を確認した。
- 終了時のPyTorch割当: MTP側28.63 GiB、他方28.97 GiB。予約29.39/29.84 GiB、driverが返す空き1.70/1.19 GiB。
- prefill中の物理VRAM観測最大: MTP側31.98 GiB、他方31.56 GiB。生成終了時よりメモリ使用量が高くなる。

品質の確認は繰り返しの多い文書での番号検索1件に限る。
[要約JSON](summary.json)、[全文実行JSON](lsa-tp2-mtp-175mi-full.json)、
[log](lsa-tp2-mtp-175mi-full.log)、[VRAM観測](lsa-tp2-mtp-175mi-hardware.json)、
[起動手順](../../../doc/qwen38_27b_lsa_tp2_mtp_175mi.md)を参照。

## 全文試験で見つかった問題

1回目は1,834,752 tokenまでprefillした後、生成への遷移でページ表のpadding不一致が発生した。
物理7,170ページに対し、Generatorが表を7,184列へ整列していた。
有効な列をすべて維持してコピーするよう修正した（`padding-fix.json`）。

2回目も同じ位置までprefillしたが、層ごとに保持したgraph用sort作業領域でdecodeがOOMになった。
pool共有に加えてcapture streamも共通化することで解消し、全文を再投入する前に上記の末尾試験を通した
（`pool-sharing-fix.json`）。各失敗のlog/JSON/hardware/runtimeもこのディレクトリへ保存した。

これらの試行にはそれぞれ約2時間23分かかった。成功した容量確保・末尾試験と、
実際の全文処理の結果は区別して記録する。

環境・ソースのrevision/hashは`runtime.json`、モデルのrevision/hash照合は
`download-source.json`と`download-verification.json`を参照。
