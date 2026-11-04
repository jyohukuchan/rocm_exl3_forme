#!/usr/bin/env python3
"""Validate paired LSA benchmark reports and emit compact CSV/JSON summaries."""
import argparse
import csv
import json
import statistics
from pathlib import Path


def html_report(rows, metadata):
    template = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>LSAの速度・長文比較</title><style>
body{margin:0;color:var(--foreground,#222);font:14px var(--font-sans,system-ui);background:transparent}
#lsa-results{display:grid;gap:16px;min-width:0}#lsa-results>*{min-width:0}.controls,.legend{display:flex;gap:18px;align-items:center;flex-wrap:wrap}
label{display:flex;gap:8px;align-items:center}select{font:inherit;color:inherit;background:var(--background,white);border:1px solid var(--border,#ccc);border-radius:6px;padding:6px}
svg{display:block;width:100%;min-width:0;height:300px}svg text{fill:var(--muted-foreground,#666);font:12px system-ui}
.legend span{display:flex;gap:6px;align-items:center}.legend i{display:inline-block;width:20px;height:3px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}th,td{text-align:right;padding:7px 4px;border-bottom:1px solid var(--border,#ddd)}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}th{font-weight:500}
p{margin:0;line-height:1.6}.note{color:var(--muted-foreground,#666);font-size:12px}summary{cursor:pointer;margin-bottom:10px}h3{font-size:14px;font-weight:500;margin:0}
@media(max-width:400px){table{font-size:12px}th,td{padding:7px 2px}.legend{gap:12px}.controls{gap:12px}}
</style></head><body><main id="lsa-results">
<div class="controls"><label>GPU<select id="gpu"></select></label><label>指標<select id="phase"><option value="prefill">入力処理（prefill）</option><option value="decode">生成（decode）</option></select></label></div>
<div class="legend"><span><i style="background:var(--chart-1,#4477aa)"></i>通常attention</span><span><i style="background:var(--chart-2,#ee7733)"></i>LSA Flat</span><span><i style="background:var(--chart-3,#009988)"></i>LSA Tree</span></div>
<svg id="chart" role="img" aria-label="コンテキスト長別の処理速度"></svg>
<p class="note">token/s：大きいほど高速。横軸は入力長のlog2目盛。通常比1.00×が基準。__KV__・同一入力・__OUTPUT__ token生成・prefixヒット0。各長さ__WARMUPS__回のウォームアップ後、__REPEATS__回測定した中央値。</p>
<table id="speeds"><thead><tr><th>入力長</th><th>方式</th><th>token/s</th><th>通常比</th><th>番号取得</th></tr></thead><tbody></tbody></table>
<details><summary>応答完了時間・VRAM・logitsの差・ばらつき</summary><table id="latency"><thead><tr><th>入力長</th><th>方式</th><th>先頭token秒</th><th>完了秒</th><th>全体通常比</th></tr></thead><tbody></tbody></table><table id="details"><thead><tr><th>入力長</th><th>方式</th><th>ピークGiB</th><th>先頭KL</th><th>速度ばらつき</th></tr></thead><tbody></tbody></table></details>
<p class="note">番号取得は文書の__DEPTHS__に埋めた6桁コード。KLは最初の回答位置で通常attentionと比較した値です。広範な長文ベンチマークの精度を表すものではありません。VRAMは確保済みキャッシュ込みのPyTorch割当ピーク（__CAPACITIES__）。現行EXL3移植版のモデル全体を比較しています。</p>
</main><script>
const data=__DATA__;const root=document.getElementById('lsa-results');const gpu=root.querySelector('#gpu'),phase=root.querySelector('#phase'),svg=root.querySelector('#chart');
const modes=['dense','flat','tree'],names={dense:'通常',flat:'Flat',tree:'Tree'},colors=['var(--chart-1,#4477aa)','var(--chart-2,#ee7733)','var(--chart-3,#009988)'];
const tokenLabel=n=>(n%1024?(n/1024).toFixed(2):n/1024)+'K';
const gpus=[...new Set(data.map(r=>r.gpu))];gpus.sort((a,b)=>Number(b.includes('R9700'))-Number(a.includes('R9700')));gpu.innerHTML=gpus.map(g=>`<option>${g}</option>`).join('');
function render(){const rows=data.filter(r=>r.gpu===gpu.value),p=phase.value,key=p+'_tps',ratio=p+'_vs_dense',spread=p+'_spread';const lengths=[...new Set(rows.map(r=>r.tokens))].sort((a,b)=>a-b);
const w=Math.max(320,svg.clientWidth||728),h=300,L=54,R=18,T=15,B=35,max=Math.max(...rows.map(r=>r[key]))*1.12;
const x=n=>L+(Math.log2(n)-Math.log2(lengths[0]))/Math.max(1,Math.log2(lengths.at(-1))-Math.log2(lengths[0]))*(w-L-R),y=v=>T+(1-v/max)*(h-T-B);svg.setAttribute('viewBox',`0 0 ${w} ${h}`);
let content='';for(let i=0;i<=4;i++){const v=max*i/4,yy=y(v);content+=`<line x1="${L}" x2="${w-R}" y1="${yy}" y2="${yy}" stroke="var(--border,#ddd)"/><text x="${L-8}" y="${yy+4}" text-anchor="end">${Math.round(v)}</text>`}
for(const [i,n] of lengths.entries())content+=`<text x="${x(n)}" y="${h-10}" text-anchor="${i===0?'start':i===lengths.length-1?'end':'middle'}">${tokenLabel(n)}</text>`;
modes.forEach((m,i)=>{const rs=rows.filter(r=>r.mode===m).sort((a,b)=>a.tokens-b.tokens);content+=`<path d="${rs.map((r,j)=>(j?'L':'M')+x(r.tokens)+','+y(r[key])).join(' ')}" fill="none" stroke="${colors[i]}" stroke-width="2" ${i===2?'stroke-dasharray="5 3"':''}/>`;rs.forEach(r=>{const xx=x(r.tokens),yy=y(r[key]);content+=`<circle cx="${xx}" cy="${yy}" r="4" fill="${colors[i]}"><title>${names[m]} ${r.tokens/1024}K: ${r[key].toFixed(2)} token/s (${r[ratio].toFixed(2)}×)</title></circle>`})});svg.innerHTML=content;
root.querySelector('#speeds tbody').innerHTML=rows.map(r=>`<tr><td>${tokenLabel(r.tokens)}</td><td>${names[r.mode]}</td><td>${r[key].toFixed(1)}</td><td>${r[ratio].toFixed(2)}×</td><td>${r.retrieval_passes}/${r.retrieval_cases}</td></tr>`).join('');
root.querySelector('#latency tbody').innerHTML=rows.map(r=>`<tr><td>${tokenLabel(r.tokens)}</td><td>${names[r.mode]}</td><td>${r.ttft_seconds.toFixed(1)}</td><td>${r.wall_seconds.toFixed(1)}</td><td>${r.wall_vs_dense.toFixed(2)}×</td></tr>`).join('');
root.querySelector('#details tbody').innerHTML=rows.map(r=>`<tr><td>${tokenLabel(r.tokens)}</td><td>${names[r.mode]}</td><td>${r.peak_allocated_gib.toFixed(2)}</td><td>${r.kl.toExponential(2)}</td><td>${(r[spread]*100).toFixed(1)}%</td></tr>`).join('');console.log('rendered',gpu.value,p,rows.length,'rows')}
gpu.onchange=phase.onchange=render;new ResizeObserver(render).observe(svg);render();
</script></body></html>'''
    values = {"__DATA__": json.dumps(rows, ensure_ascii=True), "__KV__": "K%d/V%d" % tuple(metadata["cache_quant"]),
              "__OUTPUT__": str(metadata["new_tokens"]), "__WARMUPS__": str(metadata["warmups_per_length"]),
              "__REPEATS__": str(metadata["timed_repeats"]), "__DEPTHS__": "・".join(f"{d*100:g}%" for d in metadata["depths"])}
    capacities = []
    for cap in sorted({r["cache_tokens"] for r in rows}):
        lengths = sorted({r["tokens"] for r in rows if r["cache_tokens"] == cap})
        span = f"{lengths[0]/1024:g}K" + (f"〜{lengths[-1]/1024:g}K" if len(lengths)>1 else "")
        capacities.append(f"{span}: {cap:,} token")
    values["__CAPACITIES__"] = "、".join(capacities)
    for key, value in values.items():
        template = template.replace(key, value)
    return template


def markdown_report(rows, metadata):
    names = {"dense": "通常", "flat": "Flat", "tree": "Tree"}
    lines = ["# 通常attention / LSAの速度・長文比較", "",
             f"EXL3 Qwen3.8-27B、K{metadata['cache_quant'][0]}/V{metadata['cache_quant'][1]}、batch 1、"
             f"chunk {metadata['chunk_size']}、{metadata['new_tokens']} token固定生成。GPU power modeはauto。", "",
             f"各長さ{metadata['warmups_per_length']}回のwarmup後、{metadata['timed_repeats']}回の中央値。"
             "通常比は大きいほど高速。小さい差は測定・入力間のばらつきと合わせて判断する。", "",
             "[測定方法と再現コマンド](../../../doc/lsa_benchmark_method.md) / "
             "[比較ページ](comparison.html) / [CSV](summary.csv) / [集計JSON](summary.json)", ""]
    for gpu in dict.fromkeys(r["gpu"] for r in rows):
        group = [r for r in rows if r["gpu"] == gpu]
        lines += ["## " + gpu + " ×1", "", "|入力 token|方式|prefill token/s|通常比|生成 token/s|通常比|",
                  "|---:|---|---:|---:|---:|---:|"]
        for r in group:
            lines.append(f"|{r['tokens']:,}|{names[r['mode']]}|{r['prefill_tps']:.1f}|{r['prefill_vs_dense']:.2f}×|"
                         f"{r['decode_tps']:.2f}|{r['decode_vs_dense']:.2f}×|")
        lines += ["", "|入力 token|方式|番号取得|先頭KL（nats）|割当ピーク GiB|確保cache token|",
                  "|---:|---|---:|---:|---:|---:|"]
        for r in group:
            lines.append(f"|{r['tokens']:,}|{names[r['mode']]}|{r['retrieval_passes']}/{r['retrieval_cases']}|"
                         f"{r['kl']:.3g}|{r['peak_allocated_gib']:.2f}|{r['cache_tokens']:,}|")
        lines += ["", "|入力 token|方式|先頭tokenまで 秒|64 token完了まで 秒|全体通常比|",
                  "|---:|---|---:|---:|---:|"]
        for r in group:
            lines.append(f"|{r['tokens']:,}|{names[r['mode']]}|{r['ttft_seconds']:.1f}|"
                         f"{r['wall_seconds']:.1f}|{r['wall_vs_dense']:.2f}×|")
        lines.append("")
    lines += ["## 範囲と注意点", "",
              "番号取得は合成文書の限られた配置だけを評価している。KLも最初の回答位置の値であり、"
              "RULER全体の精度や文書全体のperplexityではない。", "",
              "生成速度にはindex・階層の更新も含む。固定64 tokenの平均を定常的な単一token速度と同一視しない。"
              "約256Kの追加測定ではtoken間隔も元JSONに保存している。", "",
              "LSAは保存するKVを削減しない。indexと作業領域を追加するため、メモリ削減策としては評価しない。"
              "ピークは各リクエスト中のPyTorch割当最大値を、本測定間で中央値にしたもの。", "",
              "測定中にTreeの管理キーハッシュ、選択処理のソート作業領域、LSAのdense prefixの"
              "KV展開範囲を修正した。通常attentionの完走データは保持し、Flat/Treeは最終修正後に"
              "全長さを再測定した。各方式のcommit・ハーネスSHA256・再開履歴を元JSONに残している。", "",
              "元レポート: " + ", ".join(f"[{name}]({name})" for name in dict.fromkeys(r['source_report'] for r in rows)), ""]
    return "\n".join(lines)


def combine(paths):
    combined = []
    for path in paths:
        report = json.loads(path.read_text())
        if not report.get("complete") or set(report["modes"]) != {"dense", "flat", "tree"}:
            raise ValueError(f"Incomplete comparison: {path}")
        dense = report["modes"]["dense"]
        expected = {(r["length"], r["trial"]): r for r in dense["rows"]}
        for mode, data in report["modes"].items():
            if {(r["length"], r["trial"]) for r in data["rows"]} != set(expected):
                raise ValueError(f"Unpaired cases: {path} {mode}")
            for row in data["rows"]:
                baseline = expected[row["length"], row["trial"]]
                if row["prompt_sha256"] != baseline["prompt_sha256"] or row["cached_tokens"]:
                    raise ValueError("Comparison contains mismatched prompts or cached prefixes")
                if row["new_tokens"] != report["new_tokens"]:
                    raise ValueError("Output lengths differ")
            for length, summary in data["summaries"].items():
                baseline = dense["summaries"][length]
                measured = [r for r in data["rows"] if r["length"] == int(length) and not r["warmup"]]
                dense_measured = [r for r in dense["rows"] if r["length"] == int(length) and not r["warmup"]]
                wall = statistics.median(r["wall_seconds"] for r in measured)
                dense_wall = statistics.median(r["wall_seconds"] for r in dense_measured)
                combined.append(dict(gpu=report["gpu"], mode=mode, tokens=int(length),
                    prefill_tps=summary["engine_prefill_tps"],
                    prefill_vs_dense=summary["engine_prefill_tps"] / baseline["engine_prefill_tps"],
                    prefill_spread=summary["engine_prefill_tps_spread"],
                    decode_tps=summary["observed_decode_tps"],
                    decode_vs_dense=summary["observed_decode_tps"] / baseline["observed_decode_tps"],
                    decode_spread=summary["observed_decode_tps_spread"],
                    ttft_seconds=summary["observed_ttft_seconds"],
                    wall_seconds=wall, wall_vs_dense=dense_wall / wall,
                    peak_allocated_gib=summary["peak_allocated_bytes"] / 1024**3,
                    paged_cache_gib=data["cache_storage_bytes"] / 1024**3,
                    cache_tokens=report["cache_tokens"],
                    retrieval_passes=summary["retrieval_passes"],
                    retrieval_cases=summary["retrieval_cases"],
                    kl=summary.get("first_response_kl_dense_to_mode"),
                    decode_after_8_tps=summary.get("decode_after_8_tps"),
                    token_interval_p50_ms=(summary["token_interval_p50_seconds"] * 1000 if "token_interval_p50_seconds" in summary else None),
                    token_interval_p95_ms=(summary["token_interval_p95_seconds"] * 1000 if "token_interval_p95_seconds" in summary else None),
                    token_interval_max_ms=(summary["token_interval_max_seconds"] * 1000 if "token_interval_max_seconds" in summary else None),
                    timed_runs=summary["timed_runs"], source_report=path.name))
                combined[-1]["mode_source_commit"] = data.get("source", report["source"])["commit"]
                combined[-1]["harness_sha256"] = data.get("source", report["source"])["harness_sha256"]
    return sorted(combined, key=lambda r: (r["gpu"], r["tokens"], ["dense", "flat", "tree"].index(r["mode"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = combine(args.reports)
    reports = [json.loads(path.read_text()) for path in args.reports]
    metadata = {key: reports[0][key] for key in ("cache_quant", "chunk_size", "new_tokens", "warmups_per_length", "timed_repeats")}
    for report in reports:
        if any(report[key] != value for key, value in metadata.items()):
            raise ValueError("Reports use different output lengths or comparison settings")
    metadata["depths"] = sorted({r['depth'] for report in reports for block in report['modes'].values() for r in block['rows']})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    (args.output_dir / "comparison.html").write_text(html_report(rows, metadata))
    (args.output_dir / "README.md").write_text(markdown_report(rows, metadata))
    with (args.output_dir / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['gpu']} {row['tokens']:>6} {row['mode']:<5} "
              f"prefill {row['prefill_tps']:.1f} ({row['prefill_vs_dense']:.2f}x) "
              f"decode {row['decode_tps']:.2f} ({row['decode_vs_dense']:.2f}x) "
              f"retrieval {row['retrieval_passes']}/{row['retrieval_cases']} KL {row['kl']:.4g}")


if __name__ == "__main__":
    main()
