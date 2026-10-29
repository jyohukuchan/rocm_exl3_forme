#!/usr/bin/env python3
"""Prepare llama.cpp multi-case dump jobs + write the KLD manifest after.
Usage: prep_llama_ref.py <corpus.json> <workdir>   (writes ids/*.txt + jobs.txt)
Then run dump_logits --job jobs.txt, then:
       prep_llama_ref.py <corpus.json> <workdir> --manifest --model <name> --vocab <V>
"""
import hashlib
import json
import sys
from pathlib import Path


def main():
    corpus_p, work = sys.argv[1], Path(sys.argv[2])
    corpus = json.loads(Path(corpus_p).read_text())["cases"]
    idsdir = work / "ids"
    if "--manifest" in sys.argv:
        mi = sys.argv.index("--model")
        model = sys.argv[mi + 1]
        vi = sys.argv.index("--vocab")
        vocab = int(sys.argv[vi + 1])
        manifest = {"model": model, "engine": "llama.cpp", "dtype": "bf16", "cases": []}
        for case in corpus:
            f = work / f"{case['id']}.f32"
            import numpy as np
            a = np.fromfile(f, dtype="<f4").reshape(len(case["token_ids"]), vocab)
            nf = int((~np.isfinite(a)).sum())
            manifest["cases"].append(
                {
                    "id": case["id"],
                    "logits_file": f"{case['id']}.f32",
                    "shape": [len(case["token_ids"]), vocab],
                    "nonfinite_count": nf,
                    "positions": list(range(len(case["token_ids"]))),
                    "input_token_ids_sha256": "sha256:"
                    + hashlib.sha256(json.dumps(case["token_ids"], separators=(",", ":")).encode()).hexdigest(),
                }
            )
            print(case["id"], "rows=", len(case["token_ids"]), "nonfinite=", nf)
        (work / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
        return
    idsdir.mkdir(parents=True, exist_ok=True)
    with open(work / "jobs.txt", "w") as jf:
        for case in corpus:
            (idsdir / f"{case['id']}.txt").write_text(" ".join(map(str, case["token_ids"])))
            jf.write(f"{work}/{case['id']}.f32 {idsdir}/{case['id']}.txt\n")
    print(f"wrote {len(corpus)} jobs")


if __name__ == "__main__":
    main()
