"""
evaluate.py - run every article demo's evaluation and save results/summary.json.

    python evaluate.py          # about 2 minutes on a laptop CPU

The slideshow and the docs read their numbers from results/summary.json, so
everything shown about MY demos is reproducible. Numbers quoted from the
papers themselves are labelled as such in the slides and docs.
"""

import json
import os
import time

import numpy as np

from demos import gorilla_demo, kine2go_demo, l2r_demo, voxposer_demo

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "summary.json")


def gorilla():
    db = gorilla_demo.database()
    bm25 = gorilla_demo.BM25(db)
    rows, correct_doc = [], 0
    for q in gorilla_demo.QUERIES:
        top, score = bm25.top(q.prompt, 1)[0]
        correct_doc += top["id"] == q.reference
        rows.append(dict(prompt=q.prompt, bm25_top1=top["id"], bm25_correct=top["id"] == q.reference,
                         verdicts={label: gorilla_demo.ast_verdict(code, q.reference, db)[0]
                                   for label, code in q.candidates}))
    # test-time documentation change (paper Sec. 4.2 / Fig. 6)
    prompt = gorilla_demo.QUERIES[1].prompt
    new_db = gorilla_demo.database(updated=True)
    aware, _, _ = gorilla_demo.retriever_aware_model(prompt, new_db)
    frozen = gorilla_demo.frozen_model("fcn_resnet50")
    return dict(api_db_size=len(db), bm25_top1_accuracy=correct_doc / len(gorilla_demo.QUERIES), queries=rows,
                doc_update=dict(retriever_aware_output=aware,
                                retriever_aware_verdict=gorilla_demo.ast_verdict(aware, "fcn_resnet50", new_db)[0],
                                frozen_output=frozen,
                                frozen_verdict=gorilla_demo.ast_verdict(frozen, "fcn_resnet50", new_db)[0]))


def main():
    t0 = time.time()
    summary = {}
    print("Article 1: Gorilla ...")
    summary["gorilla"] = gorilla()
    print("Article 2: VoxPoser (300 random scenes) ...")
    summary["voxposer"] = voxposer_demo.evaluate()
    print("Article 3: Language to Rewards ...")
    summary["l2r"] = dict(tasks=l2r_demo.evaluate(),
                          horizon_sweep={str(k): v for k, v in l2r_demo.horizon_sweep().items()})
    print("Article 4: Kine2Go ...")
    k = kine2go_demo.evaluate()
    k["tracker_reward_vs_noise_deg"] = {str(a): b for a, b in k["tracker_reward_vs_noise_deg"].items()}
    summary["kine2go"] = k
    summary["runtime_seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    print(f"wrote {OUT} in {summary['runtime_seconds']} s")


if __name__ == "__main__":
    main()
