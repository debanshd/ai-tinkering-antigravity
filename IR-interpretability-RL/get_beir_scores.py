import json

def get_scores(split_name):
    # Standard reported BEIR NDCG@10 scores
    scores = {
        "SciDocs": {"ColBERTv2": 0.167, "SPLADE-v2": 0.161, "COCO-DR": 0.170},
        "ArguAna": {"ColBERTv2": 0.468, "SPLADE-v2": 0.525, "COCO-DR": 0.540},
        "NFCorpus": {"ColBERTv2": 0.355, "SPLADE-v2": 0.344, "COCO-DR": 0.352},
        "FiQA": {"ColBERTv2": 0.352, "SPLADE-v2": 0.344, "COCO-DR": 0.366},
        "TREC-COVID": {"ColBERTv2": 0.704, "SPLADE-v2": 0.725, "COCO-DR": 0.763}
    }
    return scores.get(split_name, {"ColBERTv2": 0.0, "SPLADE-v2": 0.0, "COCO-DR": 0.0})

for ds in ["SciDocs", "ArguAna", "NFCorpus", "FiQA", "TREC-COVID"]:
    c = get_scores(ds)
    print(f"{ds}: ColBERTv2: {c['ColBERTv2']}, SPLADE-v2: {c['SPLADE-v2']}, COCO-DR: {c['COCO-DR']}")
