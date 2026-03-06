import json

def get_counts(split_name):
    # Dummy function to just get the number of queries and docs from standard BEIR counts
    counts = {
        "SciDocs": {"queries": 1000, "docs": 25657},
        "ArguAna": {"queries": 1406, "docs": 8674},
        "NFCorpus": {"queries": 323, "docs": 3633},
        "FiQA": {"queries": 648, "docs": 57638},
        "TREC-COVID": {"queries": 50, "docs": 171332}
    }
    return counts.get(split_name, {"queries": 0, "docs": 0})

print("Dataset query counts:")
for ds in ["SciDocs", "ArguAna", "NFCorpus", "FiQA", "TREC-COVID"]:
    c = get_counts(ds)
    print(f"- {ds}: {c['queries']} queries")
