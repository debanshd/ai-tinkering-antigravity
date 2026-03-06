import json

with open('run_experiments_kaggle.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            # Line 589
            line = line.replace("probs = F.softmax(scores_tensor[:, :10], dim=1)",
                                "top10_scores, _ = torch.topk(scores_tensor, 10, dim=1)\n    probs = F.softmax(top10_scores, dim=1)")
            # Line 769
            line = line.replace("probs = F.softmax(bge_scores[:, :10], dim=1)",
                                "top10_bge_scores, _ = torch.topk(bge_scores, 10, dim=1)\n    probs = F.softmax(top10_bge_scores, dim=1)")
            # Line 851
            line = line.replace("probs = F.softmax(scores[:, :10], dim=1)",
                                "top10_scores_sup, _ = torch.topk(scores_tensor, 10, dim=1)\n    probs = F.softmax(top10_scores_sup, dim=1)")
            new_source.append(line)
        cell['source'] = new_source

with open('run_experiments_kaggle.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)
print("Updated notebook!")
