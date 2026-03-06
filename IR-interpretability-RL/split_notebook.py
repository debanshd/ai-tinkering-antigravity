import json

with open("run_experiments_kaggle.ipynb", "r") as f:
    orig = json.load(f)

# Base template
with open("run_meta_experiments_kaggle.ipynb", "r") as f:
    nb = json.load(f)

class_defs = """
import torch
import torch.nn as nn

class BipolarSAE(nn.Module):
    def __init__(self, d_in=768, d_out=4096, k=32):
        super().__init__()
        self.enc = nn.Linear(d_in, d_out)
        self.dec = nn.Linear(d_out, d_in)
        self.k = k
        self.eval()
        self.requires_grad_(False)
        
    def forward(self, x):
        pre_acts = self.enc(x)
        abs_acts = torch.abs(pre_acts)
        val, idx = torch.topk(abs_acts, self.k, dim=-1)
        mask = torch.zeros_like(pre_acts).scatter(-1, idx, 1.0)
        f = pre_acts * mask
        return self.dec(f), f

class LinearController(nn.Module):
    def __init__(self, in_features=128):
        super().__init__()
        self.linear = nn.Linear(in_features, in_features)
        nn.init.uniform_(self.linear.weight, -0.01, 0.01)
        nn.init.zeros_(self.linear.bias)
        self.eval()
        self.requires_grad_(False)
        
    def forward(self, x):
        return torch.clamp(self.linear(x), -2.0, 2.0), None
"""

insert_idx = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source = "".join(cell['source'])
        if "## Phase 10: Meta-Reviewer Benchmarks" in source:
            insert_idx = i
            break

if insert_idx is not None:
    nb['cells'].insert(insert_idx, {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [s + "\n" for s in class_defs.split("\n")]
    })

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        
        # Agents & SAEs Instantiations
        if "SparseAutoencoder(768)" in source_str:
             cell['source'] = [s.replace("SparseAutoencoder(768)", "BipolarSAE(768, 4096)") for s in cell['source']]
        if "PPOAgent().to(DEVICE)" in source_str:
             cell['source'] = [s.replace("PPOAgent().to(DEVICE)", "LinearController(128).to(DEVICE)") for s in cell['source']]
        if "PPOActorCritic" in source_str:
             cell['source'] = [s.replace("PPOActorCritic(768 + 128, 128).to(DEVICE)", "LinearController(128).to(DEVICE)") for s in cell['source']]
        
        # Mute DenseStateEncoder
        if "state_enc =" in source_str:
             cell['source'] = [s.replace("state_enc = DenseStateEncoder().to(DEVICE)", "state_enc = lambda *args: None") for s in cell['source']]
             cell['source'] = [s.replace("state_enc = None", "state_enc = lambda *args: None") for s in cell['source']]
             
        # Latency Matrix fixes
        if "state = state_enc(inference_batch, f_prf)" in source_str:
             cell['source'] = [s.replace("state = state_enc(inference_batch, f_prf)", "state = torch.gather(f_exp, 1, active_idx)") for s in cell['source']]
        if "perf_state = state_enc(perfect_q_embs, perf_f_prf)" in source_str:
             cell['source'] = [s.replace("perf_state = state_enc(perfect_q_embs, perf_f_prf)", "perf_state = torch.gather(perf_f_exp, 1, perf_active_idx)") for s in cell['source']]
        if "first_pass_enc - torch.matmul(delta_M, dummy_sae.decoder.weight)" in source_str:
             cell['source'] = [s.replace("first_pass_enc - torch.matmul(delta_M, dummy_sae.decoder.weight)", "first_pass_enc - torch.matmul(delta_M, dummy_sae.dec.weight.T)") for s in cell['source']]
        if "_, f_exp, f_neg = dummy_sae(first_pass_enc)" in source_str:
             cell['source'] = [s.replace("_, f_exp, f_neg = dummy_sae(first_pass_enc)", "_, f_exp = dummy_sae(first_pass_enc)") for s in cell['source']]
        
        # Orthogonality 16K -> 4096 Fix
        if "16384" in source_str:
             cell['source'] = [s.replace("16384", "4096") for s in cell['source']]
             
        # Fix Rank Stability _train variables
        if "query_embs_train" in source_str:
             cell['source'] = [s.replace("query_embs_train", "query_embs") for s in cell['source']]
        if "doc_embs_train" in source_str:
             cell['source'] = [s.replace("doc_embs_train", "doc_embs") for s in cell['source']]

with open("kaggle_meta_run/run_meta_experiments_kaggle.ipynb", "w") as f:
    json.dump(nb, f, indent=2)

print("Injected robust classes for V13.")
