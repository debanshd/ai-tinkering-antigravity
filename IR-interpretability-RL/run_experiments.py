import os
import gc
import json
import time
import math
import warnings
import logging

# Ensure Kaggle environment has required packages
print("Installing required packages on Kaggle worker...")
os.system("pip install -q beir sentence-transformers rank_bm25 accelerate")

import numpy as np
import scipy.stats as stats
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F_func
from sentence_transformers import SentenceTransformer
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from rank_bm25 import BM25Okapi

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\\n--- System Utilizing: {DEVICE.type.upper()} ---")

def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def compute_ndcg_at_k(retrieved_indices, relevant_indices, k=10):
    dcg = 0.0
    for i, idx in enumerate(retrieved_indices[:k]):
        if idx in relevant_indices:
            dcg += 1.0 / math.log2(i + 2)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_indices), k)))
    return dcg / idcg if idcg > 0 else 0.0

# ==============================================================================
# DATA LOADING (FULL BEIR SPLITS)
# ==============================================================================
def load_dataset(dataset_name, max_q=None, max_d=None):
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    out_dir = util.download_and_unzip(url, "datasets")
    split = "dev" if dataset_name == "msmarco" else "test"
    corpus, queries, qrels = GenericDataLoader(data_folder=out_dir).load(split=split)
    
    query_ids = []
    relevant_doc_ids = []
    
    # Strictly valid targets
    for qid in list(queries.keys()):
        if qid in qrels:
            valid_targets = [did for did, score in qrels[qid].items() if score > 0 and did in corpus]
            if valid_targets:
                query_ids.append(qid)
                relevant_doc_ids.extend(valid_targets)
        if max_q and len(query_ids) >= max_q: break
        
    all_doc_ids = sorted(list(set(relevant_doc_ids + list(corpus.keys()))))
    if max_d and len(all_doc_ids) > max_d:
        all_doc_ids = list(set(relevant_doc_ids + all_doc_ids[:max_d]))
        
    doc_id_to_idx = {did: i for i, did in enumerate(all_doc_ids)}
    docs = [corpus[did].get("text") for did in all_doc_ids]
    qs = [queries[qid] for qid in query_ids]
    target_idx = [[doc_id_to_idx[did] for did in qrels[qid].keys() if did in doc_id_to_idx] for qid in query_ids]
    
    print(f"-> LOADED {dataset_name.upper()}: {len(qs)} Queries, {len(docs)} Documents")
    return qs, docs, target_idx

# ==============================================================================
# ARCHITECTURE
# ==============================================================================
class SparseStateEncoder(nn.Module):
    def __init__(self, dense_dim=768, sparse_dim=4096):
        super().__init__()
        self.encoder = nn.Linear(dense_dim, sparse_dim)
        self.relu = nn.ReLU()
        self.decoder = nn.Linear(sparse_dim, dense_dim)
    def forward(self, x):
        F = self.relu(self.encoder(x))
        return F, self.decoder(F)

class LeakFreeSteeringEnv:
    def __init__(self, q_F, d_F, q_dense, d_dense, dec_w, target_idx=None, is_eval=False):
        self.q_F, self.d_F, self.q_dense, self.d_dense, self.dec_w = q_F, d_F, q_dense, d_dense, dec_w
        self.max_steps = 3
        self.num_q, self.num_d = len(q_F), len(d_F)
        self.is_eval = is_eval
        
        if not is_eval and target_idx is not None:
            self.target_mask = torch.zeros((self.num_q, self.num_d), dtype=torch.bool, device=DEVICE)
            for i, t in enumerate(target_idx): self.target_mask[i, t] = True

    def _state(self):
        facet_vals = self.expansion_candidates.gather(1, self.active_idx)
        scores = torch.matmul(self.q_curr, self.d_dense.T)
        probs = F_func.softmax(torch.topk(scores, 10, dim=1).values, dim=1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=1).unsqueeze(1)
        variance = facet_vals.var(dim=1, unbiased=False).unsqueeze(1)
        return torch.cat([self.q_curr.detach(), facet_vals, entropy, variance], dim=1)

    def reset(self):
        self.step_cnt = 0
        self.M = torch.zeros_like(self.q_F)
        self.q_curr = self.q_dense.clone()
        
        scores = torch.matmul(self.q_curr, self.d_dense.T)
        top_docs = self.d_F[torch.topk(scores, 10, dim=1).indices]
        w = F_func.softmax(torch.arange(10, 0, -1, dtype=torch.float, device=DEVICE), dim=0).view(1, 10, 1)
        self.expansion_candidates = self.q_F + (top_docs * w).sum(dim=1)
        self.active_idx = torch.topk(self.expansion_candidates, 128, dim=1).indices
        
        self.base_ranks = None
        if not self.is_eval:
            self.best_targets = scores.masked_fill(~self.target_mask, -1e4).max(dim=1).values
            self.curr_ranks = (scores > self.best_targets.unsqueeze(1)).sum(dim=1) + 1
        else:
            # We must track actual base ranks for RR
            pass # handled externally
            
        return self._state()

    def step(self, actions):
        self.step_cnt += 1
        self.M = 0.5 * self.M
        self.M.scatter_add_(1, self.active_idx, 0.2 * actions)
        self.M = torch.clamp(self.M, -2.0, 2.0)
        
        dense_shift = torch.matmul(self.M, self.dec_w.T)
        self.q_curr = F_func.normalize(self.q_dense + (0.3 * dense_shift), p=2, dim=1)
        
        new_scores = torch.matmul(self.q_curr, self.d_dense.T)
        reward = torch.zeros(self.num_q, device=DEVICE)
        
        if not self.is_eval:
            new_best = new_scores.masked_fill(~self.target_mask, -1e4).max(dim=1).values
            new_ranks = (new_scores > new_best.unsqueeze(1)).sum(dim=1) + 1
            old_ips = 1.0 / torch.log2(self.curr_ranks.float() + 1.0)
            new_ips = 1.0 / torch.log2(new_ranks.float() + 1.0)
            reward = (new_ips - old_ips) * 100.0
            self.curr_ranks = new_ranks
            
        dones = torch.full((self.num_q,), self.step_cnt >= self.max_steps, dtype=torch.bool, device=DEVICE)
        return self._state(), reward, dones, new_scores

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim=128, hidden_dim=256):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, action_dim), nn.Tanh())
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
    
    def evaluate(self, states, actions):
        mean = self.actor(states)
        std = torch.exp(torch.clamp(self.log_std, -2.0, 0.5)).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(actions).sum(dim=-1), self.critic(states).squeeze(-1), dist.entropy().sum(dim=-1)
    
    def act(self, states):
        mean = self.actor(states)
        std = torch.exp(torch.clamp(self.log_std, -2.0, 0.5)).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        act = dist.sample()
        return act.detach(), dist.log_prob(act).sum(dim=-1).detach()

# ==============================================================================
# MAIN SCRIPT
# ==============================================================================
if __name__ == "__main__":
    RESULTS = {}

    print("Loading Offline Training data (MS MARCO)...")
    ms_q, ms_d, ms_targets = load_dataset("msmarco", max_q=5000, max_d=25000)
    
    contriever = SentenceTransformer('facebook/contriever', device=DEVICE)
    with torch.no_grad():
        ms_q_dense = F_func.normalize(torch.tensor(contriever.encode(ms_q, batch_size=256)), p=2, dim=1).to(DEVICE)
        ms_d_dense = F_func.normalize(torch.tensor(contriever.encode(ms_d, batch_size=256)), p=2, dim=1).to(DEVICE)

    sae_layer = SparseStateEncoder().to(DEVICE)
    opt = optim.Adam(sae_layer.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    all_ms = torch.cat([ms_q_dense, ms_d_dense])
    
    print("Training SAE offline...")
    for epoch in range(15):
        idx = torch.randperm(all_ms.size(0))
        for i in range(0, all_ms.size(0), 1024):
            batch = all_ms[idx[i:i+1024]]
            opt.zero_grad()
            F, rec = sae_layer(batch)
            loss = mse(rec, batch) + (1e-4 * F.mean())
            loss.backward(); opt.step()
    
    with torch.no_grad():
        ms_q_F = F_func.normalize(sae_layer(ms_q_dense)[0], p=2, dim=1)
        ms_d_F = F_func.normalize(sae_layer(ms_d_dense)[0], p=2, dim=1)
        dec_weight = sae_layer.decoder.weight.detach()

    ppo_policy = ActorCritic(state_dim=898, action_dim=128).to(DEVICE)
    opt_ppo = optim.Adam(ppo_policy.parameters(), lr=1e-3)
    env = LeakFreeSteeringEnv(ms_q_F, ms_d_F, ms_q_dense, ms_d_dense, dec_weight, ms_targets)
    
    print("Training PPO Agent offline...")
    for ep in range(150):
        mem = {'s':[], 'a':[], 'logp':[], 'r':[], 'd':[]}
        s = env.reset()
        for _ in range(env.max_steps):
            a, logp = ppo_policy.act(s)
            ns, r, d, _ = env.step(a)
            mem['s'].append(s); mem['a'].append(a); mem['logp'].append(logp); mem['r'].append(r); mem['d'].append(d)
            s = ns
        old_s = torch.cat(mem['s']); old_a = torch.cat(mem['a']); old_logp = torch.cat(mem['logp'])
        ret = torch.zeros_like(r)
        discounted = []
        for step_r, step_d in zip(reversed(mem['r']), reversed(mem['d'])):
            ret = step_r + 0.99 * ret * (~step_d).float()
            discounted.insert(0, ret)
        discounted = torch.cat(discounted)
        discounted = (discounted - discounted.mean()) / (discounted.std() + 1e-8)
        for _ in range(3):
            logp, val, ent = ppo_policy.evaluate(old_s, old_a)
            adv = discounted - val.detach()
            ratio = torch.exp(logp - old_logp)
            loss = -torch.min(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv) + 0.5 * nn.MSELoss()(val, discounted)
            opt_ppo.zero_grad(); loss.mean().backward(); opt_ppo.step()
            
    clear_vram()
    
    # -------------------------------------------------------------
    # Full BEIR Evaluations
    # -------------------------------------------------------------
    eval_datasets = ["arguana", "scidocs", "fiqa"]
    
    for dset in eval_datasets:
        print(f"\\n--- Evaluating {dset.upper()} ---")
        qs, docs, targets = load_dataset(dset) # LOAD FULL DATASET
        
        # 1. BM25 Baseline
        print("Running BM25...")
        tokenized_corpus = [d.split(" ") for d in docs]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_ndcgs = []
        for i, q in enumerate(qs):
            scores = bm25.get_scores(q.split(" "))
            top_i = np.argsort(scores)[::-1][:10].tolist()
            bm25_ndcgs.append(compute_ndcg_at_k(top_i, targets[i], 10))
        res_bm25 = np.mean(bm25_ndcgs)
        print(f"BM25 NDCG@10: {res_bm25:.4f}")
        
        # 2. Dense Contriever Base
        print("Running Contriever Base...")
        with torch.no_grad():
            q_dense = F_func.normalize(torch.tensor(contriever.encode(qs, batch_size=128)), p=2, dim=1).to(DEVICE)
            d_dense = F_func.normalize(torch.tensor(contriever.encode(docs, batch_size=128)), p=2, dim=1).to(DEVICE)
            
            start_t = time.time()
            base_scores = torch.matmul(q_dense, d_dense.T)
            base_time = (time.time() - start_t) / len(qs) * 1000 # ms
            
        base_ranks = []
        base_ndcgs = []
        for i in range(len(qs)):
            ti = torch.topk(base_scores[i], 10).indices.tolist()
            base_ndcgs.append(compute_ndcg_at_k(ti, targets[i], 10))
            
            best_t = base_scores[i][targets[i]].max().item() if len(targets[i]) > 0 else -1e4
            base_ranks.append((base_scores[i] > best_t).sum().item() + 1)
            
        res_base = np.mean(base_ndcgs)
        print(f"Base Contriever NDCG@10: {res_base:.4f} (Base Routing Latency: {base_time:.3f}ms)")
        
        # 3. RL-Steered (Zero-Shot)
        print("Running RL-Steered...")
        with torch.no_grad():
            q_F = F_func.normalize(sae_layer(q_dense)[0], p=2, dim=1)
            d_F = F_func.normalize(sae_layer(d_dense)[0], p=2, dim=1)
            
            start_t = time.time()
            eval_env = LeakFreeSteeringEnv(q_F, d_F, q_dense, d_dense, dec_weight, is_eval=True)
            s = eval_env.reset()
            ppo_policy.eval()
            for _ in range(eval_env.max_steps):
                a = ppo_policy.actor(s)
                s, _, _, final_scores = eval_env.step(a)
            rl_time = (time.time() - start_t) / len(qs) * 1000 # ms
        
        rl_ndcgs = []
        rl_ranks = []
        for i in range(len(qs)):
            ti = torch.topk(final_scores[i], 10).indices.tolist()
            rl_ndcgs.append(compute_ndcg_at_k(ti, targets[i], 10))
            best_t = final_scores[i][targets[i]].max().item() if len(targets[i]) > 0 else -1e4
            rl_ranks.append((final_scores[i] > best_t).sum().item() + 1)
            
        res_rl = np.mean(rl_ndcgs)
        
        # Calculate Rescue Rate
        failed_base = np.array(base_ranks) > 10
        rescued = failed_base & (np.array(rl_ranks) <= 10)
        rescue_rate = (rescued.sum() / failed_base.sum() * 100) if failed_base.sum() > 0 else 0.0
        
        print(f"RL-Steered NDCG@10: {res_rl:.4f} (RL Pipeline Latency: {rl_time:.3f}ms)")
        print(f"Rescue Rate: {rescue_rate:.2f}%")
        
        # SAE Feature Correlation (Orthogonality check)
        # Compute correlation matrix of top 128 active facets for the first batch of queries
        facet_vars = s[:128, 768:768+128] # slice facets from state
        corr_matrix = torch.corrcoef(facet_vars.T).cpu().numpy()
        np.fill_diagonal(corr_matrix, 0) # ignore self-correlation
        mean_abs_corr = np.nanmean(np.abs(corr_matrix))
        print(f"Mean Abs SAE Feature Correlation: {mean_abs_corr:.4f}")
        
        RESULTS[dset] = {
            "BM25": res_bm25,
            "Dense_Base": res_base,
            "RL_Steered": res_rl,
            "Rescue_Rate": float(rescue_rate),
            "Base_Latency_ms": float(base_time),
            "E2E_RL_Latency_ms": float(rl_time),
            "SAE_Mean_Correlation": float(mean_abs_corr)
        }
        clear_vram()

    with open("eval_results.json", "w") as f:
        json.dump(RESULTS, f, indent=4)
        
    print("\\nEXPERIMENT COMPLETE. Results saved to eval_results.json")
