import re

with open("main.tex", "r") as f:
    text = f.read()

# 1. Fix State Space (remove oracle)
state_orig = r"The state is the concatenation of: \(1\) the lossless 768-D dense query vector \$E_q\$, \(2\) the scalar values of the top 128 most active facets from the PRF neighborhood, \(3\) the \$\\log_2\$ normalized current target rank, and \(4\) the \$\\log_2\$ step-wise rank delta\."
state_new = r"The state is the concatenation of: (1) the lossless 768-D dense query vector $E_q$, (2) the scalar values of the top 128 most active facets from the PRF neighborhood, (3) Retrieval Entropy (the softmax entropy of the top-k PRF distances), and (4) Facet Variance (variance of the active expansion candidates), ensuring no oracle metrics are utilized during inference."
text = re.sub(state_orig, state_new, text)

# 2. Fix Torque (remove oracle initial target rank)
torque_orig_1 = r"To define this mathematically, let \$\\rho_\{init\}\$ represent the initial baseline rank of the target document, and \$\|\\mathcal\{C\}\|\$ represent the total number of documents in the corpus. The dynamic torque multiplier \$\\tau\$ is computed as the normalized logarithmic rank penalty:"
torque_new_1 = r"To define this mathematically, let $H_{prf}$ represent the Retrieval Entropy of the baseline PRF documents, and $H_{max}$ the maximum possible entropy. The dynamic torque multiplier $\tau$ is computed as a normalized scaling of this entropy constraint, reserving maximal leverage for queries with high semantic uncertainty:"
text = re.sub(torque_orig_1, torque_new_1, text)

torque_eq_orig = r"\\tau = \\max\\left\(0\.1, \\min\\left\(1\.0, \\frac\{\\log_2\(\\rho_\{init\}\)\}\{\\log_2\(\|\\mathcal\{C\}\|\)\}\\right\)\\right\)"
torque_eq_new = r"\tau = \max\left(0.1, \min\left(1.0, \frac{H_{prf}}{H_{max}}\right)\right)"
text = re.sub(torque_eq_orig, torque_eq_new, text)

torque_desc_orig = r"This formulation mathematically guarantees that queries already succeeding \(\$\\rho_\{init\} \\approx 1\$\) receive near-zero physical leverage \(\$\\tau \\approx 0\.1\$\), safeguarding them from PPO exploration noise, while deep failures receive maximal corrective torque\."
torque_desc_new = r"This formulation guarantees that queries already succeeding (low ambient entropy) receive near-zero physical leverage ($\tau \approx 0.1$), safeguarding them from PPO exploration noise, while uncertain queries receive maximal corrective torque."
text = re.sub(torque_desc_orig, torque_desc_new, text)

# 3. Clarify Negative Semantic Masking
mask_orig = r"This mathematical floor enables \"Negative Semantic Masking,\" allowing the agent to subtract false-positive noise explicitly from the dense vector."
mask_new = r"This mathematical floor enables \"Negative Semantic Masking\" (i.e., using bounded signed scalar multipliers), allowing the agent to subtract false-positive noise explicitly from the dense vector."
text = re.sub(mask_orig, mask_new, text)

# 4. Clarify Reward
reward_orig = r"We shape the implicit click feedback into a continuous Deep-Rescue Logarithmic Reward to accelerate convergence."
reward_new = r"During offline training on MS MARCO, we shape simulated click feedback (derived from ground-truth relevance labels) into a continuous Deep-Rescue Logarithmic Reward (a shaped log-rank reward) to accelerate convergence prior to zero-shot deployment."
text = re.sub(reward_orig, reward_new, text)

# 5. Clarify Rescue Rate
rr_orig_text = r"our approach exhibits a strict 2\.7\\% Rescue Rate \(where previously failed queries were decisively pushed into the Top 10\)"
rr_new_text = r"our approach exhibits a strict 2.7\% Rescue Rate (defined as the absolute percentage of queries that failed to appear in the Top 10 under the base dense model, but were successfully elevated into the Top 10 by the RL agent)"
text = re.sub(rr_orig_text, rr_new_text, text)

# 6. Add Related Work (SAEs for DR & Domain Adaptation)
rw_addition = r"Recent works exploring SAEs for dense retrieval (e.g., CL-SR and \textit{k}-sparse SAEs) similarly demonstrate that modifying latent features can steer retrieval. While these works utilize direct latent manipulation and contrastive objectives, our RL-driven approach complements them by providing a dynamic, policy-based routing mechanism. Furthermore, in zero-shot domain adaptation, methods like MoDIR explicitly target representation invariance, while advanced dense models (e.g., E5, BGE) offer strong uncorrected limits. We leave direct empirical comparisons to these massive architectures to future work, constrained here by the explicit goal of demonstrating lightweight scalar routing over a frozen baseline."
# Insert at the end of section 2.3
text = text.replace("providing continuous semantic steerability without the need for offline hard-negative retraining.", "providing continuous semantic steerability without the need for offline hard-negative retraining.\n\n" + rw_addition)

# 7. Move Compute & Hardware Efficiency to Supplementary
compute_section_regex = r"(\\subsection\{Compute \\& Hardware Efficiency\}.*?)(?=\\section\{Results and Analysis\})"
import re
match = re.search(compute_section_regex, text, re.DOTALL)
if match:
    compute_text = match.group(1)
    text = text.replace(compute_text, "")
    supp_target = "\\beginSupplementaryMaterials\n"
    text = text.replace(supp_target, supp_target + "\n\\section{Compute & Hardware Efficiency}\n" + compute_text + "\n")

    # Fix Latency text
    text = text.replace("Latency (ms/query)", "Routing Overhead (ms/query)")
    text = text.replace("0.001 ms", "0.001 ms*")
    text = text.replace("0.002 ms", "0.002 ms*")
    latency_note = r"*\textit{Note:} Reported latency reflects isolated routing/forward-pass overhead, not end-to-end ANN fetch time."
    text = text.replace(r"\end{tabular}%", r"\end{tabular}%" + "\n        " + latency_note)

# 8. Shrink introduction by cutting non-cited fluff
intro_fluff1 = r"In these failure modes, models retrieve documents based on spurious distributional overlaps or dominant but irrelevant vector magnitudes rather than strict contextual relevance~\\cite\{thakur2021\}\."
text = text.replace(intro_fluff1, "Models thus often retrieve documents based on spurious distributional overlaps rather than strict contextual relevance~\\cite{thakur2021}.")

intro_fluff2 = r"The continuous action space required to perturb dense vectors \$v \\in \\mathbb\{R\}\^d\$ without explicitly defined basis directions prevents agent convergence and exacerbates out-of-distribution errors."
text = text.replace(intro_fluff2, "")

# 9. Inner-product preservation
ip_orig = r"\\begin\{equation\}\nL_\{IP\} = \\sum_\{i, j \\in \\mathcal\{B\}\} \(\(E_i \\cdot E_j\) - \(\\hat\{E\}_i \\cdot \\hat\{E\}_j\)\)\^2\n\\end\{equation\}"
ip_new = r"To ensure topological fidelity, we include an inner-product preservation term $L_{IP}$:\n\\begin{equation}\nL_{IP} = \sum_{i, j \in \mathcal{B}} ((E_i \cdot E_j) - (\hat{E}_i \cdot \hat{E}_j))^2\n\\end{equation}\nThe final SAE loss is $L = L_{SAE} + \lambda_{IP} L_{IP}$, where $\lambda_{IP}$ strictly enforces batch-local topology preservation against the MSE/L1 constraints."
text = text.replace(ip_orig, ip_new)

# 10. Shrink Related Work by moving 2.1 RL in IR early background to supplementary
rl_bg_regex = r"(The formulation of search and recommendation as sequential decision-making processes has extensive precedent.*?\\cite\{yao2018\}\.)"
match = re.search(rl_bg_regex, text, re.DOTALL)
if match:
    rl_bg_text = match.group(1)
    text = text.replace(rl_bg_text, "Early works successfully modeled session-based search, document ranking, and recommendation as Markov Decision Processes (MDPs)~\cite{afsar2022, hofmann2013, ouyang2021}, utilizing bandits and deep RL to balance exploration against listwise returns (see Supplementary Material for extended related work).")
    supp_target = "\\beginSupplementaryMaterials\n"
    text = text.replace(supp_target, supp_target + "\n\\section{Extended Related Work: RL in Information Retrieval}\n" + rl_bg_text + "\n")

with open("main.tex", "w") as f:
    f.write(text)
