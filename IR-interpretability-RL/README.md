# Online Correction of Semantic Drift in Dense Retrieval via Implicit Feedback and Sparse Autoencoders

This repository contains the code and resources for the paper **"Online Correction of Semantic Drift in Dense Retrieval via Implicit Feedback and Sparse Autoencoders"**.

## Abstract

The transition from lexical matching to dense vector representation has fundamentally altered information retrieval architectures. While dense retrieval captures deep contextual semantics, it introduces interpretability and control challenges, often leading to semantic drift. This project proposes leveraging implicit user feedback as an online correction mechanism, formulating the retrieval correction process as an Implicit Reinforcement Learning from Human Feedback (RLHF) objective. By resolving the continuous-space intractability with a structural bottleneck derived from Sparse Autoencoders (SAEs), we train a highly lightweight Proximal Policy Optimization (PPO) agent to execute Negative Semantic Masking at query time, establishing a new state-of-the-art for zero-shot domain adaptation on BEIR datasets with minimal computational overhead.

## Key Contributions

*   **Continuous-Space Intractability Solution**: Formulates semantic drift correction in dense retrieval as an MDP bounded by sparse dictionary features, explicitly solving the continuous-space intractability problem without requiring full-model contrastive retraining.
*   **Context-Aware Latent-to-Dense Routing**: Demonstrates that implicit user feedback can guide a PPO agent to execute Negative Semantic Masking and Rank-Proportional Dynamic Torque at query time.
*   **State-of-the-Art Zero-Shot Domain Adaptation**: Establishes a new state-of-the-art for zero-shot domain adaptation on BEIR datasets, achieving a +75.3% rescue rate (0.4675 NDCG@10) on Arguana and a +42.3% rescue rate (0.1562 NDCG@10) on SciDocs.

## Methodology

The approach utilizes a two-phase architecture to correct semantic drift without the need for computationally heavy offline whole-model contrastive retraining:

1.  **The Sparse State Encoder**: Projects dense representations into a high-dimensional, sparse linear space to isolate distinct semantic facets. This eliminates the "curse of dimensionality" and acts as a structural bottleneck.
2.  **RL-Driven Semantic Steering (PPO)**: A Proximal Policy Optimization (PPO) agent dynamically adjusts query vectors in the sparse space, bounded by a bi-directional action space (yielding *Negative Semantic Masking*). The corrections are scaled back into the dense space using *Rank-Proportional Dynamic Torque* to prevent exploration noise from degrading high-performing queries.

## Results

Zero-shot retrieval performance (NDCG@10) on out-of-domain BEIR datasets:

| Dataset | Domain | Dense Base | RL-Steered (Ours) | Rescue Rate |
| :--- | :--- | :--- | :--- | :--- |
| **SciDocs** | Scientific | 0.1143 | **0.1562** | +42.3% |
| **Arguana** | Argumentative | 0.3371 | **0.4675** | +75.3% |
| **NFCorpus** | Medical | 0.2662 | **0.3569** | +64.0% |

The architecture executes these improvements with minimal computational overhead practically achieving $O(1)$ routing time overhead relative to the dense baseline.

## Repository Contents

*   [`arguana-rlc.ipynb`](arguana-rlc.ipynb) - Jupyter Notebook / Colab containing the prototype implementation and demonstration of the RL agent interacting with the retrieval environment on the ArguAna dataset.

## Getting Started

You can explore the methodology and run the agent directly by opening the `arguana-rlc.ipynb` notebook.
