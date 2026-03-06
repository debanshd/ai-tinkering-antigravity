import re

with open("main.tex", "r") as f:
    text = f.read()

# 1. Remove verbose paragraph from Problem Formulation
prob_form_fluff = r"Consequently, the physical mechanism of retrieval relies entirely on geometric proximity: $score(q, d) = E_q \cdot E_d$. While this formulation is computationally efficient via Approximate Nearest Neighbor \(ANN\) indexing, it is highly sensitive to the topological structure of $\mathbb{R}^d$. If an encoder learns to project multiple distinct concepts into overlapping neighborhoods to minimize a contrastive loss, the frozen retrieval system lacks the mathematical vocabulary to disentangle them at inference time."
text = text.replace(prob_form_fluff, "Consequently, $score(q, d) = E_q \cdot E_d$. This efficiency makes frozen models highly sensitive to the topological structure of $\mathbb{R}^d$ and vulnerable to semantic entanglement at inference time.")

# 2. Remove verbose paragraph from Problem Formulation
prob_form_fluff2 = r"Traditional methods for correcting this drift, such as hard-negative contrastive mining, attempt to globally reshape the entire embedded topology by readjusting the weights of $\Phi_Q$ and $\Phi_D$. This is mathematically analogous to moving an entire solar system to correct the orbit of a single planet. It requires massive compute, complete re-indexing of the corpus, and risks catastrophic forgetting of previously stable semantic relationships."
text = text.replace(prob_form_fluff2, "Correcting drift typically requires moving the entire encoder space via hard-negative fine-tuning, which demands massive compute, complete corpus re-indexing, and risks catastrophic forgetting.")

# 3. Remove verbose explanations of SAEs in Section 4.2
sae_fluff1 = r"Sparse Autoencoders \(SAEs\) have recently emerged as a highly effective tool for mechanistic interpretability in large language models. The foundational premise of an SAE is to train a shallow, structural bottleneck that projects entangled, polysemantic dense activations into a higher-dimensional, overcomplete basis where individual dimensions correspond to monosemantic, interpretable human concepts."
text = text.replace(sae_fluff1, "Sparse Autoencoders (SAEs) project entangled dense activations into a higher-dimensional, overcomplete basis of interpretable semantic facets.")

sae_fluff2 = r"If a dense query vector $v \in \mathbb{R}^{d}$ is simply a linear superposition of $k$ distinct semantic features (where $k \gg d$), the SAE attempts to discover the original $k$ directions of variation without relying on orthogonal bases. By driving the majority of activations to strictly zero, the SAE forces each active node to take on a highly specific semantic meaning."
text = text.replace(sae_fluff2, "")

# 4. Remove Broader Impact Fluff
impact_fluff1 = r"The core premise of our methodology—leveraging implicit user clicks as a delayed reward signal—inherits the well-documented risks of click-driven feedback loops. User interactions are notoriously susceptible to position bias~\\cite\{craswell2008\} and represent a highly localized, often flawed approximation of true relevance~\\cite\{joachims2017\}."
text = text.replace(impact_fluff1, "Leveraging implicit clicks inherits risks of position bias~\\cite{craswell2008} and noisy relevance approximations~\\cite{joachims2017}.")

impact_fluff2 = r"From a broader societal perspective, training an autonomous agent to dynamically amplify or suppress semantic concepts based purely on interaction maximization introduces the risk of algorithmic reinforcement of societal biases~\\cite\{baeza-yates2018, noble2018\}. If a demographic cohort exhibits a statistically significant click-preference toward ideologically biased or factually spurious results, an unconstrained PPO agent will optimize its objective by treating the corresponding latent semantic facets as highly rewarding. This failure mode represents a classic instance of RLHF reward hacking~\\cite\{amodei2016, skalse2022\}, wherein the agent perfectly minimizes its surrogate objective at the direct expense of the system's intended semantic integrity."
text = text.replace(impact_fluff2, "Unconstrained interaction maximization also risks algorithmic reinforcement of societal biases~\\cite{baeza-yates2018, noble2018} or reward hacking~\\cite{amodei2016, skalse2022}, where the agent optimizes its surrogate objective at the direct expense of semantic integrity.")

# 5. Remove RAG Fluff
rag_fluff = r"The recent paradigm shift toward Retrieval-Augmented Generation \(RAG\) places an unprecedented premium on exact-match precision. When Large Language Models \(LLMs\) are grounded by external knowledge bases, semantic drift in the retrieval phase directly induces generative hallucination. If a dense model retrieves tangentially related but factually misaligned documents—such as fetching narcotics trafficking statistics for a query about athletic doping—the downstream LLM will confidently synthesize an inaccurate response based on that polluted context window."
text = text.replace(rag_fluff, "In Retrieval-Augmented Generation (RAG), semantic drift in the retrieval phase directly induces generative hallucination.")

rag_fluff2 = r"Our RL-Steered architecture acts as a highly interpretable, pre-generative firewall. Because the agent executes Negative Semantic Masking at the latent level, it structurally filters out out-of-domain noise before the documents ever reach the generative context window. By mathematically proving that we can dynamically map and correct the failure states of frozen LLM embeddings, this framework provides a robust retrieval foundation for high-stakes, enterprise RAG applications where factual density is paramount."
text = text.replace(rag_fluff2, "Our RL-Steered architecture acts as a pre-generative firewall, filtering out-of-domain noise before formatting the LLM context window.")

# 6. Remove fully the Supplementary \beginSupplementaryMaterials command to fix the LaTeX parsing issue
text = text.replace("\\beginSupplementaryMaterials\n", "\n")

with open("main.tex", "w") as f:
    f.write(text)

