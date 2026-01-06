# Research Paper Outlines for PhD Thesis

This document provides **detailed paper structures** for each recommended direction, including titles, abstracts, and section breakdowns.

---

## 📄 Paper 1: Explainable CTR (Direction 1)

### **Title:**
"XCTR: Towards Causally Grounded and Interpretable Click-Through Rate Prediction"

### **Abstract (250 words):**

Click-through rate (CTR) prediction is fundamental to digital advertising and recommender systems, with models increasingly leveraging deep neural networks for accuracy. However, these models operate as black boxes, lacking transparency in how features contribute to predictions—a critical limitation for trust, debugging, and regulatory compliance (e.g., GDPR's "right to explanation"). 

We present XCTR, a unified framework for explainable CTR prediction that addresses three key challenges: (1) feature attribution at scale, (2) interaction transparency, and (3) counterfactual reasoning. XCTR introduces three novel components:

**First**, we develop a Causal Attribution Layer (CAL) that efficiently computes Shapley values for high-dimensional embeddings in O(k log n) time, enabling real-time feature importance estimation for any existing CTR model.

**Second**, we propose an Interaction Transparency Module (ITM) that visualizes which feature combinations drive predictions, extending causal attribution to second-order and higher-order interactions commonly found in deep CTR models (e.g., DCN, xDeepFM).

**Third**, we design a Counterfactual Explanation Generator (CEG) that identifies minimal feature changes required to alter predictions, providing actionable insights for advertisers and users.

We evaluate XCTR across six state-of-the-art CTR models (DeepFM, DCN, xDeepFM, etc.) on three large-scale datasets (Criteo, Avazu, KDD'12). Results demonstrate that XCTR explanations achieve 0.92 fidelity score while maintaining prediction accuracy. Through user studies with 50 advertising professionals, we show that XCTR explanations improve decision-making by 34% compared to attention-based baselines. Our framework is open-sourced as part of the FuxiCTR library.

### **Paper Structure:**

#### **1. Introduction (2 pages)**
- **1.1 Motivation**
  - Growth of CTR prediction in industry ($X billion market)
  - Black-box nature of neural CTR models
  - Regulatory requirements (GDPR Art. 22, EU AI Act)
  - Case study: Ad platform needs to explain why ads were rejected

- **1.2 Limitations of Existing Approaches**
  - Attention mechanisms ≠ explanations (Jain & Wallace, 2019)
  - LIME/SHAP too slow for production (O(2^n) complexity)
  - No prior work on causal CTR explanations

- **1.3 Our Contributions**
  1. First causal attribution framework for CTR prediction
  2. Efficient algorithms for feature and interaction importance (O(k log n))
  3. Counterfactual explanation generation for CTR
  4. Comprehensive evaluation on 6 models × 3 datasets
  5. Open-source release integrated with FuxiCTR

- **1.4 Paper Organization**

#### **2. Background and Related Work (2 pages)**
- **2.1 CTR Prediction Models**
  - Feature interactions: FM, FFM, DeepFM
  - Deep models: Wide&Deep, DCN, xDeepFM
  - Attention-based: InterHAt, AutoInt, FiBiNET

- **2.2 Explainability in Machine Learning**
  - Model-agnostic: LIME, SHAP
  - Attention as explanation
  - Counterfactual explanations

- **2.3 Gap Analysis**
  - No scalable Shapley-based CTR explanations
  - No causal framework for CTR
  - No benchmark for CTR explainability

#### **3. Problem Formulation (1 page)**
- **3.1 CTR Prediction Task**
  - Input: X = (x₁, x₂, ..., xₙ) features
  - Output: P(click | X)
  - Model: f_θ : X → [0, 1]

- **3.2 Explainability Requirements**
  - R1: Feature attribution (which features mattered?)
  - R2: Interaction transparency (which combinations mattered?)
  - R3: Counterfactual reasoning (what changes flip prediction?)
  - R4: Efficiency (real-time constraints)
  - R5: Fidelity (faithful to model behavior)

#### **4. Methodology (6 pages)**
- **4.1 Causal Attribution Layer (CAL)**
  - **4.1.1 Shapley Value Framework**
    - Definition: φᵢ = ∑_{S⊆N\{i}} [|S|!(n-|S|-1)!/n!] × [f(S∪{i}) - f(S)]
    - Interpretation: Marginal contribution of feature i
  
  - **4.1.2 Efficient Approximation Algorithm**
    - Monte Carlo sampling with stratification
    - Time complexity: O(k log n) vs O(2^n)
    - Approximation bound: |φ̂ᵢ - φᵢ| ≤ ε with probability 1-δ
  
  - **4.1.3 Integration with Embeddings**
    - Handle high-dimensional embeddings (d=16-256)
    - Aggregate attribution across embedding dimensions
    - Gradient-based acceleration

- **4.2 Interaction Transparency Module (ITM)**
  - **4.2.1 Interaction Shapley Values**
    - Define: I(i,j) = f({i,j}) - f({i}) - f({j}) + f(∅)
    - Generalize to k-way interactions
  
  - **4.2.2 Visualization Techniques**
    - Heatmap for pairwise interactions
    - Graph-based for higher-order
    - Field-level aggregation

- **4.3 Counterfactual Explanation Generator (CEG)**
  - **4.3.1 Optimization Formulation**
    - Objective: min_{x'} ||x' - x||₁ + λ₁||x' - x||₂
    - Subject to: f(x') ≥ τ (target threshold)
    - Constraints: x' ∈ feasible_range
  
  - **4.3.2 Gradient-Based Solver**
    - Projected gradient descent
    - Sparsity regularization
  
  - **4.3.3 Validity and Proximity**
    - Ensure realistic feature values
    - Minimize number of changed features

- **4.4 Framework Integration**
  - Plug-and-play design for any CTR model
  - Unified API for explanations
  - Batched processing for efficiency

#### **5. Experimental Setup (2 pages)**
- **5.1 Datasets**
  - Criteo: 45M samples, 39 features
  - Avazu: 40M samples, 23 features
  - KDD'12: 150M samples, 11 features
  - Train/valid/test splits

- **5.2 Models**
  - DeepFM, DCN, xDeepFM, AutoInt, FiBiNET, InterHAt
  - Hyperparameters from BARS benchmark

- **5.3 Baselines**
  - Attention weights (for attention models)
  - LIME
  - SHAP (standard implementation)
  - Random attribution (negative baseline)

- **5.4 Metrics**
  - **Prediction:** AUC, LogLoss
  - **Explanation Fidelity:** Correlation with true feature importance
  - **Efficiency:** Wall-clock time
  - **Counterfactual Validity:** % achieving target
  - **Counterfactual Sparsity:** Average # changed features

- **5.5 Implementation Details**
  - PyTorch 1.10, FuxiCTR v2.3
  - NVIDIA V100 GPUs
  - Code: github.com/FuatOgme/FuxiCTR/xctr

#### **6. Results (4 pages)**
- **6.1 Prediction Performance**
  - Table: XCTR maintains accuracy across all models
  - Overhead: <2% inference time

- **6.2 Explanation Fidelity**
  - Table: CAL achieves 0.92 fidelity vs 0.67 (attention), 0.73 (LIME)
  - Fig: Feature importance ranking correlation

- **6.3 Efficiency Analysis**
  - Table: CAL is 50× faster than SHAP
  - Fig: Scalability (time vs # features)

- **6.4 Interaction Discovery**
  - Fig: Heatmap of feature interactions (Criteo)
  - Case study: Age × Gender interaction for fashion ads

- **6.5 Counterfactual Quality**
  - Table: CEG validity 89%, sparsity 3.2 features
  - Fig: Distribution of counterfactual changes

- **6.6 User Study**
  - 50 advertising professionals
  - Tasks: Debug low CTR campaign, optimize budget
  - Result: 34% improvement with XCTR vs baselines

- **6.7 Ablation Studies**
  - Remove CAL / ITM / CEG → performance drops
  - Vary number of Shapley samples

#### **7. Discussion (2 pages)**
- **7.1 Key Findings**
  - Causal attribution is feasible and valuable for CTR
  - Interactions matter: 40% of important patterns are 2nd-order

- **7.2 Practical Applications**
  - Ad creative optimization
  - User segment analysis
  - Model debugging and bias detection

- **7.3 Limitations**
  - Assumes feature independence (may not hold)
  - Computational cost still non-trivial
  - Human evaluation needed for true utility

- **7.4 Ethical Considerations**
  - Transparency can reveal biases
  - Privacy: Don't expose individual user data

#### **8. Conclusion (0.5 page)**
- Summary of contributions
- Impact: First causal CTR explanation framework
- Future work: Extend to multi-task CTR, real-time systems

#### **References (2 pages)**
~50 references

---

## 📄 Paper 2: Interaction Visualization (Direction 1, Extension)

### **Title:**
"Visualizing the 'Why': Interactive Exploration of Feature Interactions in Deep CTR Models"

### **Target Venue:** WWW (Web Conference) or CHI (Human-Computer Interaction)

### **Key Contribution:**
Interactive visualization toolkit for exploring CTR model decisions.

### **Structure (8 pages):**
1. Introduction (1.5 pages)
2. Related Work: Visualization for ML (1 page)
3. Design Requirements from User Study (1 page)
4. Visualization Framework (2 pages)
5. Case Studies (2 pages)
6. User Evaluation (1 page)
7. Conclusion (0.5 page)

---

## 📄 Paper 3: Federated CTR (Direction 2)

### **Title:**
"FedCTR: Privacy-Preserving Click-Through Rate Prediction via Federated Learning"

### **Abstract:**

Click-through rate (CTR) prediction increasingly faces privacy constraints due to regulations (GDPR, CCPA) and platform policies (iOS ATT, cookie deprecation). Centralized data collection is no longer tenable, yet existing federated learning (FL) approaches fail to address CTR-specific challenges: sparse categorical features, embedding tables with billions of parameters, and real-time inference requirements.

We present FedCTR, the first federated learning framework designed for large-scale CTR prediction. FedCTR addresses three key challenges:

**First**, we develop Federated Embedding Alignment (FEA) to handle heterogeneous feature vocabularies across clients (e.g., different user demographics, geographies). FEA uses secure hashing and common embedding spaces to enable cross-client learning without sharing raw vocabularies.

**Second**, we propose Differential Privacy for CTR (DP-CTR), a novel privacy mechanism that adds calibrated noise to gradient updates while preserving feature interaction learning. We prove that DP-CTR satisfies (ε, δ)-differential privacy with ε=1.0 achieving <3% accuracy loss.

**Third**, we design a Split Learning variant for sequence-based CTR models (DIN, DIEN, BST), where clients keep user behavior sequences private while the server handles item embeddings.

We evaluate FedCTR on six CTR models across three federated scenarios: (1) cross-geography (10 regions), (2) cross-app (5 mobile apps), and (3) cross-advertiser (20 brands). Results show FedCTR achieves 97.2% of centralized model accuracy while providing strong privacy guarantees (ε=1.0). In production deployment at [Partner Company], FedCTR serves 10M+ requests/day with 45ms p99 latency.

### **Paper Structure:**

#### **1. Introduction**
- Privacy crisis in digital advertising
- Limitations of centralized CTR prediction
- Challenges of federated CTR (vs. standard FL)

#### **2. Background**
- **2.1** Federated Learning Basics (FedAvg, FedProx)
- **2.2** CTR Model Architecture (embeddings, interactions)
- **2.3** Privacy Definitions (DP, secure aggregation)

#### **3. Problem Formulation**
- Federated CTR setup (N clients, 1 server)
- Privacy requirements
- Accuracy-privacy-efficiency tradeoff

#### **4. Methodology**
- **4.1** Federated Embedding Alignment (FEA)
  - Challenge: Client A has features {f1, f2}, Client B has {f2, f3}
  - Solution: Common embedding space + local projection
  - Algorithm: Hash-based feature alignment

- **4.2** Differential Privacy for CTR (DP-CTR)
  - Adaptive noise calibration for sparse gradients
  - Privacy budget allocation across layers
  - Proof of privacy guarantee

- **4.3** Split Learning for Sequence Models
  - Client: User sequence encoder
  - Server: Item encoder + interaction layer
  - Secure aggregation protocol

- **4.4** Communication Efficiency**
  - Gradient compression (top-k, quantization)
  - Asynchronous updates

#### **5. Experiments**
- **5.1** Datasets: Criteo (geo-partitioned), simulated federated
- **5.2** Models: DeepFM, DCN, DIN, DIEN
- **5.3** Baselines: Centralized, local-only, FedAvg, DP-SGD
- **5.4** Metrics: AUC, privacy budget, communication cost, latency

#### **6. Results**
- Accuracy vs privacy tradeoff curves
- Communication efficiency
- Production deployment results

#### **7. Discussion**
- When is federated CTR worth it?
- Practical deployment lessons

#### **8. Conclusion**

---

## 📄 Paper 4: Causal CTR (Direction 8)

### **Title:**
"From Correlation to Causation: Intervention-Based Click-Through Rate Prediction for Optimal Ad Strategy"

### **Abstract:**

Current CTR prediction models answer: "What is P(click | ad, user)?" However, advertisers and platforms need causal questions: "What is the effect of showing ad A vs. ad B?" and "Which users should we target to maximize ROI?" Observational data suffers from confounding (users who click were already interested) and selection bias (ads shown based on predicted CTR).

We present CausalCTR, a framework for causal inference in click-through rate prediction. CausalCTR addresses three fundamental questions:

**Q1: Average Treatment Effect (ATE)**: What is the causal effect of showing an ad?
**Q2: Conditional ATE (CATE)**: Which users benefit most from seeing an ad?
**Q3: Optimal Policy**: How should we allocate ad budget causally?

CausalCTR introduces:

**First**, a Doubly Robust (DR) estimator that combines propensity score weighting and outcome regression, achieving unbiased CTR estimates even with model misspecification.

**Second**, a Causal Feature Interaction module that identifies which feature combinations causally drive clicks (not just correlate).

**Third**, a Policy Optimization algorithm that learns optimal ad assignment using causal effect estimates, maximizing ROI under budget constraints.

We evaluate CausalCTR on three datasets with ground-truth causal effects (semi-synthetic Criteo, real A/B tests from [Partner]). CausalCTR reduces policy regret by 18.3% compared to correlation-based targeting and achieves 22.4% higher ROI. In live experiments, CausalCTR increased conversion rate by 12.7% while maintaining ad spend.

### **Paper Structure:**

#### **1. Introduction**
- **1.1** Motivating Example
  - Advertiser runs campaign, CTR model says show ad to Group A
  - But: Group A already likely to convert (confounding!)
  - Better: Show ad to Group B (higher causal effect)

- **1.2** Limitations of Correlational CTR
  - Doesn't answer "what if?"
  - Can't guide interventions
  - Ignores confounding

- **1.3** Our Approach: Causal CTR**
  - Estimate treatment effects
  - Learn optimal policies
  - Debiased from observational data

#### **2. Background and Related Work**
- **2.1** Causal Inference
  - Potential outcomes framework
  - Randomized experiments vs. observational studies
  - Confounding and selection bias

- **2.2** Treatment Effect Estimation
  - Propensity scores (Rosenbaum & Rubin, 1983)
  - Inverse propensity weighting (Horvitz & Thompson, 1952)
  - Doubly robust estimation (Bang & Robins, 2005)

- **2.3** Causal ML**
  - Causal forests (Wager & Athey, 2018)
  - Deep causal models (Shalit et al., 2017)

- **2.4** Debiasing in RecSys**
  - Position bias (Joachims et al., 2017)
  - Selection bias (Schnabel et al., 2016)

- **2.5** Gap: No Causal CTR Framework**

#### **3. Problem Formulation**
- **3.1** Causal Graph**
  - Nodes: User features (U), Ad features (A), Treatment (T), Outcome (Y)
  - Edges: U → T, A → T, U → Y, A → Y, T → Y
  - Confounders: U, A

- **3.2** Causal Estimands**
  - ATE: E[Y(T=1)] - E[Y(T=0)]
  - CATE: E[Y(T=1) | X=x] - E[Y(T=0) | X=x]
  - ATT: E[Y(T=1) - Y(T=0) | T=1]

- **3.3** Assumptions**
  - SUTVA (no interference)
  - Unconfoundedness (all confounders observed)
  - Positivity (0 < P(T=1|X) < 1)

#### **4. Methodology**
- **4.1** Doubly Robust CTR Estimator**
  - Propensity network: π(x) = P(T=1 | X=x)
  - Outcome networks: μ₁(x) = E[Y | X=x, T=1], μ₀(x) = E[Y | X=x, T=0]
  - DR estimator: τ̂(x) = μ₁(x) - μ₀(x) + T(Y - μ₁(x))/π(x) - (1-T)(Y - μ₀(x))/(1-π(x))
  - Proof: Unbiased if either propensity or outcome model correct

- **4.2** Causal Feature Interactions**
  - Extend to: I(i,j) = causal effect of feature pair (i,j)
  - Use structural causal models (Pearl, 2009)

- **4.3** Policy Learning**
  - Goal: π*(x) = argmax_{a∈A} E[Y(T=a) | X=x]
  - Algorithm: Inverse propensity weighted policy learning
  - Regret bound: O(1/√n)

- **4.4** Integration with Deep CTR Models**
  - Use existing architectures (DeepFM, DCN) as outcome networks
  - Plug-and-play causal layer

#### **5. Experiments**
- **5.1** Datasets**
  - Semi-synthetic Criteo (inject known confounding)
  - Real A/B test data from [Partner]
  - Simulated ad auction

- **5.2** Evaluation**
  - Ground truth ATE (from randomized data)
  - Policy value (expected reward)
  - Regret (vs. optimal policy)

- **5.3** Baselines**
  - Correlational CTR (standard models)
  - IPW (inverse propensity weighting only)
  - T-learner (separate models for T=0, T=1)
  - Causal forests

#### **6. Results**
- **6.1** Treatment Effect Estimation Accuracy**
  - Table: CausalCTR ATE error 0.012 vs 0.045 (correlational)
  - Fig: Estimated vs true CATE

- **6.2** Policy Performance**
  - Table: CausalCTR policy value 18.3% higher
  - Fig: ROI curves under different budgets

- **6.3** Feature Interaction Analysis**
  - Case study: Age × Product_Category causal interaction

- **6.4** Ablation Studies**
  - Remove propensity network → performance drops
  - Remove outcome network → performance drops
  - DR > IPW > outcome regression alone

#### **7. Discussion**
- **7.1** When Does Causality Help?**
  - High confounding scenarios
  - Heterogeneous treatment effects
  - Policy optimization

- **7.2** Practical Deployment**
  - Need randomized data for validation
  - Sensitivity analysis for unobserved confounding

- **7.3** Limitations**
  - Assumes no hidden confounders
  - Requires more complex models

#### **8. Conclusion**
- Shift from P(Y|X) to P(Y|do(X))
- 18-22% improvement in real systems
- Future: Multi-armed bandits with causal priors

---

## 📋 Publication Strategy

### **Year 1:**
- **Paper 1** (Explainable CTR) → Submit to **KDD 2026** (Feb deadline)
  - Expected outcome: Accept (strong novelty + open-source)

### **Year 2:**
- **Paper 2** (Visualization) → Submit to **WWW 2027** (Oct deadline)
- **Paper 3** (Federated CTR) → Submit to **NeurIPS 2027** (May deadline)

### **Year 3:**
- **Paper 4** (Causal CTR) → Submit to **ICML 2028** (Jan deadline)
- **Journal Paper** (Unified framework) → Submit to **TKDE** (rolling)

### **Year 4:**
- Workshop papers, demos
- Thesis writing

---

## 📊 Expected Impact

### **Citations (4-year projection):**
- Paper 1: 80-100 citations (explainability is hot)
- Paper 2: 30-50 citations (niche but valuable)
- Paper 3: 60-80 citations (privacy is critical)
- Paper 4: 70-90 citations (causal ML growing fast)
- **Total: 240-320 citations by defense**

### **GitHub Stars:**
- FuxiCTR already has ~1.5K stars
- XCTR module: +500 stars (similar to SHAP's popularity)
- FedCTR module: +300 stars
- CausalCTR module: +200 stars

### **Industry Adoption:**
- At least 2 companies deploy XCTR (ad platforms need explanations)
- Federated CTR adopted by privacy-focused advertisers
- Causal CTR used for budget optimization

---

## 🎓 Thesis Integration

### **Thesis Title:**
"Towards Trustworthy and Causal Click-Through Rate Prediction: Explainability, Privacy, and Intervention"

### **Thesis Structure (6 chapters):**

1. **Introduction** (30 pages)
   - Motivation, problem statement, contributions

2. **Background** (40 pages)
   - CTR models survey
   - Explainability, privacy, causality background

3. **Explainable CTR** (60 pages)
   - Papers 1 & 2 content

4. **Federated CTR** (50 pages)
   - Paper 3 content

5. **Causal CTR** (60 pages)
   - Paper 4 content

6. **Conclusion and Future Work** (20 pages)
   - Summary, limitations, future directions

**Total: ~260 pages**

---

## 🚀 Next Steps

1. **Choose your primary direction** (1, 2, or 8)
2. **Write 2-page research proposal** (this week)
3. **Implement prototype** (1 month)
4. **Run preliminary experiments** (1 month)
5. **Draft Paper 1** (2 months)
6. **Submit to KDD/SIGIR** (by deadline)

**I can help with any of these steps!**

Let me know which paper you want to start with, and I'll provide:
- Detailed outline (section-by-section)
- LaTeX template
- Experiment design
- Baseline implementations
- Writing tips

Good luck! 🎓
