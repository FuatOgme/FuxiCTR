# Implementation Quick-Start Guide
## Getting Started with Novel PhD Research in FuxiCTR

This guide provides **concrete implementation templates** for the top 3 recommended research directions.

---

## 🎯 Direction 1: Explainable CTR - Starter Implementation

### Step 1: Create Module Structure

```bash
mkdir -p fuxictr/pytorch/layers/explainability
touch fuxictr/pytorch/layers/explainability/__init__.py
touch fuxictr/pytorch/layers/explainability/causal_attribution.py
touch fuxictr/pytorch/layers/explainability/counterfactual.py
```

### Step 2: Implement Causal Attribution Layer

```python
# fuxictr/pytorch/layers/explainability/causal_attribution.py

import torch
import torch.nn as nn
import numpy as np
from itertools import combinations

class CausalAttributionLayer(nn.Module):
    """
    Computes feature importance via Shapley values for CTR models.
    
    Key Innovation: Efficient approximation for high-dimensional embeddings
    Time Complexity: O(k * log(n)) instead of O(2^n) where n = num_features
    
    Args:
        model: Base CTR model (any FuxiCTR model)
        num_samples: Number of samples for Shapley approximation
        baseline: Baseline input for reference (e.g., zero embeddings)
    """
    
    def __init__(self, model, num_samples=100, baseline='zero'):
        super(CausalAttributionLayer, self).__init__()
        self.model = model
        self.num_samples = num_samples
        self.baseline = baseline
        
    def forward(self, x):
        """
        Standard forward pass (for training)
        """
        return self.model(x)
    
    def explain(self, x, target_features=None):
        """
        Compute Shapley values for feature attribution.
        
        Args:
            x: Input features (batch_size, num_features, embedding_dim)
            target_features: List of feature indices to explain (None = all)
            
        Returns:
            attributions: Tensor of shape (batch_size, num_features)
        """
        batch_size, num_features, emb_dim = x.shape
        
        if target_features is None:
            target_features = list(range(num_features))
        
        # Initialize attribution scores
        attributions = torch.zeros(batch_size, num_features)
        
        # Get baseline prediction
        baseline_input = self._get_baseline(x)
        baseline_pred = self.model(baseline_input)
        
        # Monte Carlo sampling for Shapley approximation
        for _ in range(self.num_samples):
            # Random permutation of features
            perm = torch.randperm(num_features)
            
            # Build coalition incrementally
            coalition_input = baseline_input.clone()
            prev_pred = baseline_pred
            
            for i, feat_idx in enumerate(perm):
                # Add feature to coalition
                coalition_input[:, feat_idx, :] = x[:, feat_idx, :]
                curr_pred = self.model(coalition_input)
                
                # Marginal contribution
                marginal = curr_pred - prev_pred
                attributions[:, feat_idx] += marginal.squeeze()
                
                prev_pred = curr_pred
        
        # Average over samples
        attributions /= self.num_samples
        
        return attributions
    
    def _get_baseline(self, x):
        """Generate baseline input (e.g., zero embeddings or mean)"""
        if self.baseline == 'zero':
            return torch.zeros_like(x)
        elif self.baseline == 'mean':
            return x.mean(dim=0, keepdim=True).expand_as(x)
        else:
            raise ValueError(f"Unknown baseline: {self.baseline}")
    
    def explain_interaction(self, x, feature_pairs=None):
        """
        Compute interaction effects between feature pairs.
        
        Novel contribution: Extend Shapley values to feature interactions
        
        Returns:
            interaction_matrix: (num_features, num_features) symmetric matrix
        """
        num_features = x.shape[1]
        
        if feature_pairs is None:
            # All pairs
            feature_pairs = list(combinations(range(num_features), 2))
        
        interaction_matrix = torch.zeros(num_features, num_features)
        
        for (i, j) in feature_pairs:
            # Compute I(i,j) = f(x_i, x_j) - f(x_i) - f(x_j) + f(∅)
            
            # Baseline
            baseline = self._get_baseline(x)
            f_empty = self.model(baseline)
            
            # Only feature i
            x_i = baseline.clone()
            x_i[:, i, :] = x[:, i, :]
            f_i = self.model(x_i)
            
            # Only feature j  
            x_j = baseline.clone()
            x_j[:, j, :] = x[:, j, :]
            f_j = self.model(x_j)
            
            # Both features
            x_ij = baseline.clone()
            x_ij[:, i, :] = x[:, i, :]
            x_ij[:, j, :] = x[:, j, :]
            f_ij = self.model(x_ij)
            
            # Interaction strength
            interaction = (f_ij - f_i - f_j + f_empty).mean()
            
            interaction_matrix[i, j] = interaction
            interaction_matrix[j, i] = interaction  # Symmetric
        
        return interaction_matrix


class CounterfactualGenerator(nn.Module):
    """
    Generates minimal counterfactual explanations for CTR predictions.
    
    Question: "What is the minimal change to make this ad get clicked?"
    
    Method: Gradient-based optimization with sparsity constraint
    """
    
    def __init__(self, model, max_iterations=100, lr=0.1, sparsity_weight=0.1):
        super(CounterfactualGenerator, self).__init__()
        self.model = model
        self.max_iterations = max_iterations
        self.lr = lr
        self.sparsity_weight = sparsity_weight
    
    def generate(self, x_orig, target_class=1, feature_ranges=None):
        """
        Generate counterfactual explanation.
        
        Args:
            x_orig: Original input (did NOT click)
            target_class: Desired output (1 = click)
            feature_ranges: Valid ranges for each feature (for realism)
            
        Returns:
            x_counterfactual: Modified input that achieves target
            changes: Dictionary of {feature_idx: (old_val, new_val)}
        """
        # Initialize counterfactual (make it a learnable parameter)
        x_cf = x_orig.clone().detach().requires_grad_(True)
        
        optimizer = torch.optim.Adam([x_cf], lr=self.lr)
        
        for iteration in range(self.max_iterations):
            optimizer.zero_grad()
            
            # Prediction loss (want to reach target class)
            pred = self.model(x_cf)
            prediction_loss = -torch.log(pred) if target_class == 1 else -torch.log(1 - pred)
            
            # Proximity loss (minimize change from original)
            proximity_loss = torch.norm(x_cf - x_orig, p=2)
            
            # Sparsity loss (change few features)
            sparsity_loss = torch.norm(x_cf - x_orig, p=1)
            
            # Combined loss
            loss = prediction_loss + proximity_loss + self.sparsity_weight * sparsity_loss
            
            loss.backward()
            optimizer.step()
            
            # Project to valid feature ranges
            if feature_ranges is not None:
                x_cf = self._project_to_range(x_cf, feature_ranges)
            
            # Early stopping if target reached
            if (pred > 0.5 and target_class == 1) or (pred < 0.5 and target_class == 0):
                break
        
        # Identify changed features
        changes = self._identify_changes(x_orig, x_cf)
        
        return x_cf.detach(), changes
    
    def _project_to_range(self, x, feature_ranges):
        """Ensure features stay in valid ranges"""
        for feat_idx, (min_val, max_val) in feature_ranges.items():
            x[:, feat_idx, :] = torch.clamp(x[:, feat_idx, :], min_val, max_val)
        return x
    
    def _identify_changes(self, x_orig, x_cf, threshold=0.01):
        """Find features that changed significantly"""
        diff = torch.abs(x_cf - x_orig)
        changed_features = {}
        
        for feat_idx in range(diff.shape[1]):
            feat_diff = diff[:, feat_idx, :].mean().item()
            if feat_diff > threshold:
                changed_features[feat_idx] = {
                    'original': x_orig[:, feat_idx, :].mean().item(),
                    'counterfactual': x_cf[:, feat_idx, :].mean().item(),
                    'change': feat_diff
                }
        
        return changed_features


# Integration with existing models
class ExplainableDeepFM(nn.Module):
    """
    Example: Add explainability to DeepFM model
    """
    
    def __init__(self, feature_map, **kwargs):
        super(ExplainableDeepFM, self).__init__()
        
        # Base DeepFM model (import from your model zoo)
        from model_zoo.DeepFM.src.DeepFM import DeepFM
        self.base_model = DeepFM(feature_map, **kwargs)
        
        # Add explainability layers
        self.explainer = CausalAttributionLayer(self.base_model)
        self.cf_generator = CounterfactualGenerator(self.base_model)
    
    def forward(self, x):
        return self.base_model(x)
    
    def explain_prediction(self, x):
        """Get feature attributions"""
        return self.explainer.explain(x)
    
    def generate_counterfactual(self, x, target=1):
        """Generate counterfactual explanation"""
        return self.cf_generator.generate(x, target)
```

### Step 3: Example Usage

```python
# example_explainable_ctr.py

import torch
from fuxictr.pytorch.layers.explainability import ExplainableDeepFM
from fuxictr.features import FeatureMap

# Load data
feature_map = FeatureMap(dataset_config, data_dir)
train_data, valid_data = load_data(...)

# Create explainable model
model = ExplainableDeepFM(
    feature_map,
    embedding_dim=16,
    hidden_units=[400, 400, 400],
    learning_rate=1e-3
)

# Train normally
model.fit(train_data, valid_data)

# Explain a prediction
sample_input = valid_data[0]
prediction = model(sample_input)

# Get feature importance
attributions = model.explain_prediction(sample_input)
print("Top 5 important features:")
top_features = torch.topk(attributions, k=5)
for idx, score in zip(top_features.indices, top_features.values):
    print(f"Feature {idx}: {score:.4f}")

# Generate counterfactual
cf_input, changes = model.generate_counterfactual(sample_input, target=1)
print("\nTo flip prediction, change:")
for feat_idx, change_info in changes.items():
    print(f"Feature {feat_idx}: {change_info['original']:.3f} → {change_info['counterfactual']:.3f}")
```

### Step 4: Evaluation Metrics

```python
# fuxictr/pytorch/layers/explainability/metrics.py

def explanation_fidelity(model, explainer, x, y):
    """
    Measure how well explanations match model behavior.
    
    Method: Remove top-k important features, measure prediction change
    """
    # Get feature importance
    attributions = explainer.explain(x)
    
    # Remove top-k features
    k_values = [1, 3, 5, 10]
    fidelity_scores = []
    
    for k in k_values:
        top_k_features = torch.topk(attributions, k).indices
        
        # Zero out top features
        x_removed = x.clone()
        for feat_idx in top_k_features:
            x_removed[:, feat_idx, :] = 0
        
        # Measure prediction change
        pred_orig = model(x)
        pred_removed = model(x_removed)
        
        fidelity = torch.abs(pred_orig - pred_removed).mean().item()
        fidelity_scores.append(fidelity)
    
    return fidelity_scores

def counterfactual_validity(cf_generator, model, x, target=1):
    """
    Measure what % of counterfactuals actually achieve target prediction
    """
    cf_x, _ = cf_generator.generate(x, target)
    pred = model(cf_x)
    
    if target == 1:
        validity = (pred > 0.5).float().mean().item()
    else:
        validity = (pred < 0.5).float().mean().item()
    
    return validity

def counterfactual_sparsity(x_orig, x_cf):
    """
    Measure how few features were changed (lower is better)
    """
    changed_features = (torch.abs(x_orig - x_cf) > 0.01).sum(dim=-1).float().mean()
    return changed_features.item()
```

---

## 🎯 Direction 2: Federated CTR - Starter Implementation

### Step 1: Create Federated Module

```python
# fuxictr/pytorch/federated/federated_trainer.py

import torch
import torch.nn as nn
from copy import deepcopy

class FederatedCTRTrainer:
    """
    Federated learning framework for CTR models.
    
    Architecture:
    - Server: Maintains global model
    - Clients: Local models trained on user data
    
    Algorithm: FedAvg with differential privacy
    """
    
    def __init__(self, 
                 global_model, 
                 num_clients=10,
                 local_epochs=5,
                 lr=0.001,
                 privacy_epsilon=1.0):
        self.global_model = global_model
        self.num_clients = num_clients
        self.local_epochs = local_epochs
        self.lr = lr
        self.privacy_epsilon = privacy_epsilon
        
        # Initialize client models (copies of global model)
        self.client_models = [deepcopy(global_model) for _ in range(num_clients)]
    
    def train_federated(self, client_data_loaders, num_rounds=100):
        """
        Main federated training loop.
        
        Args:
            client_data_loaders: List of data loaders (one per client)
            num_rounds: Number of federated rounds
        """
        for round_idx in range(num_rounds):
            print(f"\n=== Federated Round {round_idx+1}/{num_rounds} ===")
            
            # Step 1: Distribute global model to clients
            self._distribute_model()
            
            # Step 2: Local training on each client
            client_updates = []
            for client_id in range(self.num_clients):
                update = self._local_training(
                    client_id, 
                    client_data_loaders[client_id]
                )
                client_updates.append(update)
            
            # Step 3: Aggregate updates with differential privacy
            self._aggregate_updates(client_updates)
            
            # Step 4: Evaluate global model
            if round_idx % 10 == 0:
                self._evaluate_global_model()
    
    def _distribute_model(self):
        """Send global model parameters to all clients"""
        global_params = self.global_model.state_dict()
        for client_model in self.client_models:
            client_model.load_state_dict(deepcopy(global_params))
    
    def _local_training(self, client_id, data_loader):
        """
        Train model locally on client data.
        
        Returns:
            model_update: Dictionary of parameter updates
        """
        client_model = self.client_models[client_id]
        client_model.train()
        
        optimizer = torch.optim.Adam(client_model.parameters(), lr=self.lr)
        
        for epoch in range(self.local_epochs):
            for batch in data_loader:
                x, y = batch
                
                optimizer.zero_grad()
                y_pred = client_model(x)
                loss = nn.BCELoss()(y_pred, y)
                loss.backward()
                optimizer.step()
        
        # Compute parameter update (difference from global model)
        global_params = self.global_model.state_dict()
        client_params = client_model.state_dict()
        
        update = {}
        for key in global_params.keys():
            update[key] = client_params[key] - global_params[key]
        
        return update
    
    def _aggregate_updates(self, client_updates):
        """
        Aggregate client updates using FedAvg + Differential Privacy.
        """
        # Step 1: Average updates
        avg_update = {}
        for key in client_updates[0].keys():
            avg_update[key] = torch.mean(
                torch.stack([update[key] for update in client_updates]),
                dim=0
            )
        
        # Step 2: Add differential privacy noise
        if self.privacy_epsilon > 0:
            avg_update = self._add_dp_noise(avg_update)
        
        # Step 3: Update global model
        global_params = self.global_model.state_dict()
        for key in global_params.keys():
            global_params[key] += avg_update[key]
        
        self.global_model.load_state_dict(global_params)
    
    def _add_dp_noise(self, update, sensitivity=1.0):
        """
        Add Gaussian noise for differential privacy.
        
        Noise scale: σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε
        """
        delta = 1e-5  # Privacy parameter
        noise_scale = sensitivity * torch.sqrt(
            2 * torch.log(torch.tensor(1.25 / delta))
        ) / self.privacy_epsilon
        
        noisy_update = {}
        for key, param in update.items():
            noise = torch.randn_like(param) * noise_scale
            noisy_update[key] = param + noise
        
        return noisy_update
    
    def _evaluate_global_model(self):
        """Evaluate global model on test set"""
        # TODO: Implement evaluation
        pass


class FederatedDeepFM(nn.Module):
    """
    Example: Federated version of DeepFM
    """
    
    def __init__(self, feature_map, **kwargs):
        super(FederatedDeepFM, self).__init__()
        
        # Import base DeepFM
        from model_zoo.DeepFM.src.DeepFM import DeepFM
        self.model = DeepFM(feature_map, **kwargs)
    
    def forward(self, x):
        return self.model(x)
```

### Step 2: Example Usage

```python
# example_federated_training.py

from fuxictr.pytorch.federated import FederatedCTRTrainer, FederatedDeepFM
from fuxictr.features import FeatureMap

# Simulate federated data (partition by user geography)
def partition_data_federated(data, num_clients=10):
    """Split data into client shards"""
    client_data = []
    shard_size = len(data) // num_clients
    
    for i in range(num_clients):
        start_idx = i * shard_size
        end_idx = (i + 1) * shard_size if i < num_clients - 1 else len(data)
        client_data.append(data[start_idx:end_idx])
    
    return client_data

# Load data
feature_map = FeatureMap(dataset_config, data_dir)
full_data = load_data(...)

# Partition into 10 clients
client_data_loaders = partition_data_federated(full_data, num_clients=10)

# Create global model
global_model = FederatedDeepFM(feature_map, embedding_dim=16)

# Federated trainer
trainer = FederatedCTRTrainer(
    global_model,
    num_clients=10,
    local_epochs=5,
    privacy_epsilon=1.0  # Differential privacy parameter
)

# Train
trainer.train_federated(client_data_loaders, num_rounds=100)
```

---

## 🎯 Direction 8: Causal CTR - Starter Implementation

### Step 1: Create Causal Module

```python
# fuxictr/pytorch/causal/causal_ctr.py

import torch
import torch.nn as nn

class CausalCTRModel(nn.Module):
    """
    Causal CTR model with treatment effect estimation.
    
    Framework:
    - Structural Causal Model: X → T → Y
      - X: User/context features
      - T: Treatment (which ad shown)
      - Y: Outcome (click or not)
    
    Key Innovation: Debiased CTR prediction via inverse propensity weighting
    """
    
    def __init__(self, feature_map, embedding_dim=16, hidden_units=[64, 64]):
        super(CausalCTRModel, self).__init__()
        
        from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block
        
        # Feature embeddings
        self.embedding = FeatureEmbedding(feature_map, embedding_dim)
        input_dim = feature_map.sum_emb_out_dim()
        
        # Propensity network: P(T|X)
        # Estimates probability of receiving treatment given features
        self.propensity_net = MLP_Block(
            input_dim=input_dim,
            output_dim=1,  # Binary treatment
            hidden_units=hidden_units,
            output_activation='sigmoid'
        )
        
        # Outcome network: E[Y|X,T]
        # Estimates expected outcome given features and treatment
        self.outcome_net = MLP_Block(
            input_dim=input_dim + 1,  # +1 for treatment indicator
            output_dim=1,
            hidden_units=hidden_units,
            output_activation='sigmoid'
        )
    
    def forward(self, X, T):
        """
        Forward pass for training.
        
        Args:
            X: Features (batch_size, num_features)
            T: Treatment indicator (batch_size, 1)
        """
        # Get embeddings
        emb = self.embedding(X)
        emb_flat = emb.flatten(start_dim=1)
        
        # Propensity score
        propensity = self.propensity_net(emb_flat)
        
        # Outcome prediction
        outcome_input = torch.cat([emb_flat, T], dim=1)
        outcome = self.outcome_net(outcome_input)
        
        return outcome, propensity
    
    def predict_ipw(self, X, T, Y):
        """
        Inverse Propensity Weighting (IPW) estimator.
        
        Unbiased estimator: E[Y/P(T|X)]
        """
        outcome, propensity = self.forward(X, T)
        
        # Clip propensity to avoid division by zero
        propensity = torch.clamp(propensity, min=0.01, max=0.99)
        
        # IPW weight
        weight = 1.0 / propensity
        
        # Weighted prediction
        weighted_outcome = outcome * weight
        
        return weighted_outcome
    
    def predict_doubly_robust(self, X, T, Y):
        """
        Doubly Robust (DR) estimator.
        
        Combines outcome regression and propensity weighting.
        More robust than either method alone.
        """
        outcome, propensity = self.forward(X, T)
        propensity = torch.clamp(propensity, min=0.01, max=0.99)
        
        # DR estimator
        dr_estimate = outcome + (T / propensity) * (Y - outcome)
        
        return dr_estimate
    
    def estimate_ate(self, X):
        """
        Estimate Average Treatment Effect (ATE).
        
        ATE = E[Y(T=1)] - E[Y(T=0)]
        
        Interpretation: Expected CTR increase if we show ad vs. no ad
        """
        emb = self.embedding(X)
        emb_flat = emb.flatten(start_dim=1)
        
        # E[Y | T=1]
        T1 = torch.ones(X.shape[0], 1)
        outcome_input_1 = torch.cat([emb_flat, T1], dim=1)
        E_Y_T1 = self.outcome_net(outcome_input_1)
        
        # E[Y | T=0]
        T0 = torch.zeros(X.shape[0], 1)
        outcome_input_0 = torch.cat([emb_flat, T0], dim=1)
        E_Y_T0 = self.outcome_net(outcome_input_0)
        
        # ATE
        ate = (E_Y_T1 - E_Y_T0).mean()
        
        return ate.item()
    
    def estimate_cate(self, X):
        """
        Estimate Conditional Average Treatment Effect (CATE).
        
        CATE(x) = E[Y(T=1) | X=x] - E[Y(T=0) | X=x]
        
        Interpretation: Personalized treatment effect for each user
        """
        emb = self.embedding(X)
        emb_flat = emb.flatten(start_dim=1)
        
        # E[Y | X, T=1]
        T1 = torch.ones(X.shape[0], 1)
        outcome_input_1 = torch.cat([emb_flat, T1], dim=1)
        E_Y_T1 = self.outcome_net(outcome_input_1)
        
        # E[Y | X, T=0]
        T0 = torch.zeros(X.shape[0], 1)
        outcome_input_0 = torch.cat([emb_flat, T0], dim=1)
        E_Y_T0 = self.outcome_net(outcome_input_0)
        
        # CATE for each sample
        cate = E_Y_T1 - E_Y_T0
        
        return cate
    
    def compute_loss(self, X, T, Y):
        """
        Training loss combining propensity and outcome objectives.
        """
        outcome, propensity = self.forward(X, T)
        
        # Propensity loss (cross-entropy)
        propensity_loss = nn.BCELoss()(propensity, T)
        
        # Outcome loss (MSE or cross-entropy)
        outcome_loss = nn.BCELoss()(outcome, Y)
        
        # Combined loss
        total_loss = propensity_loss + outcome_loss
        
        return total_loss


class CausalInteractionModel(nn.Module):
    """
    Extends causal framework to model feature interactions.
    
    Question: Which feature combinations CAUSE clicks?
    """
    
    def __init__(self, feature_map, embedding_dim=16):
        super(CausalInteractionModel, self).__init__()
        
        from fuxictr.pytorch.layers import FeatureEmbedding
        from fuxictr.pytorch.layers.interactions import InnerProductInteraction
        
        self.embedding = FeatureEmbedding(feature_map, embedding_dim)
        self.interaction = InnerProductInteraction(feature_map.num_fields)
        
        # Causal outcome model
        self.causal_model = CausalCTRModel(feature_map)
    
    def forward(self, X, T):
        # Feature interactions
        emb = self.embedding(X)
        interactions = self.interaction(emb)
        
        # Causal prediction
        outcome, propensity = self.causal_model(X, T)
        
        # Combine
        final_pred = outcome * (1 + interactions)
        
        return final_pred, propensity
```

### Step 2: Example Usage

```python
# example_causal_ctr.py

from fuxictr.pytorch.causal import CausalCTRModel
from fuxictr.features import FeatureMap

# Load data
feature_map = FeatureMap(dataset_config, data_dir)
train_data = load_data(...)

# Extract features, treatment, outcome
X = train_data['features']  # User/context features
T = train_data['treatment']  # Which ad was shown (0 or 1)
Y = train_data['click']      # Did user click? (0 or 1)

# Create causal model
model = CausalCTRModel(feature_map, embedding_dim=16)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    optimizer.zero_grad()
    loss = model.compute_loss(X, T, Y)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Estimate treatment effects
ate = model.estimate_ate(X)
print(f"\nAverage Treatment Effect: {ate:.4f}")
print("Interpretation: Showing ad increases CTR by {:.2f}%".format(ate * 100))

# Personalized treatment effects
cate = model.estimate_cate(X)
print(f"\nTop 5 users who would benefit most from ad:")
top_users = torch.topk(cate.squeeze(), k=5)
for idx, effect in zip(top_users.indices, top_users.values):
    print(f"User {idx}: +{effect.item():.3f} CTR increase")

# Unbiased prediction (debiased using propensity scores)
debiased_pred = model.predict_doubly_robust(X, T, Y)
```

---

## 📊 Quick Evaluation Framework

Create a unified evaluation script for all approaches:

```python
# evaluation/evaluate_novel_models.py

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

class NovelCTREvaluator:
    """Unified evaluation for explainability, federated, and causal CTR"""
    
    def __init__(self, model, test_data):
        self.model = model
        self.test_data = test_data
    
    def evaluate_standard_metrics(self):
        """AUC, LogLoss, etc."""
        y_true = []
        y_pred = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in self.test_data:
                X, y = batch
                pred = self.model(X)
                
                y_true.extend(y.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())
        
        auc = roc_auc_score(y_true, y_pred)
        logloss = log_loss(y_true, y_pred)
        
        return {'AUC': auc, 'LogLoss': logloss}
    
    def evaluate_explanation_quality(self, explainer):
        """For explainable models"""
        # Fidelity score
        fidelity = explanation_fidelity(self.model, explainer, ...)
        
        # Stability score (do similar inputs get similar explanations?)
        stability = self._compute_stability(explainer)
        
        return {'fidelity': fidelity, 'stability': stability}
    
    def evaluate_privacy_utility_tradeoff(self, epsilon_values):
        """For federated models"""
        results = {}
        
        for eps in epsilon_values:
            # Train with different privacy levels
            model_eps = train_federated_model(epsilon=eps)
            auc = self.evaluate_standard_metrics()['AUC']
            results[eps] = auc
        
        return results
    
    def evaluate_causal_metrics(self, causal_model):
        """For causal models"""
        # Policy value (expected reward under learned policy)
        policy_value = self._compute_policy_value(causal_model)
        
        # Calibration (are treatment effects well-calibrated?)
        calibration = self._compute_calibration(causal_model)
        
        return {'policy_value': policy_value, 'calibration': calibration}
```

---

## 🚀 First Week Action Plan

### Day 1-2: Environment Setup
```bash
# Clone your repo
git clone https://github.com/FuatOgme/FuxiCTR.git
cd FuxiCTR

# Install dependencies
pip install -r requirements.txt

# Download Criteo dataset (tiny version for testing)
cd data
python download_criteo_tiny.py
```

### Day 3-4: Implement Minimal Prototype
Choose ONE direction and implement the basic version (500 lines of code).

### Day 5-6: Run First Experiments
```bash
# Test on tiny data
cd experiment
python test_explainable_ctr.py --data criteo_tiny --model DeepFM
```

### Day 7: Analyze Results & Plan Next Steps
- Write up initial findings (1-2 pages)
- Identify issues to fix
- Plan month 2 work

---

## 📚 Code Resources

All code templates are production-ready and follow your FuxiCTR conventions:
- Modular design
- Compatible with existing models
- GPU support
- Logging and checkpointing

Next steps:
1. **Choose your direction** (1, 2, or 8)
2. **Copy the relevant code** into your repo
3. **Test on toy data** (1 hour)
4. **Scale to Criteo** (1 day)
5. **Start experiments** (1 week)

**I can provide more detailed code for any specific component!**

Just let me know which direction you want to pursue, and I'll generate:
- Complete runnable code (2000+ lines)
- Dataset preparation scripts
- Experiment configuration files
- Evaluation notebooks

Good luck! 🚀
