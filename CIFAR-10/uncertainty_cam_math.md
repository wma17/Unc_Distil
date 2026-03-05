# UncertaintyCAM: Mathematical Formulation

This document describes how the uncertainty maps are calculated in the UncertaintyCAM visualization.

---

## 1. Epistemic Uncertainty (EU) Definition

For the BNN teacher ensemble:

$$
EU = TU - AU = H(\bar{p}) - \frac{1}{M}\sum_{m=1}^{M} H(p_m)
$$

where:
- $\bar{p} = \frac{1}{M}\sum_{m} p_m$ is the mean predictive distribution over $M$ members
- $p_m = \text{softmax}(\logits_m)$ is member $m$'s class distribution
- $H(p) = -\sum_{c=1}^{C} p_c \log p_c$ is the entropy (in nats)

This equals the mutual information $I[y;\theta|x,D]$ between the label and parameters given the data (Depeweg et al., 2018).

---

## 2. GradCAM for Uncertainty (General)

GradCAM builds a spatial importance map from the gradient of our scalar target $y$ w.r.t. the last conv layer activations $A$.

**GradCAM weights** (per channel $k$):

$$
\alpha^k = \frac{1}{Z}\sum_{i,j} \frac{\partial y}{\partial A^k_{ij}}
$$

where $Z$ = number of spatial locations (global average pooling of gradients).

**CAM construction**:

$$
L^{\text{Grad-CAM}} = \text{ReLU}\left(\sum_k \alpha^k A^k\right)
$$

ReLU keeps only features that *positively* influence the target.

---

## 3. Teacher (BNN Ensemble) Uncertainty Map

EU depends on all members' predictions:

$$
EU = H\!\left(\frac{1}{M}\sum_m p_m\right) - \frac{1}{M}\sum_m H(p_m)
$$

Backpropagation yields $\partial EU/\partial A^m$ for each member $m$ (gradient of EU w.r.t. that member's conv features).

**Per-member GradCAM**:

$$
L_m = \text{ReLU}\left(\sum_k \alpha^m_k A^m_k\right), \quad
\alpha^m_k = \frac{1}{Z}\sum_{i,j}\frac{\partial EU}{\partial A^{m,k}_{ij}}
$$

**Aggregation**: average over members (all contribute to EU):

$$
L^{\text{Teacher}} = \frac{1}{M}\sum_{m=1}^{M} L_m
$$

---

## 4. Student Uncertainty Map

The student predicts EU via a regression head:

$$
EU_S = \text{softplus}\bigl(W_2 \cdot \text{ReLU}(W_1 \cdot [\text{feat}; \text{softmax}(\logits).\text{detach()}])\bigr)
$$

Gradients flow only through `feat` (the backbone features); `softmax(logits)` is detached.

**GradCAM** on layer4 activations $A$:

$$
\alpha^k = \frac{1}{Z}\sum_{i,j} \frac{\partial EU_S}{\partial A^k_{ij}}
$$

$$
L^{\text{Student}} = \text{ReLU}\left(\sum_k \alpha^k A^k\right)
$$

---

## 5. Summary Table

| Model | Target $y$ | Gradient $\partial y / \partial A$ | Map |
|-------|------------|-------------------------------------|-----|
| **Teacher** | $EU = H(\bar{p}) - \frac{1}{M}\sum_m H(p_m)$ | $\partial EU/\partial A^m$ for each member $m$ | $\frac{1}{M}\sum_m L_m$ |
| **Student** | $EU_S$ (scalar from EU head) | $\partial EU_S/\partial A$ from backbone | Single $L$ |

Both maps highlight *where in the image* the features that increase epistemic uncertainty are localized.
