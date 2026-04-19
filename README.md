# 🧬 HybridMythos: Hierarchical Recurrent-Depth Transformer

**HybridMythos** is an open-source, theoretical architecture that merges the infinite-depth looping of **OpenMythos** with the brain-inspired, dual-timescale cognitive processing of the **Hierarchical Reasoning Model (HRM)**. 

By replacing the standard "flat" recurrent block of a looped transformer with a structured, nested-loop hierarchy, HybridMythos achieves the parameter efficiency of a continuous latent reasoner while maintaining the strategic oversight necessary for complex, multi-hop problem-solving.

---

## 💡 The Core Concept

Modern large language models struggle with deep reasoning because they are fundamentally shallow, relying on explicit, token-heavy Chain-of-Thought (CoT) to solve algorithmic problems. 

HybridMythos solves this by combining two breakthrough paradigms:
1. **OpenMythos (Recurrent-Depth Transformer):** Reuses the same transformer weights in a continuous loop, allowing the model to "think" deeper in latent space without expanding its parameter count. It maintains stability across infinite loops using a strictly constrained LTI (Linear Time-Invariant) injection matrix.
2. **HRM (Hierarchical Reasoning Model):** Abandons flat loops in favor of two coupled recurrent modules operating at different timescales: a high-level module for slow, abstract planning and a low-level module for rapid, detailed execution.

**The Synergy:** HybridMythos uses HRM's "Manager/Worker" architecture inside the OpenMythos stability loop. It gives the looped transformer a structured brain that can dynamically course-correct during long reasoning trajectories.

---

## 🏗️ Architecture

The model forward pass is divided into three functional stages:

**1. Prelude [Run Once]**
Standard transformer layers process the input tokens to generate the initial hidden state ($h_0$) and the encoded injection vector ($e$).

**2. Hierarchical Recurrent Block [The Core Engine]**
Instead of a single loop, the recurrent block maintains two hidden states ($h_{high}$ and $h_{low}$) through a nested execution:
*   **The Outer Loop (H-Module / "Manager"):** Updates occasionally to set the global reasoning direction and establish a new local equilibrium for the worker. 
*   **The Inner Loop (L-Module / "Worker"):** Runs multiple fast iterations per H-Module step, rapidly refining details and grinding through computations.
*   **Stability Injection:** At every step, the original input $e$ is injected using learned matrices $A$ and $B$, where the spectral radius $\rho(A) < 1$. This prevents the deep recurrent loops from exploding into NaN values.

**3. Coda [Run Once]**
Standard transformer layers process the final $h_{low}$ state into the output logits.

---

## 🚀 Quick Start & PyTorch Implementation

### Installation
Clone the required base repositories (from our conversation history context):
```bash
git clone https://github.com/kyegomez/OpenMythos.git
git clone https://github.com/sapientinc/HRM.git
```

### The PyTorch Wrapper (`hybrid_model.py`)
Here is the structural PyTorch pseudo-code to instantiate the hybrid model:

```python
import torch
import torch.nn as nn
from open_mythos.main import MythosConfig, OpenMythos
from hrm_modules import HighLevelModule, LowLevelModule

class HybridMythos(nn.Module):
    def __init__(self, cfg: MythosConfig):
        super().__init__()
        # 1. Prelude and Coda from OpenMythos
        self.prelude = OpenMythos(cfg).prelude
        self.coda = OpenMythos(cfg).coda
        
        # 2. Dual-timescale reasoning modules from HRM
        self.h_module = HighLevelModule(cfg.dim)
        self.l_module = LowLevelModule(cfg.dim)
        
        # 3. LTI Stability matrices
        self.injection_A = nn.Parameter(torch.randn(cfg.dim)) # constrained to negative diag
        self.injection_B = nn.Parameter(torch.randn(cfg.dim, cfg.dim))

    def forward(self, input_ids, n_outer=4, n_inner=4):
        # Prelude
        h_low, e = self.prelude(input_ids)
        h_high = h_low.clone()
        
        # Hierarchical Recurrent Block
        for t in range(n_outer):
            # Slow Manager Update
            h_high = self.h_module(h_high, h_low, e)
            
            for s in range(n_inner):
                # Fast Worker Update
                h_low = self.l_module(h_low, h_high, e)
                
                # Stability Injection (Spectral Radius < 1)
                h_low = self.injection_A * h_low + self.injection_B @ e + h_low
                
        # Coda
        return self.coda(h_low)
```

---

## 🧠 Training & Scaling Strategy

As discussed in our configuration strategies:
1. **Efficient Pre-training:** Train your ~700M parameter base model on your dataset with a fixed, small loop count (e.g., `n_outer=4`, `n_inner=4`).
2. **Instruction Tuning:** Freeze the hierarchical recurrent block and only fine-tune the Prelude, Coda, and injection matrices to prevent catastrophic forgetting of reasoning pathways. 
3. **Inference-Time Scaling:** At inference time, dynamically increase the loops (e.g., `n_outer=8`, `n_inner=8`). The 700M parameter model will execute a deeper reasoning chain.

---

## 🔗 Alternative Open-Source Implementations

If you are looking for a pre-built datacenter-ready alternative that leverages similar mechanics, check out **Hierarchos** (`necat101/Hierarchos`). 
Hierarchos natively synergizes HRM-style reasoning with an **RWKV v8** linear-attention backbone for $O(1)$ inference cost, and integrates **Titans** neural memory for lifelong context retention.

## 📜 Acknowledgements & References
*   **OpenMythos:** Recurrent-Depth Transformer architecture by Kye Gomez.
*   **HRM:** Hierarchical Reasoning Model by Sapient Intelligence.
*   **Hierarchos:** Hybrid HRM/Titans/RWKV framework by Makhi Burroughs.
