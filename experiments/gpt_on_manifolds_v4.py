"""
═══════════════════════════════════════════════════════════════════════════════
  GPT ON MANIFOLDS v4 — "Is Attention Commutative?" Synthesis
═══════════════════════════════════════════════════════════════════════════════

  Synthesis of gpt_on_manifolds_v3.py with the framework of:

      "Is Attention Commutative? Quantifying Contextuality via a
       Discrete Cartan Curvature Diagnostic" (February 2026)

  The paper's operational definitions (Appendix A, Table A1) replace the
  ad-hoc v3 diagnostics:

  ┌──────────────────────────────────────────────────────────────────────────┐
  │  Eq.  Object                     Definition                              │
  │  (1)  Sequence bundle            E = ⨆ₜ {t} × ℝᵈ                        │
  │  (2)  Transport generator        ωₜ ∈ 𝔰𝔬(d) ≅ 𝒢₂(ℝᵈ)                  │
  │  (3)  Connection bivector        ωₜ = Σₛ αₜ→ₛ · Bₜ,ₛ                    │
  │  (4)  Interaction bivector       Bₜ,ₛ = norm(q̂ₜ ∧ Δvₜ,ₛ)               │
  │  (5)  Discrete ext. derivative   (dω)ₜ = ωₜ₊₁ − ωₜ                      │
  │  (6)  Non-abelian term           (ω∧ω)ₜ = [ωₜ, ωₜ₊₁]                    │
  │  (7)  Discrete Cartan curvature  Ωₜ = (dω)ₜ + (ω∧ω)ₜ                    │
  │  (8)  Non-abelian ratio          ρ = Σ‖ω∧ω‖ / (Σ‖dω‖ + Σ‖ω∧ω‖)         │
  │  (9)  Transport map              Tₜ = exp(−ηωₜ)                         │
  │  (10) Fisher conditioning        κ_diag = max F̂ᵢᵢ / (min F̂ᵢᵢ + ε)      │
  └──────────────────────────────────────────────────────────────────────────┘

  CHANGES FROM v3 (each traceable to the paper):

  1. §3.2  Curvature is now computed on ADJACENT position pairs only
           ((dω)ₜ = ωₜ₊₁ − ωₜ, (ω∧ω)ₜ = [ωₜ, ωₜ₊₁]), replacing the v3
           all-pairs construction, per Eqs. (5)–(7).
  2. §3.3  NEW headline statistic: the non-abelian ratio ρ (Eq. 8), the
           paper's coordinate-robust measure of path dependence.
           ρ ≈ 0 → near-commuting transports; larger ρ → order-sensitive.
  3. §3.4  Holonomy uses EXACT transport maps Tₜ = exp(−ηωₜ) (Eq. 9) and
           closed triangular loops H = Tₖ Tⱼ Tᵢ with the principal
           rotation angle extracted from H, replacing v3's first-order
           Euler path comparison.
  4. §3.5  Fisher conditioning reported as κ_diag (Eq. 10) — an explicit
           diagonal-conditioning proxy, with ε in the denominator.
  5. §4.3  CONTROLS: (a) random/frozen baseline measured at init before
           any training; (b) order destruction (positional-embedding
           shuffle) measured after training. A credible contextuality
           metric should separate these from the trained model.
  6. §5.4  Hessian probes reframed as NEGATIVE-CURVATURE PROBES: we report
           the count of negative probe Rayleigh quotients, not an exact
           Morse index (the paper explicitly disclaims the latter).
  7. §6.4  Per-head taxonomy at convergence: flat / commutative-varying /
           order-sensitive, from the (‖dω‖, ‖ω∧ω‖) decomposition.

  Retained from v3: scalar cotangent-bundle autograd, BivectorND algebra,
  non-Euclidean embedding manifolds (Poincaré / product H×S / Grassmannian),
  natural-gradient optimizer (diagonal Fisher + Adam blend).

  References:
      Amari (1998). Natural Gradient Works Efficiently in Learning.
      Absil, Mahony, Sepulchre (2008). Optimization on Matrix Manifolds.
      Nickel & Kiela (2017). Poincaré Embeddings for Learning Hierarchies.
      Hestenes & Sobczyk (1984). Clifford Algebra to Geometric Calculus.
      Vaswani et al. (2017). Attention Is All You Need.

═══════════════════════════════════════════════════════════════════════════════
"""

import os
import math
import random
import numpy as np
from itertools import combinations
random.seed(42)
np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════
# §0. DATA MANIFOLD — 30 English names (paper §4.1)
# ═══════════════════════════════════════════════════════════════════════════

FALLBACK_NAMES = [
    'emma', 'olivia', 'ava', 'isabella', 'sophia', 'charlotte', 'mia',
    'amelia', 'harper', 'evelyn', 'liam', 'noah', 'william', 'james',
    'oliver', 'benjamin', 'elijah', 'lucas', 'mason', 'logan', 'ethan',
    'jacob', 'michael', 'daniel', 'henry', 'jackson', 'sebastian',
    'aiden', 'matthew', 'samuel',
]

if not os.path.exists('input.txt'):
    try:
        import urllib.request
        url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
        urllib.request.urlretrieve(url, 'input.txt')
    except Exception:
        with open('input.txt', 'w') as f:
            f.write('\n'.join(FALLBACK_NAMES))

with open('input.txt') as f:
    docs = [l.strip() for l in f.read().strip().split('\n') if l.strip()]
random.shuffle(docs)

# Paper §4.1: "a dataset of 30 English names (standard minimal-GPT benchmark)"
NUM_DOCS = int(os.environ.get('NUM_DOCS', 30))
docs = docs[:NUM_DOCS]
print(f"num docs: {len(docs)}")

alphabet = sorted(set(''.join(docs)))
BOS = len(alphabet)
vocab_size = len(alphabet) + 1
print(f"vocab size |V|: {vocab_size}")

# ═══════════════════════════════════════════════════════════════════════════
# §1. COTANGENT BUNDLE — Automatic Differentiation as Pullback
# ═══════════════════════════════════════════════════════════════════════════

class CotangentNode:
    """
    Node on the computation manifold. Pullback accumulator for reverse-mode AD.
    d(G∘F)* = dF* ∘ dG*  (functoriality of the cotangent functor T*)
    """
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0.0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, CotangentNode) else CotangentNode(other)
        return CotangentNode(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, CotangentNode) else CotangentNode(other)
        return CotangentNode(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, n):
        return CotangentNode(self.data**n, (self,), (n * self.data**(n-1),))

    def log(self):
        return CotangentNode(math.log(self.data), (self,), (1/self.data,))

    def exp(self):
        e = math.exp(self.data)
        return CotangentNode(e, (self,), (e,))

    def relu(self):
        return CotangentNode(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo, visited = [], set()
        def _sort(v):
            if v not in visited:
                visited.add(v)
                for c in v._children: _sort(c)
                topo.append(v)
        _sort(self)
        self.grad = 1.0
        for v in reversed(topo):
            for child, jac in zip(v._children, v._local_grads):
                child.grad += jac * v.grad


# ═══════════════════════════════════════════════════════════════════════════
# §2. CONFIGURATION (paper Table 1)
# ═══════════════════════════════════════════════════════════════════════════

n_embd = 16
n_head = 4
n_layer = 1
block_size = 16
head_dim = n_embd // n_head   # = 4 → bivector space ∧²ℝ⁴ has dim C(4,2) = 6

# Manifold selection: 'euclidean', 'hyperbolic', 'product', 'grassmannian'
# Paper §4.2: reported experiments use the product manifold H⁸ × S⁸.
MANIFOLD_TYPE = os.environ.get('MANIFOLD_TYPE', 'product')

HYPERBOLIC_C = 1.0            # κ = -c for Poincaré ball
PRODUCT_SPLIT_K = n_embd // 2 # H^8 × S^8
GRASSMANN_K = 4               # Gr(4, 4) per group
GRASSMANN_N = n_embd

# Diagnostics cadence (paper Table 1: curvature every 25, Hessian every 50)
COMPUTE_HESSIAN_EVERY = 50
COMPUTE_CURVATURE_EVERY = 25
HESSIAN_POWER_ITERS = 4
FISHER_SAMPLES = 3

# Paper-specific constants
HOLONOMY_ETA = 1.0            # step size η in Tₜ = exp(−ηωₜ)  (Eq. 9)
NA_EPS = 1e-12                # ε guarding ratio denominators (Eqs. 8, 10)
CONTROL_DOCS = 8              # docs per control diagnostic pass (§4.3)

num_steps = int(os.environ.get('NUM_STEPS', 1000))


# ═══════════════════════════════════════════════════════════════════════════
# §3. BIVECTOR ALGEBRA — ωₜ ∈ 𝔰𝔬(d) ≅ 𝒢₂(ℝᵈ)   (Eq. 2)
# ═══════════════════════════════════════════════════════════════════════════
#
#  A bivector B ∈ ∧²ℝⁿ is an antisymmetric 2-form. For ℝ⁴ (head_dim=4):
#      B = Σ_{i<j} B_{ij} (eᵢ ∧ eⱼ)
#
#  The commutator product acts on vectors:  B × v = ½(Bv - vB)
#  which for antisymmetric matrices reduces to B·v — the infinitesimal
#  rotation the connection applies to fiber vectors.

class BivectorND:
    """
    Bivector in ℝⁿ: element of ∧²ℝⁿ ≅ 𝔰𝔬(n).

    Stored as an antisymmetric matrix B_{ij}.
    dim(∧²ℝⁿ) = C(n,2) = n(n-1)/2.
    """
    def __init__(self, n, components=None):
        self.n = n
        if components is not None:
            self.B = np.array(components, dtype=float)
        else:
            self.B = np.zeros((n, n))

    @classmethod
    def from_wedge(cls, u, v):
        """Construct simple bivector u ∧ v = u⊗v - v⊗u."""
        u, v = np.asarray(u), np.asarray(v)
        n = len(u)
        B = np.outer(u, v) - np.outer(v, u)
        return cls(n, B)

    def commutator_with_vector(self, v):
        """B × v = ½(Bv - vB) = B·v for antisymmetric B."""
        return self.B @ np.asarray(v)

    def commutator_with_bivector(self, other):
        """[B₁, B₂] = B₁B₂ - B₂B₁ — the (ω∧ω) Lie bracket (Eq. 6)."""
        C = self.B @ other.B - other.B @ self.B
        return BivectorND(self.n, C)

    def inner(self, other):
        """⟨B₁, B₂⟩ = -½ tr(B₁ B₂)."""
        return -0.5 * np.trace(self.B @ other.B)

    @property
    def norm_squared(self):
        return self.inner(self)

    @property
    def norm(self):
        return math.sqrt(max(self.norm_squared, 0))

    def transport_map(self, eta=HOLONOMY_ETA):
        """
        Exact transport map Tₜ = exp(−η ωₜ)  (Eq. 9).

        exp of an antisymmetric matrix is a rotation in SO(n), computed
        via complex eigendecomposition (exact for our small fibers).
        """
        w, V = np.linalg.eig(-eta * self.B)
        return np.real(V @ np.diag(np.exp(w)) @ np.linalg.inv(V))

    def upper_triangle(self):
        vals = []
        for i in range(self.n):
            for j in range(i+1, self.n):
                vals.append(self.B[i, j])
        return np.array(vals)

    def __add__(self, other):
        return BivectorND(self.n, self.B + other.B)

    def __sub__(self, other):
        return BivectorND(self.n, self.B - other.B)

    def __mul__(self, scalar):
        return BivectorND(self.n, self.B * scalar)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __neg__(self):
        return BivectorND(self.n, -self.B)

    def __repr__(self):
        vals = self.upper_triangle()
        pairs = list(combinations(range(self.n), 2))
        terms = [f"{v:.4f} e{''.join(str(k+1) for k in p)}"
                 for v, p in zip(vals, pairs) if abs(v) > 1e-8]
        return f"Bivector({' + '.join(terms) if terms else '0'})"


def principal_rotation_angle(H):
    """
    Principal rotation angle of an orthogonal matrix H ∈ SO(n).

    Eigenvalues of H come in pairs e^{±iθⱼ}; the principal angle is
    max |θⱼ|. This is the paper's holonomy readout (§3.4).
    """
    eigvals = np.linalg.eigvals(H)
    return float(np.max(np.abs(np.angle(eigvals))))


# ═══════════════════════════════════════════════════════════════════════════
# §4. ATTENTION CONNECTION — Discrete Cartan Curvature Diagnostic
# ═══════════════════════════════════════════════════════════════════════════
#
#  Paper §2–3: attention defines a transport rule on the sequence bundle
#  E = ⨆ₜ {t} × ℝᵈ (Eq. 1). Per head, per position, the local transport
#  generator is the connection bivector (Eq. 3):
#
#      ωₜ = Σ_{s≤t} αₜ→ₛ · Bₜ,ₛ,     Bₜ,ₛ = norm(q̂ₜ ∧ Δvₜ,ₛ)   (Eq. 4)
#
#  The discrete Cartan curvature is a per-segment quantity (Eqs. 5–7):
#
#      Ωₜ = (dω)ₜ + (ω∧ω)ₜ,   (dω)ₜ = ωₜ₊₁ − ωₜ,   (ω∧ω)ₜ = [ωₜ, ωₜ₊₁]
#
#  and the headline contextuality statistic is the non-abelian ratio (Eq. 8):
#
#      ρ = Σₜ ‖(ω∧ω)ₜ‖ / (Σₜ ‖(dω)ₜ‖ + Σₜ ‖(ω∧ω)ₜ‖)
#
#  ρ ≈ 0  → local transports nearly commute (weak path dependence)
#  ρ > 0  → measurable order sensitivity in contextual composition

class AttentionConnectionGA:
    """
    Attention as discrete parallel transport on the sequence bundle.

    For each (layer, head), stores attention weights, query vectors and
    value frames, and computes:
      - Discrete Cartan curvature Ωₜ = (dω)ₜ + (ω∧ω)ₜ   (adjacent pairs)
      - Non-abelian ratio ρ                             (Eq. 8)
      - Holonomy via exact transport maps Tₜ = exp(−ηωₜ) (Eq. 9)
      - Bivector rotation-rate spectrum μⱼ of ωₜ
    """
    def __init__(self, head_dim):
        self.d = head_dim
        self.biv_dim = head_dim * (head_dim - 1) // 2
        self.reset()

    def reset(self):
        # Indexed by (layer, head, target_pos)
        self.alpha = {}           # αₜ→ₛ: connection coefficients
        self.value_frames = {}    # vₛ: fiber content at each position
        self.query_vecs = {}      # qₜ: query vectors
        self.key_vecs = {}        # kₛ: key vectors (frame field)

    def record(self, layer, head, target_pos, alpha_data, q_data, k_data_list, v_data_list):
        """Record attention data for post-hoc GA analysis."""
        key = (layer, head, target_pos)
        self.alpha[key] = np.array(alpha_data)
        self.query_vecs[key] = np.array(q_data)
        self.key_vecs[key] = [np.array(k) for k in k_data_list]
        self.value_frames[key] = [np.array(v) for v in v_data_list]

    def _positions(self, layer, head):
        return sorted(set(
            pos for (l, h, pos) in self.alpha if l == layer and h == head
        ))

    def _connection_bivector(self, layer, head, t):
        """
        Local transport generator ωₜ = Σₛ αₜ→ₛ · Bₜ,ₛ   (Eq. 3)

        with the model-agnostic interaction bivector of Eq. (4):
            Bₜ,ₛ = norm(q̂ₜ ∧ Δvₜ,ₛ),   Δvₜ,ₛ = vₛ − v̄ₜ

        i.e. the (normalized) plane spanned by "what position t is looking
        for" (q̂ₜ) and "what it receives from s" (centered value deviation).
        """
        key = (layer, head, t)
        if key not in self.alpha:
            return BivectorND(self.d)

        alpha = self.alpha[key]
        v_list = self.value_frames[key]
        T = len(v_list)

        if T < 2:
            return BivectorND(self.d)

        v_mean = np.mean([v for v in v_list], axis=0)

        q_t = self.query_vecs[key]
        q_norm = np.linalg.norm(q_t)
        if q_norm < 1e-10:
            return BivectorND(self.d)
        q_hat = q_t / q_norm

        omega = BivectorND(self.d)
        for s in range(T):
            delta_v = v_list[s] - v_mean
            delta_norm = np.linalg.norm(delta_v)
            if delta_norm < 1e-10 or alpha[s] < 1e-10:
                continue
            B_ts = BivectorND.from_wedge(q_hat, delta_v / delta_norm)
            omega = omega + B_ts * alpha[s]

        return omega

    def compute_discrete_cartan(self, layer, head):
        """
        Discrete Cartan curvature over ADJACENT position segments (Eqs. 5-7):

            (dω)ₜ  = ωₜ₊₁ − ωₜ            variation term
            (ω∧ω)ₜ = [ωₜ, ωₜ₊₁]           non-abelian (commutator) term
            Ωₜ     = (dω)ₜ + (ω∧ω)ₜ

        plus the non-abelian ratio (Eq. 8):

            ρ = Σₜ‖(ω∧ω)ₜ‖ / (Σₜ‖(dω)ₜ‖ + Σₜ‖(ω∧ω)ₜ‖)

        and mean/max sectional curvature K(Π) of Ωₜ over the canonical
        coordinate 2-planes of the fiber.

        Returns dict with per-head curvature diagnostics, including the
        raw variation energy Ed = Σ‖dω‖ and commutator energy Ec = Σ‖ω∧ω‖
        so callers can aggregate ρ across heads/steps in an
        energy-weighted way.
        """
        positions = self._positions(layer, head)

        empty = {'mean_K': 0., 'max_K': 0., 'norm_Omega': 0.,
                 'dw_norm': 0., 'ww_norm': 0., 'Ed': 0., 'Ec': 0.,
                 'rho': 0., 'n_segments': 0}
        if len(positions) < 2:
            return empty

        omegas = {t: self._connection_bivector(layer, head, t) for t in positions}

        sectional_curvatures = []
        omega_norms = []
        dw_norms = []
        ww_norms = []

        for t_idx in range(len(positions) - 1):
            t0, t1 = positions[t_idx], positions[t_idx + 1]

            dw_t = omegas[t1] - omegas[t0]                                # (Eq. 5)
            ww_t = omegas[t0].commutator_with_bivector(omegas[t1])        # (Eq. 6)
            Omega_t = dw_t + ww_t                                         # (Eq. 7)

            omega_norms.append(Omega_t.norm)
            dw_norms.append(dw_t.norm)
            ww_norms.append(ww_t.norm)

            # Sectional curvature K(Π) = Ω(u,v)·(u∧v)/|u∧v|² on canonical planes
            for a in range(self.d):
                for b in range(a+1, self.d):
                    u = np.zeros(self.d); u[a] = 1.0
                    v = np.zeros(self.d); v[b] = 1.0
                    plane = BivectorND.from_wedge(u, v)
                    K = Omega_t.inner(plane) / max(plane.norm_squared, 1e-10)
                    sectional_curvatures.append(K)

        Ed = float(np.sum(dw_norms))
        Ec = float(np.sum(ww_norms))
        rho = Ec / (Ed + Ec + NA_EPS)                                     # (Eq. 8)

        return {
            'mean_K': float(np.mean(np.abs(sectional_curvatures))),
            'max_K': float(np.max(np.abs(sectional_curvatures))),
            'norm_Omega': float(np.mean(omega_norms)),
            'dw_norm': float(np.mean(dw_norms)),
            'ww_norm': float(np.mean(ww_norms)),
            'Ed': Ed,
            'Ec': Ec,
            'rho': rho,
            'n_segments': len(positions) - 1,
        }

    def compute_holonomy(self, layer, head, eta=HOLONOMY_ETA, max_k=8):
        """
        Holonomy over triangular loops i → j → k → i  (paper §3.4).

        Uses exact transport maps Tₜ = exp(−ηωₜ) (Eq. 9). The holonomy
        operator is H_{i,j,k} = Tₖ Tⱼ Tᵢ; we report its principal
        rotation angle — the "angle of context".
        """
        positions = self._positions(layer, head)

        if len(positions) < 3:
            return {'mean_holonomy': 0., 'max_holonomy': 0., 'n_loops': 0}

        omegas = {t: self._connection_bivector(layer, head, t) for t in positions}
        transports = {t: omegas[t].transport_map(eta) for t in positions}

        holonomies = []
        for idx_k in range(2, min(len(positions), max_k)):
            k = positions[idx_k]
            for idx_j in range(1, idx_k):
                j = positions[idx_j]
                for idx_i in range(idx_j):
                    i = positions[idx_i]
                    H = transports[k] @ transports[j] @ transports[i]
                    holonomies.append(principal_rotation_angle(H))

        if not holonomies:
            return {'mean_holonomy': 0., 'max_holonomy': 0., 'n_loops': 0}

        return {
            'mean_holonomy': float(np.mean(holonomies)),
            'max_holonomy': float(np.max(holonomies)),
            'n_loops': len(holonomies),
        }

    def bivector_spectrum(self, layer, head):
        """
        Rotation-rate spectrum of ω at the last position (paper §5.2).

        Eigenvalues of the antisymmetric matrix ω come in ±iμ pairs;
        the μⱼ are the rotation rates of the invariant decomposition
        ω = Σⱼ μⱼ bⱼ into commuting simple bivectors.
        """
        positions = self._positions(layer, head)
        if not positions:
            return {'eigenvalues': [], 'rotation_rates': []}

        omega = self._connection_bivector(layer, head, positions[-1])
        eigvals = np.linalg.eigvals(omega.B)
        rates = sorted([abs(v.imag) for v in eigvals if v.imag > 1e-10], reverse=True)

        return {'eigenvalues': eigvals, 'rotation_rates': rates}


# Global connection tracker
attn_conn = AttentionConnectionGA(head_dim)


# ═══════════════════════════════════════════════════════════════════════════
# §5. NON-EUCLIDEAN EMBEDDING MANIFOLDS (paper §4.2)
# ═══════════════════════════════════════════════════════════════════════════

# ── Hyperbolic (Poincaré Ball) ──

def project_poincare(x, c=HYPERBOLIC_C, max_norm=0.95):
    """Project to Poincaré ball B^n = { x : √c·‖x‖ < 1 }."""
    nsq = sum(xi.data**2 if isinstance(xi, CotangentNode) else xi**2 for xi in x)
    norm = math.sqrt(nsq) if nsq > 0 else 1e-8
    radius = max_norm / math.sqrt(c)
    if norm > radius:
        scale = radius / norm
        return [xi * scale for xi in x]
    return x

def mobius_add(x, y, c=HYPERBOLIC_C):
    """Möbius addition x ⊕_c y — the group operation on H^n."""
    xd = [xi.data if isinstance(xi, CotangentNode) else xi for xi in x]
    yd = [yi.data if isinstance(yi, CotangentNode) else yi for yi in y]
    x_sq = sum(a**2 for a in xd)
    y_sq = sum(b**2 for b in yd)
    xy = sum(a*b for a, b in zip(xd, yd))
    denom = max(1.0 + 2*c*xy + c*c*x_sq*y_sq, 1e-8)
    cx = (1.0 + 2*c*xy + c*y_sq) / denom
    cy = (1.0 - c*x_sq) / denom
    return [xi * cx + yi * cy for xi, yi in zip(x, y)]


# ── Sphere ──

def project_sphere(x, target_norm=None):
    """RMSNorm = projection to S^{n-1}(√n)."""
    n = len(x)
    if target_norm is None:
        target_norm = math.sqrt(n)
    mean_sq = sum(xi * xi for xi in x) / n
    scale = (mean_sq + 1e-5) ** -0.5
    return [xi * scale for xi in x]


# ── Product: H^k × S^m ──

def project_product(x, k=PRODUCT_SPLIT_K):
    hyp = project_poincare(x[:k])
    sph = project_sphere(x[k:])
    return hyp + sph


def retract_product(x_base, delta, k=PRODUCT_SPLIT_K):
    hyp = mobius_add(x_base[:k], delta[:k])
    hyp = project_poincare(hyp)
    sph = [a + b for a, b in zip(x_base[k:], delta[k:])]
    return project_product(hyp + sph, k)


# ── Grassmannian: Gr(k, n) ──

def project_grassmannian(x, k=GRASSMANN_K):
    """Project embedding to Gr(k, n/k) by normalizing k groups (Stiefel)."""
    dim = len(x)
    group_size = dim // k
    result = list(x)
    for j in range(k):
        gs = j * group_size
        group = x[gs : gs + group_size]
        nsq = sum(xi * xi for xi in group)
        scale = (nsq + 1e-5) ** -0.5
        for i in range(group_size):
            result[gs + i] = group[i] * scale
    return result

def grassmann_retraction(x_base, delta, k=GRASSMANN_K):
    """First-order retraction: Euclidean step + re-orthogonalization."""
    result = [a + b for a, b in zip(x_base, delta)]
    return project_grassmannian(result, k)


# ── Unified Manifold Interface ──

def manifold_project(x):
    if MANIFOLD_TYPE == 'hyperbolic': return project_poincare(x)
    elif MANIFOLD_TYPE == 'product':  return project_product(x)
    elif MANIFOLD_TYPE == 'grassmannian': return project_grassmannian(x)
    else: return project_sphere(x)

def manifold_retract(x_base, delta):
    if MANIFOLD_TYPE == 'hyperbolic':
        return project_poincare(mobius_add(x_base, delta))
    elif MANIFOLD_TYPE == 'product':
        return retract_product(x_base, delta)
    elif MANIFOLD_TYPE == 'grassmannian':
        return grassmann_retraction(x_base, delta)
    else:
        return [a + b for a, b in zip(x_base, delta)]


# ═══════════════════════════════════════════════════════════════════════════
# §6. PARAMETER MANIFOLD & ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════

def init_fiber(nout, nin, sigma=0.08):
    return [[CotangentNode(random.gauss(0, sigma)) for _ in range(nin)]
            for _ in range(nout)]

atlas = {
    'wte': init_fiber(vocab_size, n_embd),
    'wpe': init_fiber(block_size, n_embd),
    'lm_head': init_fiber(vocab_size, n_embd),
}
for i in range(n_layer):
    atlas[f'layer{i}.attn_wq'] = init_fiber(n_embd, n_embd)
    atlas[f'layer{i}.attn_wk'] = init_fiber(n_embd, n_embd)
    atlas[f'layer{i}.attn_wv'] = init_fiber(n_embd, n_embd)
    atlas[f'layer{i}.attn_wo'] = init_fiber(n_embd, n_embd)
    atlas[f'layer{i}.mlp_fc1'] = init_fiber(4 * n_embd, n_embd)
    atlas[f'layer{i}.mlp_fc2'] = init_fiber(n_embd, 4 * n_embd)

params = [p for mat in atlas.values() for row in mat for p in row]
print(f"dim(M) = {len(params)} parameters")
print(f"Embedding manifold: {MANIFOLD_TYPE}")
print(f"Attention fiber: ℝ^{head_dim}, bivector space ∧²ℝ^{head_dim} has dim {head_dim*(head_dim-1)//2}")


# ═══════════════════════════════════════════════════════════════════════════
# §7. FIBER BUNDLE OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════

def bundle_morphism(x, W):
    """Section of Hom(E, E'): linear map between fibers."""
    return [sum(w * xi for w, xi in zip(row, x)) for row in W]

def exp_map_simplex(logits):
    """Softmax = inverse log chart on probability simplex Δⁿ."""
    mx = max(v.data for v in logits)
    exps = [(v - mx).exp() for v in logits]
    Z = sum(exps)
    return [e / Z for e in exps]


# ═══════════════════════════════════════════════════════════════════════════
# §8. TRANSFORMER GAUGE TRANSFORMATION
# ═══════════════════════════════════════════════════════════════════════════

def transformer_gauge(token_id, pos_id, frame_keys, frame_values):
    """
    Full forward pass as gauge transformation on the sequence bundle.
    Records attention data for post-hoc GA curvature analysis.
    """
    tok_fiber = atlas['wte'][token_id]
    pos_fiber = atlas['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_fiber, pos_fiber)]
    x = manifold_project(x)

    for li in range(n_layer):
        x_res = x
        x = manifold_project(x)

        q = bundle_morphism(x, atlas[f'layer{li}.attn_wq'])
        k = bundle_morphism(x, atlas[f'layer{li}.attn_wk'])
        v = bundle_morphism(x, atlas[f'layer{li}.attn_wv'])

        frame_keys[li].append(k)
        frame_values[li].append(v)

        x_transported = []
        for h in range(n_head):
            fs = h * head_dim
            q_h = q[fs:fs+head_dim]
            k_h = [ki[fs:fs+head_dim] for ki in frame_keys[li]]
            v_h = [vi[fs:fs+head_dim] for vi in frame_values[li]]

            # Compatibility (inner product on sub-fiber)
            compat = [
                sum(q_h[j] * k_h[s][j] for j in range(head_dim)) / head_dim**0.5
                for s in range(len(k_h))
            ]
            alpha = exp_map_simplex(compat)

            # Record for GA analysis (extract .data for post-hoc numpy analysis)
            attn_conn.record(
                li, h, pos_id,
                alpha_data=[a.data for a in alpha],
                q_data=[qi.data for qi in q_h],
                k_data_list=[[kij.data for kij in ks] for ks in k_h],
                v_data_list=[[vij.data for vij in vs] for vs in v_h],
            )

            # Parallel transport: Σ_s α_{ts} · v_s
            transported = [
                sum(alpha[s] * v_h[s][j] for s in range(len(v_h)))
                for j in range(head_dim)
            ]
            x_transported.extend(transported)

        x = bundle_morphism(x_transported, atlas[f'layer{li}.attn_wo'])
        x = manifold_retract(x_res, x)   # Riemannian retraction

        # MLP
        x_res = x
        x = manifold_project(x)
        x = bundle_morphism(x, atlas[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = bundle_morphism(x, atlas[f'layer{li}.mlp_fc2'])
        x = manifold_retract(x_res, x)

    return bundle_morphism(x, atlas['lm_head'])


# ═══════════════════════════════════════════════════════════════════════════
# §9. RIEMANNIAN OPTIMIZER — Fisher Natural Gradient (paper §3.5)
# ═══════════════════════════════════════════════════════════════════════════

class RiemannianOptimizer:
    """
    Natural gradient with empirical Fisher: Δθ = -η · G⁻¹(θ) · ∇L(θ).
    Blends Adam second moment with explicit Fisher diagonal for stability.
    """
    def __init__(self, params, lr=0.01, beta1=0.85, beta2=0.99, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.n = len(params)
        self.m = [0.0] * self.n
        self.v = [0.0] * self.n
        self.fisher = [1e-4] * self.n
        self.fisher_accum = [0.0] * self.n
        self.fisher_count = 0
        self.step_count = 0

    def accumulate_fisher(self):
        for i, p in enumerate(self.params):
            self.fisher_accum[i] += p.grad ** 2
        self.fisher_count += 1

    def update_fisher(self, gamma=0.95):
        if self.fisher_count == 0: return
        for i in range(self.n):
            self.fisher[i] = gamma * self.fisher[i] + (1-gamma) * self.fisher_accum[i] / self.fisher_count
            self.fisher_accum[i] = 0.0
        self.fisher_count = 0

    def step(self, lr_t):
        self.step_count += 1
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.beta1 * self.m[i] + (1-self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1-self.beta2) * g**2
            m_hat = self.m[i] / (1 - self.beta1**self.step_count)
            v_hat = self.v[i] / (1 - self.beta2**self.step_count)
            metric = max(math.sqrt(v_hat) + self.eps, math.sqrt(self.fisher[i]) + self.eps)
            p.data -= lr_t * m_hat / metric
            p.grad = 0.0

    def kappa_diag(self):
        """
        Diagonal empirical Fisher conditioning (Eq. 10):

            κ_diag = maxᵢ F̂ᵢᵢ / (minᵢ F̂ᵢᵢ + ε)

        A diagonal-conditioning proxy, NOT a full-Fisher condition number.
        """
        f = [fi for fi in self.fisher if fi > 1e-10]
        if not f: return 1.0
        return max(f) / (min(f) + NA_EPS)


# ═══════════════════════════════════════════════════════════════════════════
# §10. NEGATIVE-CURVATURE PROBES (paper §5.4)
# ═══════════════════════════════════════════════════════════════════════════
#
#  Hessian-vector-product probes via power iteration. Following the paper,
#  we report the COUNT of negative probe Rayleigh quotients as evidence of
#  saddle-like curvature — we do not claim an exact Morse index.

def compute_gradient(doc):
    """Compute gradient at current params for a sample."""
    tokens = [BOS] + [alphabet.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)
    fk = [[] for _ in range(n_layer)]
    fv = [[] for _ in range(n_layer)]
    losses = []
    for pos in range(n):
        logits = transformer_gauge(tokens[pos], pos, fk, fv)
        probs = exp_map_simplex(logits)
        losses.append(-probs[tokens[pos+1]].log())
    loss = (1/n) * sum(losses)
    loss.backward()
    grads = [p.grad for p in params]
    for p in params: p.grad = 0.0
    attn_conn.reset()
    return grads, loss.data

def hessian_vec_product(doc, v, eps=0.01):
    """Hv ≈ (∇L(θ+εv̂) - ∇L(θ-εv̂)) / (2ε) via central differences."""
    v_norm = math.sqrt(sum(vi**2 for vi in v)) + 1e-12
    v_hat = [vi / v_norm for vi in v]
    orig = [p.data for p in params]

    for p, vi in zip(params, v_hat): p.data += eps * vi
    g_plus, _ = compute_gradient(doc)

    for p, o, vi in zip(params, orig, v_hat): p.data = o - eps * vi
    g_minus, _ = compute_gradient(doc)

    for p, o in zip(params, orig): p.data = o
    return [(gp - gm) / (2 * eps) for gp, gm in zip(g_plus, g_minus)]

def negative_curvature_probe(doc, k=HESSIAN_POWER_ITERS):
    """
    Power iteration for extremal Rayleigh quotients of ∇²L, plus random
    interior probes. Returns (probe quotients sorted, negative count).
    """
    n = len(params)

    # λ_max via power iteration
    v = [random.gauss(0, 1) for _ in range(n)]
    nv = math.sqrt(sum(vi**2 for vi in v))
    v = [vi / nv for vi in v]
    for _ in range(k):
        Hv = hessian_vec_product(doc, v)
        nHv = math.sqrt(sum(h**2 for h in Hv)) + 1e-12
        v = [h / nHv for h in Hv]
    Hv = hessian_vec_product(doc, v)
    lam_max = sum(v[i]*Hv[i] for i in range(n))

    # λ_min via power iteration on -H
    w = [random.gauss(0, 1) for _ in range(n)]
    nw = math.sqrt(sum(wi**2 for wi in w))
    w = [wi / nw for wi in w]
    for _ in range(k):
        Hw = hessian_vec_product(doc, w)
        neg_Hw = [-h for h in Hw]
        nH = math.sqrt(sum(h**2 for h in neg_Hw)) + 1e-12
        w = [h / nH for h in neg_Hw]
    Hw = hessian_vec_product(doc, w)
    lam_min = sum(w[i]*Hw[i] for i in range(n))

    # Random Rayleigh quotients for interior
    interior = []
    for _ in range(3):
        r = [random.gauss(0, 1) for _ in range(n)]
        nr = math.sqrt(sum(ri**2 for ri in r))
        r = [ri / nr for ri in r]
        Hr = hessian_vec_product(doc, r)
        interior.append(sum(r[i]*Hr[i] for i in range(n)))

    probes = sorted([lam_min] + interior + [lam_max])
    neg_count = sum(1 for e in probes if e < -1e-6)
    return probes, neg_count


# ═══════════════════════════════════════════════════════════════════════════
# §11. CONTROL DIAGNOSTICS (paper §4.3)
# ═══════════════════════════════════════════════════════════════════════════
#
#  Two controls establish that ρ reflects LEARNED order sensitivity:
#    (a) Random/frozen baseline — ρ and holonomy at random init, no training.
#    (b) Order destruction — shuffle positional embeddings of the trained
#        model; a credible contextuality metric should drop substantially.

def diagnostic_pass(sample_docs):
    """
    Forward-only pass over sample_docs; aggregates energy-weighted ρ
    (Eq. 8, pooled over docs/layers/heads) and mean holonomy.
    """
    Ed_total, Ec_total = 0.0, 0.0
    hol_angles = []

    for doc in sample_docs:
        attn_conn.reset()
        tokens = [BOS] + [alphabet.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)
        fk = [[] for _ in range(n_layer)]
        fv = [[] for _ in range(n_layer)]
        for pos in range(n):
            transformer_gauge(tokens[pos], pos, fk, fv)

        for li in range(n_layer):
            for h in range(n_head):
                c = attn_conn.compute_discrete_cartan(li, h)
                Ed_total += c['Ed']
                Ec_total += c['Ec']
                hol = attn_conn.compute_holonomy(li, h)
                if hol['n_loops'] > 0:
                    hol_angles.append(hol['mean_holonomy'])

    attn_conn.reset()
    rho = Ec_total / (Ed_total + Ec_total + NA_EPS)
    mean_hol = float(np.mean(hol_angles)) if hol_angles else 0.0
    return {'rho': rho, 'mean_holonomy': mean_hol,
            'Ed': Ed_total, 'Ec': Ec_total}


def order_destruction_control(sample_docs):
    """
    Control (§4.3): shuffle the positional-embedding table (destroying
    consistent positional information) and re-measure ρ and holonomy.
    Restores the original wpe afterwards.
    """
    wpe = atlas['wpe']
    original_data = [[p.data for p in row] for row in wpe]

    perm = list(range(len(wpe)))
    random.shuffle(perm)
    for r, src in enumerate(perm):
        for c in range(len(wpe[r])):
            wpe[r][c].data = original_data[src][c]

    result = diagnostic_pass(sample_docs)

    for r in range(len(wpe)):
        for c in range(len(wpe[r])):
            wpe[r][c].data = original_data[r][c]

    return result


# ═══════════════════════════════════════════════════════════════════════════
# §12. TRAINING — Riemannian Gradient Flow with Cartan Diagnostics
# ═══════════════════════════════════════════════════════════════════════════

learning_rate = 0.01
optimizer = RiemannianOptimizer(params, lr=learning_rate)

curv_history = []
hess_history = []
fisher_history = []

control_sample = docs[:min(CONTROL_DOCS, len(docs))]

print(f"\n{'='*72}")
print(f"  CONTROL (a): random/frozen baseline at initialization (§4.3)")
print(f"{'='*72}")
frozen_baseline = diagnostic_pass(control_sample)
print(f"  ρ(frozen)        = {frozen_baseline['rho']:.4f}")
print(f"  holonomy(frozen) = {frozen_baseline['mean_holonomy']:.4f} rad")

print(f"\n{'='*72}")
print(f"  TRAINING: Riemannian Gradient Flow on ({MANIFOLD_TYPE}) Manifold")
print(f"  Discrete Cartan curvature: Ωₜ = (dω)ₜ + (ω∧ω)ₜ   (Eqs. 5-7)")
print(f"  Non-abelian ratio: ρ = Σ‖ω∧ω‖ / (Σ‖dω‖ + Σ‖ω∧ω‖)   (Eq. 8)")
print(f"{'='*72}\n")

for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [alphabet.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    attn_conn.reset()
    fk = [[] for _ in range(n_layer)]
    fv = [[] for _ in range(n_layer)]
    losses = []
    for pos in range(n):
        logits = transformer_gauge(tokens[pos], pos, fk, fv)
        probs = exp_map_simplex(logits)
        losses.append(-probs[tokens[pos+1]].log())
    loss = (1/n) * sum(losses)
    loss.backward()

    optimizer.accumulate_fisher()
    if (step+1) % FISHER_SAMPLES == 0:
        optimizer.update_fisher()

    lr_t = learning_rate * (1 - step / num_steps)
    optimizer.step(lr_t)

    # ── DISCRETE CARTAN CURVATURE DIAGNOSTICS ──
    if (step+1) % COMPUTE_CURVATURE_EVERY == 0 and n >= 3:
        curv_data = {'step': step+1}
        all_K, all_hol, all_dw, all_ww, all_rates = [], [], [], [], []
        Ed_step, Ec_step = 0.0, 0.0

        for li in range(n_layer):
            for h in range(n_head):
                c = attn_conn.compute_discrete_cartan(li, h)
                if c['n_segments'] > 0:
                    all_K.append(c['mean_K'])
                    all_dw.append(c['dw_norm'])
                    all_ww.append(c['ww_norm'])
                    Ed_step += c['Ed']
                    Ec_step += c['Ec']

                hol = attn_conn.compute_holonomy(li, h)
                if hol['n_loops'] > 0:
                    all_hol.append(hol['mean_holonomy'])

                spec = attn_conn.bivector_spectrum(li, h)
                all_rates.extend(spec['rotation_rates'])

        curv_data['mean_K'] = float(np.mean(all_K)) if all_K else 0.
        curv_data['max_K'] = float(np.max(all_K)) if all_K else 0.
        curv_data['mean_hol'] = float(np.mean(all_hol)) if all_hol else 0.
        curv_data['dw_norm'] = float(np.mean(all_dw)) if all_dw else 0.
        curv_data['ww_norm'] = float(np.mean(all_ww)) if all_ww else 0.
        curv_data['rho'] = Ec_step / (Ed_step + Ec_step + NA_EPS)
        curv_data['Ed'] = Ed_step
        curv_data['Ec'] = Ec_step
        curv_data['top_rate'] = max(all_rates) if all_rates else 0.
        curv_history.append(curv_data)

    # ── NEGATIVE-CURVATURE PROBES ──
    if (step+1) % COMPUTE_HESSIAN_EVERY == 0:
        probes, neg = negative_curvature_probe(docs[(step+7) % len(docs)])
        hess_history.append({'step': step+1, 'probes': probes, 'neg': neg})
        fisher_history.append({'step': step+1, 'kappa': optimizer.kappa_diag()})

    # ── Logging ──
    if (step+1) % 50 == 0:
        log = f"step {step+1:4d}/{num_steps} | L={loss.data:.4f}"
        if curv_history:
            ch = curv_history[-1]
            log += f" | K̄={ch['mean_K']:.4f} ρ={ch['rho']:.3f} hol={ch['mean_hol']:.4f}"
        if hess_history:
            hh = hess_history[-1]
            log += f" | λ∈[{min(hh['probes']):.2e},{max(hh['probes']):.2e}] neg={hh['neg']}"
        if fisher_history:
            log += f" | κ_diag={fisher_history[-1]['kappa']:.1e}"
        print(log)


# ═══════════════════════════════════════════════════════════════════════════
# §13. GEOMETRIC ANALYSIS REPORT — paper Tables 2-5
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*72}")
print(f"  IS ATTENTION COMMUTATIVE? — {MANIFOLD_TYPE.upper()} MANIFOLD")
print(f"  Discrete Cartan diagnostic: Ωₜ = (dω)ₜ + (ω∧ω)ₜ")
print(f"{'='*72}")

if curv_history:
    print(f"\n── Table 2: Curvature evolution during training ──")
    print(f"  (dω)ₜ = ωₜ₊₁ − ωₜ    (ω∧ω)ₜ = [ωₜ, ωₜ₊₁]    (adjacent segments)")
    print()
    print(f"  {'Step':>6}  {'K̄(sect)':>9}  {'‖dω‖':>9}  {'‖ω∧ω‖':>9}  {'Hol(rad)':>9}  {'ρ':>7}  {'Top μ':>9}")
    print(f"  {'─'*6}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*7}  {'─'*9}")
    for ch in curv_history:
        print(f"  {ch['step']:6d}  {ch['mean_K']:9.5f}  {ch['dw_norm']:9.5f}  "
              f"{ch['ww_norm']:9.5f}  {ch['mean_hol']:9.5f}  {ch['rho']:7.4f}  {ch['top_rate']:9.5f}")

    # ── Non-abelian ratio ρ: the headline statistic (Eq. 8, §5.1) ──
    final = curv_history[-1]
    rho_final = final['rho']
    rho_traj_mean = float(np.mean([ch['rho'] for ch in curv_history]))
    Ed_traj = sum(ch['Ed'] for ch in curv_history)
    Ec_traj = sum(ch['Ec'] for ch in curv_history)
    rho_traj_energy = Ec_traj / (Ed_traj + Ec_traj + NA_EPS)

    print(f"\n── Non-Abelian Ratio ρ (Eq. 8) ──")
    print(f"  ρ (final step, energy-weighted)      = {rho_final:.4f}")
    print(f"  ρ (trajectory, mean-weighted)        = {rho_traj_mean:.4f}")
    print(f"  ρ (trajectory, energy-weighted)      = {rho_traj_energy:.4f}")
    print(f"  → {rho_traj_mean*100:.1f}% of learned curvature is attributable to")
    print(f"    NON-COMMUTATIVITY of successive transport generators.")
    if rho_final < 0.01:
        print(f"  Verdict: attention transport is NEARLY COMMUTATIVE here.")
    else:
        print(f"  Verdict: attention transport is MEASURABLY NON-COMMUTATIVE —")
        print(f"  order of local transport steps materially changes the result.")

    c0, cN = curv_history[0]['mean_K'], curv_history[-1]['mean_K']
    h0, hN = curv_history[0]['mean_hol'], curv_history[-1]['mean_hol']
    m0, mN = curv_history[0]['top_rate'], curv_history[-1]['top_rate']
    print(f"\n  Sectional curvature: {c0:.4f} → {cN:.4f} "
          f"({'×%.1f growth' % (cN/max(c0,1e-10)) if cN > c0 else 'decreasing'})")
    print(f"  Mean holonomy angle: {h0:.4f} → {hN:.4f} rad "
          f"(≈{math.degrees(hN):.1f}° loop-induced rotation)")
    print(f"  Top rotation rate μ: {m0:.4f} → {mN:.4f}")

    # ── §6.4 Per-head taxonomy at convergence ──
    print(f"\n── Per-Head Taxonomy at Convergence (§6.4) ──")
    print(f"  flat: ‖Ω‖≈0 (copy/average) | commutative-varying: dω≫ω∧ω | order-sensitive: large ω∧ω")
    attn_conn.reset()
    _tax_doc = max(control_sample, key=len)
    _tokens = [BOS] + [alphabet.index(ch) for ch in _tax_doc] + [BOS]
    _n = min(block_size, len(_tokens) - 1)
    _fk = [[] for _ in range(n_layer)]
    _fv = [[] for _ in range(n_layer)]
    for pos in range(_n):
        transformer_gauge(_tokens[pos], pos, _fk, _fv)
    for li in range(n_layer):
        for h in range(n_head):
            c = attn_conn.compute_discrete_cartan(li, h)
            if c['norm_Omega'] < 1e-3:
                kind = 'FLAT (near-trivial transport)'
            elif c['rho'] < 0.05:
                kind = 'COMMUTATIVE-VARYING (position-dependent, generators commute)'
            else:
                kind = 'ORDER-SENSITIVE (genuinely non-commutative transport)'
            print(f"  layer {li} head {h}: ‖Ω‖={c['norm_Omega']:.4f} "
                  f"‖dω‖={c['dw_norm']:.4f} ‖ω∧ω‖={c['ww_norm']:.4f} "
                  f"ρ={c['rho']:.3f} → {kind}")
    attn_conn.reset()

if fisher_history:
    print(f"\n── Table 3: Diagonal empirical Fisher conditioning κ_diag (Eq. 10) ──")
    print(f"  {'Step':>6}  {'κ_diag':>12}")
    print(f"  {'─'*6}  {'─'*12}")
    for fh in fisher_history:
        print(f"  {fh['step']:6d}  {fh['kappa']:12.2e}")
    print(f"  (diagonal proxy only — not a full-Fisher condition number)")

if hess_history:
    print(f"\n── Table 4: Negative-curvature probes (§5.4) ──")
    print(f"  {'Step':>6}  {'λ_min':>12}  {'λ_max':>12}  {'Neg.':>4}  Interpretation")
    print(f"  {'─'*6}  {'─'*12}  {'─'*12}  {'─'*4}  {'─'*28}")
    prev_neg = 0
    transitions = 0
    for hh in hess_history:
        probes = hh['probes']
        if hh['neg'] == 0:
            interp = 'positive-definite region'
        elif hh['neg'] == 1:
            interp = 'saddle-like curvature'
        else:
            interp = 'higher-order saddle probe'
        if hh['neg'] != prev_neg:
            transitions += 1
        prev_neg = hh['neg']
        print(f"  {hh['step']:6d}  {min(probes):12.4e}  {max(probes):12.4e}  "
              f"{hh['neg']:4d}  {interp}")
    print(f"\n  {transitions} negative-curvature transitions across {num_steps} steps.")
    print(f"  (probe-based evidence only — no exact Morse index claimed)")


# ═══════════════════════════════════════════════════════════════════════════
# §14. CONTROLS (paper §4.3) — does ρ reflect LEARNED order sensitivity?
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*72}")
print(f"  CONTROLS (§4.3)")
print(f"{'='*72}")

trained_diag = diagnostic_pass(control_sample)
destroyed_diag = order_destruction_control(control_sample)

print(f"\n  {'Condition':<32}  {'ρ':>8}  {'Holonomy (rad)':>15}")
print(f"  {'─'*32}  {'─'*8}  {'─'*15}")
print(f"  {'random/frozen (init)':<32}  {frozen_baseline['rho']:8.4f}  "
      f"{frozen_baseline['mean_holonomy']:15.4f}")
print(f"  {'trained':<32}  {trained_diag['rho']:8.4f}  "
      f"{trained_diag['mean_holonomy']:15.4f}")
print(f"  {'trained + order destruction':<32}  {destroyed_diag['rho']:8.4f}  "
      f"{destroyed_diag['mean_holonomy']:15.4f}")

delta_frozen = trained_diag['rho'] - frozen_baseline['rho']
delta_destroy = trained_diag['rho'] - destroyed_diag['rho']
print(f"\n  Δρ(trained − frozen)     = {delta_frozen:+.4f}")
print(f"  Δρ(trained − destroyed)  = {delta_destroy:+.4f}")
if delta_frozen > 0 and delta_destroy > 0:
    print(f"  → ρ separates the trained model from both controls:")
    print(f"    the measured non-commutativity is LEARNED order sensitivity.")
else:
    print(f"  → Controls do not clearly separate; treat ρ with caution here")
    print(f"    (see paper §6.5: construction dependence & small-scale caveats).")


# ═══════════════════════════════════════════════════════════════════════════
# §15. SUMMARY (paper Table 5) & INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n── Table 5: Summary of key empirical findings ──")
if curv_history:
    print(f"  Non-abelian ratio ρ            ≈ {rho_traj_mean:.3f} "
          f"({rho_traj_mean*100:.1f}% of curvature from non-commutativity)")
    print(f"  Mean holonomy angle            {hN:.3f} rad (≈{math.degrees(hN):.1f}° loop-induced rotation)")
    print(f"  Sectional curvature            {c0:.3f} → {cN:.3f}")
    print(f"  Top rotation rate μ            {m0:.2f} → {mN:.2f}")
if fisher_history:
    print(f"  Fisher conditioning κ_diag     {fisher_history[-1]['kappa']:.1e}")
if hess_history:
    print(f"  Negative-curvature transitions {transitions} across {num_steps} steps")

print(f"\n── Manifold Summary ──")
print(f"  Type: {MANIFOLD_TYPE}")
if MANIFOLD_TYPE == 'hyperbolic':
    print(f"  Curvature: κ = -{HYPERBOLIC_C} (constant negative)")
elif MANIFOLD_TYPE == 'product':
    print(f"  Geometry: H^{PRODUCT_SPLIT_K} × S^{n_embd - PRODUCT_SPLIT_K}")
elif MANIFOLD_TYPE == 'grassmannian':
    gs = n_embd // GRASSMANN_K
    print(f"  Geometry: Gr({GRASSMANN_K}, {gs}) — {GRASSMANN_K}-planes in ℝ^{gs}")
print(f"  Attention fiber: ℝ^{head_dim}, ω ∈ ∧²ℝ^{head_dim} (dim = {head_dim*(head_dim-1)//2})")
print(f"  Transport map: Tₜ = exp(−{HOLONOMY_ETA}·ωₜ)")

temperature = 0.5
print(f"\n── Sampling from learned statistical manifold ({MANIFOLD_TYPE}) ──")
print(f"── Temperature = {temperature} (conformal rescaling g_T = (1/T)·g) ──\n")

for s_idx in range(20):
    attn_conn.reset()
    fk = [[] for _ in range(n_layer)]
    fv = [[] for _ in range(n_layer)]
    tid = BOS
    path = []
    for pos in range(block_size):
        logits = transformer_gauge(tid, pos, fk, fv)
        probs = exp_map_simplex([l / temperature for l in logits])
        tid = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if tid == BOS: break
        path.append(alphabet[tid])
    print(f"  path {s_idx+1:2d}: {''.join(path)}")

print(f"\n{'='*72}")
print(f"  GPT on Manifolds v4 — 'Is Attention Commutative?' synthesis complete")
print(f"  Manifold: {MANIFOLD_TYPE} | Params: {len(params)}")
print(f"  Ωₜ = (dω)ₜ + (ω∧ω)ₜ | ρ = Σ‖ω∧ω‖/(Σ‖dω‖+Σ‖ω∧ω‖) | Tₜ = exp(−ηωₜ)")
print(f"{'='*72}")
