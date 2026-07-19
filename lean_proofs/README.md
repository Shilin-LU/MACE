# Lean 4 / Mathlib formalization of MACE's closed-form solution

Machine-checked (Lean 4 + [Mathlib](https://github.com/leanprover-community/mathlib4)) proofs of the mathematical results of MACE: the closed-form cross-attention refinement solution (Eq. (2), derived in Appendix B), its multi-LoRA fusion analogue (Eq. (7)) and domain-specific variant (Eq. (19)), and the non-negativity of the concept-focal importance sampling density (Eq. (5)).

Everything builds with zero `sorry`, and `#print axioms` reports only the standard classical axioms (`propext`, `Classical.choice`, `Quot.sound`) for every result. The files are:

- `ClosedForm.lean` — the closed-form refinement solution (Eq. (2));
- `GramMatrix.lean` — the Gram matrix is positive (semi)definite, hence invertible;
- `Instances.lean` — the fusion (Eq. (7)) and domain-specific (Eq. (19)) closed forms;
- `CFIS.lean` — the sampling density (Eq. (5)) is non-negative;
- `Basic.lean` — shared definitions; `Mace.lean` — the umbrella import.

`verify.sh` runs the full check: `lake build`, a `sorry` sweep, and `#print axioms` on every headline result.

## How to build

Requires a Lean 4 toolchain (`elan`/`lake`). Run `lake build` in this folder. The Mathlib version is pinned in `lakefile.toml` / `lake-manifest.json`.

## What is proved

The refinement objective (Eq. (1)) is

$$\mathcal{L}(W'_k) = \sum_{i=1}^{n} \left\lVert W'_k \mathbf{e}^f_i - W_k \mathbf{e}^g_i \right\rVert_2^2 + \lambda_1 \sum_{i=n+1}^{n+m} \left\lVert W'_k \mathbf{e}^p_i - W_k \mathbf{e}^p_i \right\rVert_2^2 .$$

It is formalized as a weighted least-squares problem over a finite index set: for inputs $a_i \in \mathbb{R}^{d_2}$, targets $b_i \in \mathbb{R}^{d_1}$ and weights $c_i \ge 0$,

$$L(W) = \sum_i c_i \left\lVert W a_i - b_i \right\rVert_2^2,$$

with Gram matrix $G = \sum_i c_i a_i a_i^\top$ and cross matrix $A = \sum_i c_i b_i a_i^\top$. The paper's mapping terms take $c_i = 1$ and the preserving terms $c_i = \lambda_1$.

- **`Basic.lean`** — the objective $L$, the Gram matrix $G$, and the cross matrix $A$, with the entrywise identities used downstream.

- **`GramMatrix.lean`** — the quadratic-form identity $x^\top G x = \sum_i c_i (x \cdot a_i)^2$ (Eq. (18)), from which $G$ is positive semidefinite (weights $c_i \ge 0$), and positive definite — so $\det G$ is a unit and $G$ is invertible — under the condition that no nonzero $x$ is orthogonal to every $a_i$ carrying positive weight. This is the precise sufficient condition for the full-rank step in Appendix B.

- **`ClosedForm.lean`** — the closed-form solution. The paper obtains the normal equations $W'_k G = A$ (Eq. (14)) by setting the matrix derivative of $\mathcal{L}$ to zero and then right-multiplies by $G^{-1}$ (Eqs. (12)–(16)). The formalization proves the equivalent completed-square identity: for any $W$ and any $W'$ with $W' G = A$,

  $$L(W) = L(W') + \sum_i c_i \left\lVert (W - W') a_i \right\rVert_2^2,$$

  so $W'$ is a minimizer (each $c_i \ge 0$). When $G$ is invertible, $A G^{-1}$ satisfies $W' G = A$ and is the unique such solution, giving Eq. (2).

- **`Instances.lean`** — the paper's grouped formulas. `closedForm_twoGroup` is the refinement solution Eq. (2), with mapping terms weighted $1$ and preserving terms weighted $\lambda_1$. The paper states the fusion problem (Eq. (7)) has a closed-form solution "similar to Eq. (2)"; it is the same statement with the mapping targets set to $(W'_k + \Delta W_{k,i}) \mathbf{e}^f_j$, so it is formalized as an instance of `closedForm_twoGroup`. `closedForm_threeGroup` is the domain-specific variant Eq. (19), with a third weighted group ($\lambda_3$). The `gram_twoGroup` / `cross_twoGroup` lemmas confirm the specialized $G$ and $A$ reproduce the paper's two-sum expressions.

- **`CFIS.lean`** — the concept-focal importance sampling density $\xi(t) = \big(\sigma(\gamma(t - t_1)) - \sigma(\gamma(t - t_2))\big)/Z$ (Eq. (5)) is non-negative for $t_1 < t_2$, $\gamma > 0$, $Z > 0$, from monotonicity of the logistic sigmoid.

Each file's statements follow the paper's notation, and vectors/matrices are modeled as $\mathrm{Fin}\ d \to \mathbb{R}$ and $\mathrm{Matrix}\ (\mathrm{Fin}\ d_1)\ (\mathrm{Fin}\ d_2)\ \mathbb{R}$, with the squared norm taken coordinatewise. Full statements and the paper's own proof transcriptions are in `catalog.json`.
