# Lean 4 / Mathlib formalization of MACE's closed-form solution

Machine-checked (Lean 4 + [Mathlib](https://github.com/leanprover-community/mathlib4)) proofs of the mathematical content of

> S. Lu, Z. Wang, L. Li, Y. Liu, A. W.-K. Kong. *MACE: Mass Concept Erasure in Diffusion Models.* CVPR 2024. [arXiv:2403.06135](https://arxiv.org/abs/2403.06135).

MACE is a framework paper; its self-contained mathematical content is the closed-form solution derived in Appendix A ("Closed-Form Solution Proof"): the minimizer of the regularized least-squares objective used for the cross-attention refinement (Eq. (1) → Eq. (2)), together with the positive-(semi)definiteness argument that justifies the matrix inversion. The multi-LoRA fusion closed form (Eq. (6)) and the domain-specific `λ₃` variant (Appendix A) are the same least-squares problem instantiated differently; they are derived here as instances of a single general theorem.

## Status

All results build with zero `sorry`, and `#print axioms` reports only the three standard classical axioms (`propext`, `Classical.choice`, `Quot.sound`) for every headline theorem.

| Result | File | Statement |
|--------|------|-----------|
| Closed-form minimizer (Eq. (2)) | `ClosedForm.lean` | `A·G⁻¹` is a global minimizer of `L(W) = ∑ᵢ cᵢ ‖W·aᵢ − bᵢ‖²` when `G` is invertible |
| Gram matrix positive (semi)definite | `GramMatrix.lean` | `xᵀGx = ∑ᵢ cᵢ(x·aᵢ)² ≥ 0`; positive definite ⇒ invertible under a non-degeneracy condition |
| Refinement / fusion (Eq. (2), Eq. (6)) | `Instances.lean` | two-group instance of the general closed form |
| Domain-specific `λ₃` variant | `Instances.lean` | three-group instance of the general closed form |
| CFIS sampling density (Eq. (5)) | `CFIS.lean` | the density `ξ(t)` is non-negative |

## How to build

Requires a Lean 4 toolchain (`elan`/`lake`). Run `lake build` in this folder. The pinned Mathlib version is in `lakefile.toml` / `lake-manifest.json`. `verify.sh` runs the full check (build + a `sorry` sweep + `#print axioms` on every headline result).

## What is proved

The objective is stated in general form as a weighted least-squares problem over a finite index set: for data `aᵢ ∈ ℝ^{d₂}` (inputs), `bᵢ ∈ ℝ^{d₁}` (targets), and weights `cᵢ ≥ 0`,
`L(W) = ∑ᵢ cᵢ ‖W·aᵢ − bᵢ‖²`, with Gram matrix `G = ∑ᵢ cᵢ aᵢ aᵢᵀ` and cross matrix `A = ∑ᵢ cᵢ bᵢ aᵢᵀ`.

- **`Basic.lean`** — the objective `L`, the Gram matrix `G`, and the cross matrix `A`, with the entrywise identities used downstream.

- **`GramMatrix.lean`** — the quadratic-form identity `xᵀGx = ∑ᵢ cᵢ (x·aᵢ)²`, from which `G` is positive semidefinite (weights `cᵢ ≥ 0`), and positive definite — hence `det G` is a unit and `G` is invertible — under the condition that no nonzero `x` is orthogonal to every `aᵢ` carrying positive weight. This is the precise sufficient condition for the full-rank step in Appendix A.

- **`ClosedForm.lean`** — the closed-form solution. The proof completes the square: for any `W` and any `W'` satisfying the normal equations `W'·G = A`,
  `L(W) = L(W') + ∑ᵢ cᵢ ‖(W − W')·aᵢ‖²`,
  so `W'` is a global minimizer (each `cᵢ ≥ 0`). The paper obtains the normal equations by setting the matrix derivative of `L` to zero; the completed-square identity establishes global minimality directly and supplies the minimality that the derivative-zero step leaves implicit. When `G` is invertible, `A·G⁻¹` satisfies `W'·G = A` and is the unique such solution, giving Eq. (2).

- **`Instances.lean`** — the paper's grouped formulas as instances of the general theorem. `closedForm_twoGroup` is the refinement solution Eq. (2) (mapping terms weighted `1`, preserving terms weighted `λ₁`); the fusion solution Eq. (6) is the same statement with the mapping targets set to `(W'_k + ΔW_{k,i})·eⱼᶠ`, so it is an instance of the same theorem. `closedForm_threeGroup` is the domain-specific variant with a third weighted group (`λ₃`). The `gram_twoGroup` / `cross_twoGroup` lemmas confirm the specialized `G` and `A` reproduce the paper's two-sum expressions.

- **`CFIS.lean`** — the concept-focal importance sampling density `ξ(t) = (σ(γ(t−t₁)) − σ(γ(t−t₂)))/Z` (Eq. (5)) is non-negative for `t₁ < t₂`, `γ > 0`, `Z > 0`, from monotonicity of the logistic sigmoid. The paper introduces `ξ` as a design choice; this records its non-negativity.

## Notes on the formalization

- The objective is proved in general (arbitrary finite index set, arbitrary per-term weights `cᵢ ≥ 0`), so the three closed forms in the paper are instances of one theorem rather than three separate derivations.
- The invertibility of the Gram matrix is stated as an explicit non-degeneracy hypothesis (no nonzero vector is orthogonal to all positively-weighted inputs), which is the precise sufficient condition behind Appendix A's remark that the matrix is "in general" positive definite.
- Vectors and matrices use `Fin d → ℝ` and `Matrix (Fin d₁) (Fin d₂) ℝ`; the squared norm is defined coordinatewise.
