import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse

/-!
# MACE closed-form cross-attention refinement — shared setup

Machine-checked formalization of the closed-form solution of MACE. See `catalog.json` in this
folder for the full inventory. MACE's self-contained provable mathematical content is one
closed-form-solution derivation (Appendix A, "Closed-Form Solution Proof"): the minimizer of a
regularized matrix least-squares objective, Eq. (1) → Eq. (2).

We abstract that objective to a single weighted least-squares problem and prove the closed form
once, in general (`ClosedForm.lean`); the paper's Eq. (2) (refinement), Eq. (7) (fusion) and the
`λ₃` domain-specific variant are then instances of the same theorem (`Instances.lean`). The Gram
matrix's positive (semi)definiteness — grounding the invertibility step — is `GramMatrix.lean`.

## Setup

The data of a weighted least-squares problem, indexed by a finite type `ι`. For each `i`:

* `a i : Fin d₂ → ℝ` — an input embedding (the paper's `eᶠᵢ` / `eᵖᵢ`);
* `b i : Fin d₁ → ℝ` — the target vector (`W_k eᵍᵢ` / `W_k eᵖᵢ`);
* `c i : ℝ`, `0 ≤ c i` — the term weight (`1` for the mapping terms, `λ₁` for the preserving terms).

The unknown is a matrix `W : Matrix (Fin d₁) (Fin d₂) ℝ` (the refined projection `W'_k`).
-/

namespace Mace

open Matrix
open scoped BigOperators

variable {ι : Type*} [Fintype ι] {d₁ d₂ : ℕ}

/-- Squared Euclidean norm of a vector, `‖v‖² = ∑ r, (v r)²`. Defined pointwise to keep the
algebra elementary (no `EuclideanSpace`/`Fin d → ℝ` defeq juggling). -/
def sqNorm {d : ℕ} (v : Fin d → ℝ) : ℝ := ∑ r, (v r) ^ 2

/-- The standard dot product `⟪u, v⟫ = ∑ r, u r * v r` on `Fin d → ℝ`. -/
def dotp {d : ℕ} (u v : Fin d → ℝ) : ℝ := ∑ r, u r * v r

@[simp] lemma sqNorm_nonneg {d : ℕ} (v : Fin d → ℝ) : 0 ≤ sqNorm v :=
  Finset.sum_nonneg fun r _ => sq_nonneg (v r)

lemma sqNorm_eq_dotp {d : ℕ} (v : Fin d → ℝ) : sqNorm v = dotp v v := by
  simp only [sqNorm, dotp, sq]

/-- Polarization: `‖u + v‖² = ‖u‖² + 2⟪u, v⟫ + ‖v‖²`. -/
lemma sqNorm_add {d : ℕ} (u v : Fin d → ℝ) :
    sqNorm (u + v) = sqNorm u + 2 * dotp u v + sqNorm v := by
  simp only [sqNorm, dotp, Pi.add_apply, add_sq]
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib, Finset.mul_sum]
  ring_nf

/-- The MACE least-squares objective `L(W) = ∑ᵢ cᵢ ‖W·aᵢ − bᵢ‖²` (paper Eq. (1)). -/
def objective (a : ι → Fin d₂ → ℝ) (b : ι → Fin d₁ → ℝ) (c : ι → ℝ)
    (W : Matrix (Fin d₁) (Fin d₂) ℝ) : ℝ :=
  ∑ i, c i * sqNorm (W.mulVec (a i) - b i)

/-- The Gram matrix `G = ∑ᵢ cᵢ aᵢ aᵢᵀ` (paper's `∑ eᶠᵢ (eᶠᵢ)ᵀ + λ₁ ∑ eᵖᵢ (eᵖᵢ)ᵀ`). -/
def gram (a : ι → Fin d₂ → ℝ) (c : ι → ℝ) : Matrix (Fin d₂) (Fin d₂) ℝ :=
  ∑ i, c i • vecMulVec (a i) (a i)

/-- The cross matrix `A = ∑ᵢ cᵢ bᵢ aᵢᵀ` (paper's `∑ W_k eᵍᵢ (eᶠᵢ)ᵀ + λ₁ ∑ W_k eᵖᵢ (eᵖᵢ)ᵀ`). -/
def cross (a : ι → Fin d₂ → ℝ) (b : ι → Fin d₁ → ℝ) (c : ι → ℝ) :
    Matrix (Fin d₁) (Fin d₂) ℝ :=
  ∑ i, c i • vecMulVec (b i) (a i)

section entries
variable (a : ι → Fin d₂ → ℝ) (b : ι → Fin d₁ → ℝ) (c : ι → ℝ)

@[simp] lemma gram_apply (j k : Fin d₂) : gram a c j k = ∑ i, c i * (a i j * a i k) := by
  simp [gram, Matrix.sum_apply, Matrix.smul_apply, vecMulVec_apply]

@[simp] lemma cross_apply (r : Fin d₁) (k : Fin d₂) :
    cross a b c r k = ∑ i, c i * (b i r * a i k) := by
  simp [cross, Matrix.sum_apply, Matrix.smul_apply, vecMulVec_apply]

/-- `(W · G)` entrywise in terms of the data: `(W G)_{r k} = ∑ᵢ cᵢ (W·aᵢ)_r (aᵢ)_k`. -/
lemma mul_gram_apply (W : Matrix (Fin d₁) (Fin d₂) ℝ) (r : Fin d₁) (k : Fin d₂) :
    (W * gram a c) r k = ∑ i, c i * ((W.mulVec (a i)) r * a i k) := by
  simp only [Matrix.mul_apply, gram_apply, Matrix.mulVec, dotProduct, Finset.mul_sum,
    Finset.sum_mul]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl (fun i _ => ?_)
  refine Finset.sum_congr rfl (fun j _ => ?_)
  ring

end entries

end Mace
