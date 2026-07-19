import Basic
import GramMatrix

/-!
# MACE — closed-form solution of the least-squares refinement objective (catalog item C1)

The mathematical heart of MACE (Appendix A, "Closed-Form Solution Proof"): the regularized matrix
least-squares objective `L(W) = ∑ᵢ cᵢ ‖W·aᵢ − bᵢ‖²` is minimized by `W = A·G⁻¹`, where
`G = ∑ᵢ cᵢ aᵢ aᵢᵀ` (the Gram matrix) and `A = ∑ᵢ cᵢ bᵢ aᵢᵀ` — the normal-equation /
ridge-regression / linear-associative-memory closed form (paper Eq. (1) → Eq. (2)).

The paper differentiates `L` and sets the derivative to zero, obtaining the normal equations
`W·G = A`, then right-multiplies by `G⁻¹`. We instead **complete the square**: for any `W` and any
`W'` satisfying the normal equations `W'·G = A`,
`L(W) = L(W') + ∑ᵢ cᵢ ‖(W − W')·aᵢ‖²`,
so `W'` is a *global* minimizer (each `cᵢ ≥ 0`). This subsumes the paper's stationarity argument:
it needs no matrix calculus, and it establishes global minimality directly rather than relying on
implicit convexity. Invertibility of `G` (from `GramMatrix.lean`, item C2) then makes `A·G⁻¹` the
unique normal-equation solution, hence the minimizer.
-/

namespace Mace

open Matrix
open scoped BigOperators

variable {ι : Type*} [Fintype ι] {d₁ d₂ : ℕ}
variable (a : ι → Fin d₂ → ℝ) (b : ι → Fin d₁ → ℝ) (c : ι → ℝ)

/-- The cross term in the completed square vanishes when `W'` satisfies the normal equations
`W'·G = A`. This is where the normal equations enter. -/
lemma crossTerm_eq_zero (W W' : Matrix (Fin d₁) (Fin d₂) ℝ)
    (hW' : W' * gram a c = cross a b c) :
    ∑ i, c i * dotp (W'.mulVec (a i) - b i) ((W - W').mulVec (a i)) = 0 := by
  -- Componentwise form of the normal equations.
  have hnorm : ∀ r k, ∑ i, c i * ((W'.mulVec (a i) r - b i r) * a i k) = 0 := by
    intro r k
    have h1 := congrFun (congrFun hW' r) k
    rw [mul_gram_apply, cross_apply] at h1
    calc ∑ i, c i * ((W'.mulVec (a i) r - b i r) * a i k)
        = (∑ i, c i * (W'.mulVec (a i) r * a i k)) - ∑ i, c i * (b i r * a i k) := by
          rw [← Finset.sum_sub_distrib]; exact Finset.sum_congr rfl fun i _ => by ring
      _ = 0 := by rw [h1, sub_self]
  -- For each output coordinate `r`, the inner sum vanishes.
  have inner : ∀ r, ∑ i, c i * ((W'.mulVec (a i) - b i) r * (W - W').mulVec (a i) r) = 0 := by
    intro r
    have hterm : ∀ i, c i * ((W'.mulVec (a i) - b i) r * (W - W').mulVec (a i) r)
        = ∑ k, (W - W') r k * (c i * ((W'.mulVec (a i) r - b i r) * a i k)) := by
      intro i
      rw [show (W - W').mulVec (a i) r = ∑ k, (W - W') r k * a i k from rfl,
        Finset.mul_sum, Finset.mul_sum]
      refine Finset.sum_congr rfl fun k _ => ?_
      simp only [Pi.sub_apply]; ring
    rw [Finset.sum_congr rfl (fun i _ => hterm i), Finset.sum_comm]
    refine Finset.sum_eq_zero fun k _ => ?_
    rw [← Finset.mul_sum, hnorm r k, mul_zero]
  simp only [dotp, Finset.mul_sum]
  rw [Finset.sum_comm]
  exact Finset.sum_eq_zero fun r _ => inner r

/-- **Completing the square.** If `W'` satisfies the normal equations `W'·G = A`, then for every
`W`, `L(W) = L(W') + ∑ᵢ cᵢ ‖(W − W')·aᵢ‖²`. -/
theorem objective_completeSquare (W W' : Matrix (Fin d₁) (Fin d₂) ℝ)
    (hW' : W' * gram a c = cross a b c) :
    objective a b c W = objective a b c W' + ∑ i, c i * sqNorm ((W - W').mulVec (a i)) := by
  have hdecomp : ∀ i, W.mulVec (a i) - b i
      = (W'.mulVec (a i) - b i) + (W - W').mulVec (a i) := by
    intro i; ext r
    simp only [Pi.add_apply, Pi.sub_apply, Matrix.sub_mulVec]; ring
  have hexpand : ∀ i, c i * sqNorm (W.mulVec (a i) - b i)
      = c i * sqNorm (W'.mulVec (a i) - b i)
        + c i * (2 * dotp (W'.mulVec (a i) - b i) ((W - W').mulVec (a i)))
        + c i * sqNorm ((W - W').mulVec (a i)) := by
    intro i; rw [hdecomp i, sqNorm_add]; ring
  have hmid : ∑ i, c i * (2 * dotp (W'.mulVec (a i) - b i) ((W - W').mulVec (a i))) = 0 := by
    have h2 : ∀ i, c i * (2 * dotp (W'.mulVec (a i) - b i) ((W - W').mulVec (a i)))
        = 2 * (c i * dotp (W'.mulVec (a i) - b i) ((W - W').mulVec (a i))) := fun i => by ring
    rw [Finset.sum_congr rfl (fun i _ => h2 i), ← Finset.mul_sum,
      crossTerm_eq_zero a b c W W' hW', mul_zero]
  calc objective a b c W
      = ∑ i, (c i * sqNorm (W'.mulVec (a i) - b i)
          + c i * (2 * dotp (W'.mulVec (a i) - b i) ((W - W').mulVec (a i)))
          + c i * sqNorm ((W - W').mulVec (a i))) := by
        rw [objective]; exact Finset.sum_congr rfl fun i _ => hexpand i
    _ = objective a b c W' + ∑ i, c i * sqNorm ((W - W').mulVec (a i)) := by
        rw [Finset.sum_add_distrib, Finset.sum_add_distrib, hmid, add_zero]; rfl

/-- **Minimality.** Any `W'` satisfying the normal equations `W'·G = A` is a global minimizer. -/
theorem isMinimizer (hc : ∀ i, 0 ≤ c i) (W' : Matrix (Fin d₁) (Fin d₂) ℝ)
    (hW' : W' * gram a c = cross a b c) (W : Matrix (Fin d₁) (Fin d₂) ℝ) :
    objective a b c W' ≤ objective a b c W := by
  rw [objective_completeSquare a b c W W' hW']
  have : 0 ≤ ∑ i, c i * sqNorm ((W - W').mulVec (a i)) :=
    Finset.sum_nonneg fun i _ => mul_nonneg (hc i) (sqNorm_nonneg _)
  linarith

/-- The closed form `A·G⁻¹` satisfies the normal equations when `G` is invertible. -/
theorem closedForm_normalEq (hG : IsUnit (gram a c).det) :
    cross a b c * (gram a c)⁻¹ * gram a c = cross a b c := by
  rw [Matrix.mul_assoc, Matrix.nonsing_inv_mul _ hG, Matrix.mul_one]

/-- **C1 (main theorem).** When the Gram matrix is invertible, `W = A·G⁻¹` is a global minimizer of
the MACE least-squares objective — the paper's Eq. (2). -/
theorem closedForm_isMinimizer (hc : ∀ i, 0 ≤ c i) (hG : IsUnit (gram a c).det)
    (W : Matrix (Fin d₁) (Fin d₂) ℝ) :
    objective a b c (cross a b c * (gram a c)⁻¹) ≤ objective a b c W :=
  isMinimizer a b c hc _ (closedForm_normalEq a b c hG) W

/-- The normal-equation solution is unique when `G` is invertible: any `W` with `W·G = A` equals the
closed form `A·G⁻¹`. So `A·G⁻¹` is the unique stationary point (the `=` in the paper's Eq. (2)). -/
theorem normalEq_unique (hG : IsUnit (gram a c).det)
    (W : Matrix (Fin d₁) (Fin d₂) ℝ) (hW : W * gram a c = cross a b c) :
    W = cross a b c * (gram a c)⁻¹ := by
  have h := congrArg (· * (gram a c)⁻¹) hW
  simpa [Matrix.mul_assoc, Matrix.mul_nonsing_inv _ hG] using h

end Mace
