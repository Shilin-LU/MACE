import ClosedForm

/-!
# MACE — the paper's three closed forms as instances of C1 (catalog items C3, C4)

The general theorem `closedForm_isMinimizer` (C1) is stated over an arbitrary finite index set. The
paper's formulas group the data into weighted blocks (mapping terms with weight `1`, preserving
terms with weight `λ₁`, and — in the domain-specific variant — a third block with weight `λ₃`). We
realize the grouping with `Sum` types and show:

* **Eq. (1) / Eq. (2)** (Section 3.1, refinement) — the two-group objective and its closed-form
  minimizer (`objective_twoGroup`, `closedForm_twoGroup`);
* **Eq. (7)** (Section 3.3, LoRA fusion) — the paper states this "has a closed-form solution as
  well, similar to Eq. (2)". It is the same two-group statement with the mapping targets `bf`
  replaced by `(W'_k + ΔW_{k,i})·eⱼᶠ`; since `closedForm_twoGroup` takes the targets as arbitrary
  data, Eq. (7) is literally an instance of it (C3);
* the **domain-specific `λ₃` variant** (Appendix A) — three weighted groups
  (`closedForm_threeGroup`) (C4).

The Gram/cross split lemmas confirm these specializations reproduce the paper's own two- and
three-sum expressions verbatim.
-/

namespace Mace

open Matrix
open scoped BigOperators

variable {ιf ιp ιq : Type*} [Fintype ιf] [Fintype ιp] [Fintype ιq] {d₁ d₂ : ℕ}

/-! ## Two groups — paper Eq. (1)/(2) (refinement) and Eq. (7) (fusion) -/

section twoGroup
variable (af : ιf → Fin d₂ → ℝ) (bf : ιf → Fin d₁ → ℝ)
  (ap : ιp → Fin d₂ → ℝ) (bp : ιp → Fin d₁ → ℝ) (lam : ℝ)

/-- Weights for the paper's two-group problem: `1` on the mapping terms, `λ` on the preserving. -/
def twoWeights : ιf ⊕ ιp → ℝ := Sum.elim (fun _ => 1) (fun _ => lam)

/-- The two-group objective is the paper's Eq. (1):
`L(W) = ∑ᵢ ‖W·afᵢ − bfᵢ‖² + λ ∑ᵢ ‖W·apᵢ − bpᵢ‖²`. -/
theorem objective_twoGroup (W : Matrix (Fin d₁) (Fin d₂) ℝ) :
    objective (Sum.elim af ap) (Sum.elim bf bp) (twoWeights lam) W
      = (∑ i, sqNorm (W.mulVec (af i) - bf i))
        + lam * ∑ i, sqNorm (W.mulVec (ap i) - bp i) := by
  rw [objective, twoWeights, Fintype.sum_sum_type]
  simp only [Sum.elim_inl, Sum.elim_inr, one_mul, Finset.mul_sum]

/-- The two-group Gram matrix is the paper's `G = ∑ᵢ afᵢ afᵢᵀ + λ ∑ᵢ apᵢ apᵢᵀ`. -/
theorem gram_twoGroup :
    gram (Sum.elim af ap) (twoWeights lam)
      = (∑ i, vecMulVec (af i) (af i)) + lam • ∑ i, vecMulVec (ap i) (ap i) := by
  rw [gram, twoWeights, Fintype.sum_sum_type]
  simp only [Sum.elim_inl, Sum.elim_inr, one_smul, Finset.smul_sum]

/-- The two-group cross matrix is the paper's `A = ∑ᵢ bfᵢ afᵢᵀ + λ ∑ᵢ bpᵢ apᵢᵀ`. -/
theorem cross_twoGroup :
    cross (Sum.elim af ap) (Sum.elim bf bp) (twoWeights lam)
      = (∑ i, vecMulVec (bf i) (af i)) + lam • ∑ i, vecMulVec (bp i) (ap i) := by
  rw [cross, twoWeights, Fintype.sum_sum_type]
  simp only [Sum.elim_inl, Sum.elim_inr, one_smul, Finset.smul_sum]

/-- **C1 specialized to Eq. (2), and C3 (Eq. (7) fusion).** With `λ ≥ 0` and the two-group Gram
matrix invertible, `A·G⁻¹` minimizes the two-group objective — MACE's Eq. (2). The fusion Eq. (7)
is the same statement with different mapping targets `bf`, hence an instance of this theorem. -/
theorem closedForm_twoGroup (hlam : 0 ≤ lam)
    (hG : IsUnit (gram (Sum.elim af ap) (twoWeights lam)).det)
    (W : Matrix (Fin d₁) (Fin d₂) ℝ) :
    objective (Sum.elim af ap) (Sum.elim bf bp) (twoWeights lam)
        (cross (Sum.elim af ap) (Sum.elim bf bp) (twoWeights lam) *
          (gram (Sum.elim af ap) (twoWeights lam))⁻¹)
      ≤ objective (Sum.elim af ap) (Sum.elim bf bp) (twoWeights lam) W := by
  refine closedForm_isMinimizer (Sum.elim af ap) (Sum.elim bf bp) (twoWeights lam) ?_ hG W
  rintro (i | i) <;> simp [twoWeights, hlam, zero_le_one]

end twoGroup

/-! ## Three groups — paper's domain-specific `λ₃` variant (Appendix A) -/

section threeGroup
variable (af : ιf → Fin d₂ → ℝ) (bf : ιf → Fin d₁ → ℝ)
  (ap : ιp → Fin d₂ → ℝ) (bp : ιp → Fin d₁ → ℝ)
  (aq : ιq → Fin d₂ → ℝ) (bq : ιq → Fin d₁ → ℝ) (lam1 lam3 : ℝ)

/-- Weights for the three-group (general `λ₁` + domain-specific `λ₃`) problem. -/
def threeWeights : ιf ⊕ ιp ⊕ ιq → ℝ :=
  Sum.elim (fun _ => 1) (Sum.elim (fun _ => lam1) (fun _ => lam3))

/-- The three-group objective is the paper's `λ₃`-variant of Eq. (1). -/
theorem objective_threeGroup (W : Matrix (Fin d₁) (Fin d₂) ℝ) :
    objective (Sum.elim af (Sum.elim ap aq)) (Sum.elim bf (Sum.elim bp bq))
        (threeWeights lam1 lam3) W
      = (∑ i, sqNorm (W.mulVec (af i) - bf i))
        + lam1 * (∑ i, sqNorm (W.mulVec (ap i) - bp i))
        + lam3 * ∑ i, sqNorm (W.mulVec (aq i) - bq i) := by
  rw [objective, threeWeights, Fintype.sum_sum_type, Fintype.sum_sum_type]
  simp only [Sum.elim_inl, Sum.elim_inr, one_mul, Finset.mul_sum, add_assoc]

/-- **C4.** The domain-specific `λ₃` variant of Eq. (2): with `λ₁, λ₃ ≥ 0` and the three-group Gram
matrix invertible, `A·G⁻¹` minimizes the three-group objective. -/
theorem closedForm_threeGroup (hlam1 : 0 ≤ lam1) (hlam3 : 0 ≤ lam3)
    (hG : IsUnit (gram (Sum.elim af (Sum.elim ap aq)) (threeWeights lam1 lam3)).det)
    (W : Matrix (Fin d₁) (Fin d₂) ℝ) :
    objective (Sum.elim af (Sum.elim ap aq)) (Sum.elim bf (Sum.elim bp bq))
        (threeWeights lam1 lam3)
        (cross (Sum.elim af (Sum.elim ap aq)) (Sum.elim bf (Sum.elim bp bq))
            (threeWeights lam1 lam3) *
          (gram (Sum.elim af (Sum.elim ap aq)) (threeWeights lam1 lam3))⁻¹)
      ≤ objective (Sum.elim af (Sum.elim ap aq)) (Sum.elim bf (Sum.elim bp bq))
        (threeWeights lam1 lam3) W := by
  refine closedForm_isMinimizer (Sum.elim af (Sum.elim ap aq)) (Sum.elim bf (Sum.elim bp bq))
    (threeWeights lam1 lam3) ?_ hG W
  rintro (i | i | i) <;> simp [threeWeights, hlam1, hlam3, zero_le_one]

end threeGroup

end Mace
