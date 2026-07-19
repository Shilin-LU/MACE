import Basic

/-!
# MACE — the Gram matrix is positive (semi)definite and invertible (catalog item C2)

Grounds the invertibility step of the closed-form solution (`ClosedForm.lean`). This is the
argument in MACE's Appendix A: for any `x`,
`xᵀ G x = ∑ᵢ cᵢ (x · aᵢ)² ≥ 0`, so the Gram matrix `G = ∑ᵢ cᵢ aᵢ aᵢᵀ` is positive semidefinite;
and it is positive definite (hence invertible) exactly when no nonzero `x` is orthogonal to every
`aᵢ` carrying positive weight — the precise sufficient condition behind the paper's informal "it is
highly improbable that all terms vanish ⇒ in general positive definite" remark.
-/

namespace Mace

open Matrix
open scoped BigOperators

variable {ι : Type*} [Fintype ι] {d₂ : ℕ}

variable (a : ι → Fin d₂ → ℝ) (c : ι → ℝ)

/-- `(G x)_j = ∑ᵢ cᵢ (aᵢ)_j (aᵢ · x)`. -/
lemma gram_mulVec_apply (x : Fin d₂ → ℝ) (j : Fin d₂) :
    (gram a c *ᵥ x) j = ∑ i, c i * (a i j * (a i ⬝ᵥ x)) := by
  simp only [Matrix.mulVec, dotProduct, gram_apply, Finset.mul_sum, Finset.sum_mul]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun i _ => ?_
  refine Finset.sum_congr rfl fun k _ => ?_
  ring

/-- The paper's quadratic-form identity: `xᵀ G x = ∑ᵢ cᵢ (x · aᵢ)²`. -/
lemma dotProduct_gram_mulVec (x : Fin d₂ → ℝ) :
    x ⬝ᵥ (gram a c *ᵥ x) = ∑ i, c i * (x ⬝ᵥ a i) ^ 2 := by
  have key : ∀ i, x ⬝ᵥ ((c i • vecMulVec (a i) (a i)) *ᵥ x) = c i * (x ⬝ᵥ a i) ^ 2 := by
    intro i
    simp only [Matrix.mulVec, dotProduct, Matrix.smul_apply, vecMulVec_apply, smul_eq_mul, pow_two]
    rw [Finset.sum_mul_sum]
    simp only [Finset.mul_sum]
    refine Finset.sum_congr rfl fun j _ => ?_
    refine Finset.sum_congr rfl fun k _ => ?_
    ring
  rw [gram, Matrix.sum_mulVec, dotProduct_sum]
  exact Finset.sum_congr rfl fun i _ => key i

/-- `G` is symmetric (Hermitian over `ℝ`). -/
lemma gram_isHermitian : (gram a c).IsHermitian := by
  ext j k
  rw [Matrix.conjTranspose_apply, gram_apply, gram_apply, star_trivial]
  exact Finset.sum_congr rfl fun i _ => by ring

/-- **C2 (PSD).** `G = ∑ᵢ cᵢ aᵢ aᵢᵀ` is positive semidefinite when every weight `cᵢ ≥ 0`. -/
theorem gram_posSemidef (hc : ∀ i, 0 ≤ c i) : (gram a c).PosSemidef := by
  refine PosSemidef.of_dotProduct_mulVec_nonneg (gram_isHermitian a c) (fun x => ?_)
  rw [star_trivial, dotProduct_gram_mulVec]
  exact Finset.sum_nonneg fun i _ => mul_nonneg (hc i) (sq_nonneg _)

/-- **C2 (PD).** `G` is positive definite — hence invertible — when, in addition, no nonzero `x`
is orthogonal to every `aᵢ` carrying positive weight. This is the precise sufficient condition
behind the paper's informal full-rank argument. -/
theorem gram_posDef (hc : ∀ i, 0 ≤ c i)
    (hspan : ∀ x : Fin d₂ → ℝ, x ≠ 0 → ∃ i, 0 < c i ∧ x ⬝ᵥ a i ≠ 0) :
    (gram a c).PosDef := by
  refine PosDef.of_dotProduct_mulVec_pos (gram_isHermitian a c) (fun x hx => ?_)
  rw [star_trivial, dotProduct_gram_mulVec]
  obtain ⟨i₀, hci₀, hxi₀⟩ := hspan x hx
  refine Finset.sum_pos' (fun i _ => mul_nonneg (hc i) (sq_nonneg _)) ⟨i₀, Finset.mem_univ _, ?_⟩
  exact mul_pos hci₀ (lt_of_le_of_ne (sq_nonneg _) (Ne.symm (pow_ne_zero 2 hxi₀)))

/-- `det G` is a unit (so `G` is invertible), under the positive-definiteness hypothesis. -/
theorem gram_isUnit_det (hc : ∀ i, 0 ≤ c i)
    (hspan : ∀ x : Fin d₂ → ℝ, x ≠ 0 → ∃ i, 0 < c i ∧ x ⬝ᵥ a i ≠ 0) :
    IsUnit (gram a c).det :=
  Matrix.isUnit_iff_isUnit_det _ |>.mp (gram_posDef a c hc hspan).isUnit

end Mace
