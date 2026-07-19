import Mathlib.Analysis.SpecialFunctions.Exp

/-!
# MACE — the concept-focal importance sampling (CFIS) density is non-negative (catalog item C5)

A minor, self-contained property of the timestep sampling density MACE uses when training the LoRA
modules (Section 3.2, Eq. (5); Appendix C):
`ξ(t) = (σ(γ(t − t₁)) − σ(γ(t − t₂))) / Z`, with `σ` the logistic sigmoid, `t₁ < t₂`, `γ > 0`,
`Z > 0`. The paper defines `ξ` as a design choice and does not prove it is a density; here we record
that it is non-negative, which follows immediately from the monotonicity of `σ` and `t₁ < t₂`.
-/

namespace Mace

open Real

/-- The logistic sigmoid `σ(x) = 1/(1 + e^{-x})`. -/
noncomputable def sigmoid (x : ℝ) : ℝ := 1 / (1 + Real.exp (-x))

lemma sigmoid_pos (x : ℝ) : 0 < sigmoid x := by
  unfold sigmoid; positivity

/-- The logistic sigmoid is monotone increasing. -/
lemma sigmoid_mono : Monotone sigmoid := by
  intro x y hxy
  have hexp : Real.exp (-y) ≤ Real.exp (-x) := Real.exp_le_exp.mpr (by linarith)
  unfold sigmoid
  exact one_div_le_one_div_of_le (by positivity) (by linarith)

/-- The CFIS sampling density `ξ(t) = (σ(γ(t−t₁)) − σ(γ(t−t₂)))/Z` (paper Eq. (5)). -/
noncomputable def cfisDensity (t₁ t₂ γ Z t : ℝ) : ℝ :=
  (sigmoid (γ * (t - t₁)) - sigmoid (γ * (t - t₂))) / Z

/-- **C5.** The CFIS density is non-negative when `t₁ < t₂`, `γ > 0`, `Z > 0`. -/
theorem cfisDensity_nonneg (t₁ t₂ γ Z t : ℝ)
    (ht : t₁ < t₂) (hγ : 0 < γ) (hZ : 0 < Z) :
    0 ≤ cfisDensity t₁ t₂ γ Z t := by
  unfold cfisDensity
  refine div_nonneg ?_ hZ.le
  rw [sub_nonneg]
  exact sigmoid_mono (mul_le_mul_of_nonneg_left (by linarith) hγ.le)

end Mace
