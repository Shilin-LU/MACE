#!/usr/bin/env bash
# MACE Lean formalization — the "really done" verification gate.
#
# A green `lake build` only proves the code compiles; it says nothing about whether the proofs
# are honest. This script runs the full gate in one shot:
#   1. `lake build`                       — everything compiles;
#   2. `grep -rn sorry`                   — must find NOTHING (no `sorry`/`admit`);
#   3. `#print axioms` on every headline  — must be exactly [propext, Classical.choice, Quot.sound]
#                                           (no `sorryAx`, no extra axioms).
#
# Usage: run from a build directory that has this project's modules + a built Mathlib
#        (see PLAN.md for the disk-frugal build-outside-repo setup), e.g.
#          cp *.lean <builddir>/ && cd <builddir> && bash verify.sh
#        or point it at the repo copy if a full local build is available.
set -uo pipefail

echo "=== 1. lake build ==="
if ! lake build; then echo "FAIL: build failed"; exit 1; fi

echo "=== 2. grep for sorry / admit (must be empty) ==="
if grep -rniE '\b(sorry|admit)\b' ./*.lean; then
  echo "FAIL: found sorry/admit"; exit 1
else
  echo "OK: no sorry/admit"
fi

echo "=== 3. #print axioms on headline results ==="
HEADLINES=(
  Mace.closedForm_isMinimizer     # C1  — closed form is a global minimizer (paper Eq. 2)
  Mace.objective_completeSquare   # C1  — the complete-the-square identity
  Mace.normalEq_unique            # C1  — uniqueness of the normal-equation solution
  Mace.gram_posSemidef            # C2  — Gram matrix positive semidefinite
  Mace.gram_posDef                # C2  — Gram matrix positive definite under non-degeneracy
  Mace.gram_isUnit_det            # C2  — hence invertible
  Mace.closedForm_twoGroup        # C3  — Eq. 2 refinement / Eq. 7 fusion instance
  Mace.closedForm_threeGroup      # C4  — domain-specific λ₃ variant instance
  Mace.cfisDensity_nonneg         # C5  — CFIS density non-negative
)
{ echo 'import Mace'; for h in "${HEADLINES[@]}"; do echo "#print axioms $h"; done; } > _axioms_check.lean
OUT=$(lake env lean _axioms_check.lean 2>&1)
rm -f _axioms_check.lean
echo "$OUT"
# Enforce EXACTLY the three standard classical axioms (not just "no sorryAx").
# 1. no sorryAx / no compile error;
# 2. every axiom token printed inside a [...] list is one of the three allowed;
# 3. one report line per headline (guards against a typo'd/renamed decl silently reporting nothing).
BAD=$(echo "$OUT" | grep -oE '\[[^]]*\]' | tr -d '[]' | tr ',' '\n' \
      | sed 's/^[[:space:]]*//; s/[[:space:]]*$//' | grep -v '^$' \
      | grep -vxE 'propext|Classical\.choice|Quot\.sound' || true)
NREPORT=$(echo "$OUT" | grep -cE "^'Mace\.")
if echo "$OUT" | grep -qiE 'sorryAx|error'; then
  echo "FAIL: sorryAx or error in axiom output"; exit 1
fi
if [ -n "$BAD" ]; then
  echo "FAIL: non-standard axiom(s) present:"; echo "$BAD"; exit 1
fi
if [ "$NREPORT" -ne "${#HEADLINES[@]}" ]; then
  echo "FAIL: expected ${#HEADLINES[@]} '#print axioms' report lines, got $NREPORT"; exit 1
fi
echo "=== ALL CHECKS PASSED ==="
