import Mathlib.Data.Real.Basic

-- We verify the algebraic equivalence used in STEP 1.

example (a b : ℝ) : (a^2 < b^2) ↔ (0 < (b - a) * (b + a)) := by
  constructor
  · intro h
    have h' : b^2 - a^2 > 0 := sub_pos.mpr h
    -- rewrite difference of squares
    have : b^2 - a^2 = (b - a) * (b + a) := by
      ring
    simpa [this] using h'
  · intro h
    have h' : b^2 - a^2 > 0 := by
      -- rewrite difference of squares
      have : b^2 - a^2 = (b - a) * (b + a) := by
        ring
      simpa [this] using h
    -- convert back to a^2 < b^2
    exact sub_pos.mp h
