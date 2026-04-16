"""Prompt collections used by this project's experiments.

These are project-specific data, not framework infrastructure — the
gemma4_mlx_interp framework supplies the Prompt / PromptSet classes; this
package supplies the specific instances we've curated.

  FACTUAL_15           — used by step_01 through step_09
  BIG_SWEEP_96         — 12 categories x 8 prompts (step_12)
  STRESS_TEMPLATE_VAR  — 4 phrasings * 4 countries (step_13)
  STRESS_CROSS_LINGUAL — 5 languages * 4 countries (step_13)
  STRESS_CREATIVE      — 8 subjective / metaphorical prompts (step_13)
"""

from .big_sweep import BIG_SWEEP_96
from .factual import FACTUAL_15
from .stress import STRESS_CREATIVE, STRESS_CROSS_LINGUAL, STRESS_TEMPLATE_VAR

__all__ = [
    "FACTUAL_15",
    "BIG_SWEEP_96",
    "STRESS_TEMPLATE_VAR",
    "STRESS_CROSS_LINGUAL",
    "STRESS_CREATIVE",
]
