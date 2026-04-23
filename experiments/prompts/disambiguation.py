"""Disambiguation prompt sets for the operation-vs-word-presence experiment.

A 2x2 design varying:
  - operation-type:        capital-lookup (A) vs string-letter-counting (B)
  - 'capital' word presence: present (1) vs absent (2)

= 4 prompt sets, 8 prompts each.

The point: do mid-layer subject-position activations cluster by
OPERATION-TYPE (the factorization claim) or by surface-token PRESENCE
(the confound)? If A1+A2 cluster vs B1+B2, factorization survives.
If A1+B1 vs A2+B2, the original cluster signal was about the surface
word 'capital' all along.

Subject for A1/A2: the country whose capital we're asking about (operand
of the lookup).
Subject for B1/B2: the operand word being counted.
"""

from __future__ import annotations

from mechbench_core import Prompt, PromptSet

_COUNTRIES = ("France", "Japan", "Germany", "Italy",
              "Spain", "Russia", "Egypt", "Greece")
_CAPITALS = ("Paris", "Tokyo", "Berlin", "Rome",
             "Madrid", "Moscow", "Cairo", "Athens")

_CAPITAL_OPERANDS = ("capital", "capitals", "capitalism", "capitalist",
                     "capitalize", "capitalized", "capitalizing",
                     "capitalization")
_NEUTRAL_OPERANDS = ("elephant", "mountain", "computer", "philosopher",
                     "butterfly", "helicopter", "restaurant", "umbrella")


# A1: capital-lookup, with the word 'capital' present in the prompt.
DISAMBIG_A1 = PromptSet(
    name="DISAMBIG_A1_lookup_with_capital",
    prompts=tuple(
        Prompt(
            text=f"Complete this sentence with one word: The capital of {country} is",
            target=capital, subject=country, category="A1_lookup_capital_present",
        )
        for country, capital in zip(_COUNTRIES, _CAPITALS)
    ),
)

# A2: capital-lookup, paraphrased so the word 'capital' is absent.
DISAMBIG_A2 = PromptSet(
    name="DISAMBIG_A2_lookup_no_capital",
    prompts=tuple(
        Prompt(
            text=f"The administrative center of {country} is named, in one word,",
            target=capital, subject=country, category="A2_lookup_capital_absent",
        )
        for country, capital in zip(_COUNTRIES, _CAPITALS)
    ),
)

# B1: letter-counting, operand word contains 'capital'.
DISAMBIG_B1 = PromptSet(
    name="DISAMBIG_B1_counting_with_capital",
    prompts=tuple(
        Prompt(
            text=f"Complete this sentence with one number: The number of letters in the word '{word}' is",
            subject=word, category="B1_counting_capital_present",
        )
        for word in _CAPITAL_OPERANDS
    ),
)

# B2: letter-counting on operand words that do NOT contain 'capital'.
DISAMBIG_B2 = PromptSet(
    name="DISAMBIG_B2_counting_no_capital",
    prompts=tuple(
        Prompt(
            text=f"Complete this sentence with one number: The number of letters in the word '{word}' is",
            subject=word, category="B2_counting_capital_absent",
        )
        for word in _NEUTRAL_OPERANDS
    ),
)

# Combined set for one-shot iteration in the experiment.
DISAMBIG_ALL = PromptSet(
    name="DISAMBIG_ALL",
    prompts=(
        *DISAMBIG_A1.prompts,
        *DISAMBIG_A2.prompts,
        *DISAMBIG_B1.prompts,
        *DISAMBIG_B2.prompts,
    ),
)
