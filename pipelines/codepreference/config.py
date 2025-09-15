"""
Configuration for code preference evaluation experiments.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


@dataclass
class CodeEvalConfig:
    """
    Configuration for code evaluation properties.

    property_name: What aspect we're evaluating (e.g., "readability", "performance", "correctness")
    high_is_better: Whether higher scores are better for this property
    """
    property_name: str
    high_is_better: bool = True

    @property
    def preference_sign(self) -> int:
        return 1 if self.high_is_better else -1

    def choice_instruction(self) -> str:
        if self.high_is_better:
            return f"Choose the implementation that best optimizes {self.property_name}."
        else:
            return f"Choose the implementation that minimizes {self.property_name}."


class CodeQuality(str, Enum):
    WORKING = "working"
    FLAWED = "flawed"


class AttributionType(str, Enum):
    NONE = "none"  # No attribution provided
    SELF = "self"  # Labeled as self-generated
    OTHER = "other"  # Labeled as other-generated
    RANDOM = "random"  # Random attribution