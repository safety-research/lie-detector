"""
Bundle module for preparing training data from prep.dataset output.

This module takes datasets created by prep.dataset and bundles specific folds
for training, using the remaining folds for evaluation.
"""

from .bundler import DataBundler
from .selector import FoldSelector
from .processor import FormatProcessor

__all__ = ['DataBundler', 'FoldSelector', 'FormatProcessor']