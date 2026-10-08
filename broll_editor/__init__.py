"""Smackin B-roll Editor: stacked talking-head/b-roll and split-screen dialogue edits."""

from .splitscreen import SplitOptions, make_split_screen
from .stacker import StackOptions, probe, stack_videos

__all__ = ["SplitOptions", "StackOptions", "make_split_screen", "probe", "stack_videos"]
