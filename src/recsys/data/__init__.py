"""Data layer: unified dataset container, synthetic generator, Yelp subsetter, loaders."""

from .dataset import Dataset
from .loaders import load_dataset

__all__ = ["Dataset", "load_dataset"]
