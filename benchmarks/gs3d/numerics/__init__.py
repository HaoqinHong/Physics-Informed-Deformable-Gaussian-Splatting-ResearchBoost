"""Standard precision evaluation: record -> calibrate -> compare (+ fp64 arbitration) -> selftest -> TOST.

Tasks only implement numerics.adapter.Adapter. See docs/numerics.md.
"""
from .adapter import Adapter, Objects

__all__ = ['Adapter', 'Objects']
