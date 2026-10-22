"""Compatibility imports for the earlier Krea2 adapter module.

New callers should use model_patch_krea2.apply_krea2_model_patch.
"""
from .model_patch_krea2 import (
    apply_krea2_model_patch as apply_krea2_spectrum_patch,
    capture_owner, capture_safe, extract_features, head, key, name, require_model,
)
