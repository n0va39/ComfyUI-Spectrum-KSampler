"""Compatibility imports for the earlier native MOSS adapter module.

New callers should use model_patch_moss.apply_moss_model_patch.
"""
from .model_patch_moss import (
    apply_moss_model_patch as apply_native_moss_spectrum_patch,
    capture_owner, capture_safe, extract_features, head, key, name, require_model,
)
