"""Compatibility imports for the earlier native Spectrum adapter module.

New adapters should import from model_patch.
"""
from .model_patch import (
    apply_spectrum_model_patch as apply_native_spectrum_patch,
    record_feature,
)
