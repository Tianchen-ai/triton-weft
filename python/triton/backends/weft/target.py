"""Inspect CPU compiler targets and native execution capabilities.

Catalog and explicit-target queries work on a cross-build host and require no
C compiler, sysroot or device. Native discovery reports the current CPU set;
it neither changes affinity nor grants permission to execute an extension.
"""
from weft_core.target import extension_catalog, target_capabilities, native_capabilities

__all__ = ["extension_catalog", "target_capabilities", "native_capabilities"]
