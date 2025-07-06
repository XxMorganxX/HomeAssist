"""
Centralized warning suppression for the RasPi Smart Home project.

This module handles suppression of various deprecation warnings from dependencies.
Import this module at the beginning of any entry point to suppress known warnings.
"""

import warnings

def suppress_common_warnings():
    """Suppress common deprecation warnings from dependencies."""
    
    # Suppress pkg_resources deprecation warning
    # This warning comes from older packages that haven't migrated to importlib yet
    warnings.filterwarnings("ignore", category=UserWarning, 
                          message=".*pkg_resources is deprecated.*")
    
    # Suppress numpy dtype deprecation warnings if they occur
    warnings.filterwarnings("ignore", category=DeprecationWarning,
                          message=".*numpy.dtype size changed.*")
    
    # Suppress tensorflow/onnx warnings if they occur
    warnings.filterwarnings("ignore", category=FutureWarning,
                          module="tensorflow|onnxruntime")
    
    # Suppress google auth warnings about using file-based credentials
    warnings.filterwarnings("ignore", category=UserWarning,
                          message=".*credentials were discovered.*")

# Automatically suppress warnings when this module is imported
suppress_common_warnings()