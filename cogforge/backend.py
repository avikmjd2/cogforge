import numpy 
try:
    import numexpr as _ne
    _NUMEXPR_AVAILABLE = True
except ImportError:
    _NUMEXPR_AVAILABLE = False
    
USE_NUMEXPR = False  

try:
    import cupy as _cp
    _CUPY_AVAILABLE = True
except ImportError:
    _cp = None
    _CUPY_AVAILABLE = False        
    
np = numpy
USE_GPU = False
NO_GRAD = False

def use_gpu(enabled = True):
    global np, USE_GPU
    if enabled and not _CUPY_AVAILABLE:
        raise RuntimeError("cupy not installed")
    np = _cp if enabled else numpy
    USE_GPU = bool(enabled)
    return USE_GPU