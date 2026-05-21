import os
import torch
from torch.utils.cpp_extension import load

# We can either load the extension JIT here or assume it's pre-compiled via setup.py.
# Assuming setup.py is used to install it into the environment:
try:
    import plantclef_ext
except ImportError:
    print("Warning: plantclef_ext is not installed. Please run `python setup.py build_ext --inplace` in src/extensions.")
