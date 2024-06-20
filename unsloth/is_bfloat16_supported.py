# unsloth/is_bfloat16_supported.py

import torch

def is_bfloat16_supported():
    return torch.cuda.is_bfloat16_supported()
