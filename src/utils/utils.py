import time
from datetime import timedelta
from functools import wraps

import torch
import numpy as np
import warnings
import logging
import inspect

def time_this(func):
    # here, wraps serves as an interface to keep the original function's metadata
    # (name, docstring, etc) even after it's been wrapped (substituted by wrapper),
    # so we avoid this: wrapper.__name__ = func.__name__; wrapper.__doc__ = func.__doc__; etc.
    # https://stackoverflow.com/a/309000
    @wraps(func)
    def wrapper(*arg, **kw):
        should_time = kw.pop('time', False)
        
        if should_time:
            t1 = time.time()
            res = func(*arg, **kw)
            t2 = time.time()
            t = str(timedelta(seconds=round(t2-t1, 3)))[:-3] # shave off trailing zeroes after rounding
            # TODO replace : with unit if unit is nonzero
                        
            # just do this since knowing if the func is a method is hard without knowing the object type
            if func.__name__ == "__init__":
                print(f"\t>> {arg[0].__class__.__name__}.{func.__name__} took {t}")
            else:
                print(f"\t>> {func.__name__} took {t}")
            return res
        else:
            return func(*arg, **kw)
    return wrapper

def save_and_export(model, dummy_input, target_path="best_model.pth"):
    '''Save model in PyTorch format and export to ONNX.
    - Dummy input is needed to trace the model's operations. Must be of the proper size.
    '''
    # save pt model
    torch.save(model.state_dict(), target_path)

    onnx_path = target_path.replace(".pth", ".onnx")

    # Keep ONNX export quiet without muting warnings globally for the full script.
    # if you want it to display warnings, just remove the onnx_logger 
    # lines and the warnings.catch_warnings block
    onnx_logger = logging.getLogger("torch.onnx")
    old_level = onnx_logger.level
    onnx_logger.setLevel(logging.ERROR)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=r".*LeafSpec.*")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=18,
            input_names=['input'],
            output_names=['output'],
            verbose=False,
            dynamo=False,
        )

    onnx_logger.setLevel(old_level)
    #print(f"Modelo exportado a {onnx_path}")

def count_params(net):
    '''Counts the number of trainable parameters in a PyTorch model 
    and prints them in a human-readable format.'''
    print(f"---> {net} params:")
    total = 0
    for name, param in net.named_parameters():
        if param.requires_grad:
            p = param.numel()
            type = param.dtype
            total += p
        else:
            p = None
        print("\t", name, param.size(), p, type)

    print()
    print(f"Total number of trainable parameters: {total}")
    bytes = total * torch.tensor([], dtype=type).element_size()
    print(f"\tIn bits: {bytes * 8} bits")
    print(f"\tIn bytes: {bytes} bytes")
    print(f"\tIn kilobytes: {bytes / 1024} KB")
    print(f"\tIn megabytes: {bytes / 1024**2} MB", end="\n\n")

def set_seed(seed):
    """Set seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)