import time
from datetime import timedelta
from functools import wraps

import torch
import numpy as np
import warnings
import logging
import inspect

import base64
import os

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

def pretty_time(seconds: int) -> str:
    """Format total seconds into HH:MM:SS"""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}h {minutes:02d}m {seconds:02d}s"
    elif minutes > 0:
        return f"{minutes:02d}m {seconds:02d}s"
    else:
        return f"{seconds:02d}s"

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

def count_params(net, module_structure=False, perlayer=False):
    '''Counts the number of trainable parameters in a PyTorch model 
    and prints them in a human-readable format.
    If verbose=True, also outputs module __repr__, and each layer's 
        name, size (dims), exact number of params and dtype
    '''
    if module_structure: print(f"---> {net} params:")
    total = 0
    for name, param in net.named_parameters():
        if param.requires_grad:
            p = param.numel()
            type = param.dtype
            total += p
        else:
            p = None
        if perlayer: print("\t", name, param.size(), p, type)

    print()
    print(f"Total number of trainable parameters: {total:,}")
    bytes = total * torch.tensor([], dtype=type).element_size()
    print(f"\tIn bits: {bytes * 8:,} bits")
    print(f"\tIn bytes: {bytes:,} bytes")
    print(f"\tIn kilobytes: {bytes / 1024:,} KB")
    print(f"\tIn megabytes: {bytes / 1024**2} MB", end="\n\n")

def set_seed(seed):
    """Set seed for reproducibility.
    """
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


UNICODE_PIECES = {
    'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙',
    'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟'
}

def print_board(fen: str):
    """Muestra el tablero 5x5 en la terminal usando caracteres Unicode."""
    board_part = fen.split()[0]
    rows = board_part.split('/')
    print("\n  +---+---+---+---+---+")
    for i, row in enumerate(rows):
        rank = 5 - i
        row_str = f"{rank} |"
        for char in row:
            if char.isdigit():
                for _ in range(int(char)):
                    row_str += "   |"
            else:
                row_str += f" {UNICODE_PIECES.get(char, char)} |"
        print(row_str)
        print("  +---+---+---+---+---+")
    print("    a   b   c   d   e  \n")

def _get_piece_svg_b64(char):
    color_code = 'l' if char.isupper() else 'd'
    piece_code = char.lower()
    filename = f"Chess_{piece_code}{color_code}t45.svg"
    filepath = os.path.join("src", "chess", "svg", filename)
    if not os.path.exists(filepath):
        return None
    with open(filepath, "rb") as f:
        data = base64.b64encode(f.read()).decode('utf-8')
    return f"data:image/svg+xml;base64,{data}"

def get_svg_board(fen: str) -> str:
    """Devuelve el string del contenido SVG del tablero (sin la etiqueta <svg> envolvente externa)"""
    board_part = fen.split()[0]
    rows = board_part.split('/')
    
    sq = 60 # Tamaño de la casilla
    
    svg = []
    
    # Dibujar el tablero a cuadros
    for r in range(5):
        for c in range(5):
            color = "#f0d9b5" if (r + c) % 2 == 0 else "#b58863"
            svg.append(f'<rect x="{c * sq}" y="{r * sq}" width="{sq}" height="{sq}" fill="{color}" />')

    # Añadir las piezas (incrustando los SVG)
    for r, row in enumerate(rows):
        c = 0
        for char in row:
            if char.isdigit():
                c += int(char)
            else:
                b64_data = _get_piece_svg_b64(char)
                if b64_data:
                    # Lo escalamos para que llene la casilla
                    svg.append(f'<image href="{b64_data}" x="{c * sq}" y="{r * sq}" width="{sq}" height="{sq}" />')
                else:
                    # Fallback a Unicode si no se encuentra el SVG
                    piece_char = UNICODE_PIECES.get(char, char)
                    color = "#000000" if char.islower() else "#ffffff"
                    svg.append(
                        f'<text x="{c * sq + sq/2}" y="{r * sq + sq/2 + 20}" '
                        f'font-size="{sq * 0.8}" text-anchor="middle" font-family="Arial" '
                        f'fill="{color}">{piece_char}</text>'
                    )
                c += 1
                
    return "\n".join(svg)

def export_svg(fen: str, filename: str = "tablero.svg"):
    """Exporta el FEN actual a un archivo de imagen vectorial SVG con las piezas oficiales."""
    sq = 60
    width, height = 5 * sq, 5 * sq
    
    svg = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']
    svg.append(get_svg_board(fen))
    
    svg.append('</svg>')
    
    with open(filename, "w") as f:
        f.write("\n".join(svg))
    print(f"[*] Imagen SVG guardada en: {filename}")

class Tee:
    """Writes to both the terminal and a file."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal.isatty()