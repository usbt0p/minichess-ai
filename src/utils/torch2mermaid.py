import re
import operator
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.fx import symbolic_trace


class MermaidNNTranscriber:
    """
    Convierte un modelo PyTorch en un diagrama Mermaid flowchart.

    - Usa torch.fx para capturar el grafo de forward.
    - Usa los módulos reales para etiquetar capas (por ejemplo, nn.Linear).
    - Puede representar ramas y múltiples salidas.
    """

    def __init__(
        self,
        model: nn.Module,
        input_dim: Optional[int] = None,
        input_label: str = "Input",
        output_labels: Optional[Dict[str, Tuple[str, str]]] = None,
    ):
        """
        Parameters
        ----------
        model : nn.Module
            Modelo PyTorch.
        input_dim : int, optional
            Dimensión de entrada. Si no se pasa, intenta inferirla del primer Linear.
        input_label : str
            Texto para el nodo de entrada.
        output_labels : dict, optional
            Mapa opcional para renombrar salidas.
            Formato:
                {
                    "value_result_head": ("value_result", "Logits de Clasificación"),
                    "policy_head": ("policy_logits", "Probabilidades de Movimiento"),
                }

            Si no se pasa, se usarán nombres genéricos basados en el módulo.
        """
        self.model = model.eval()
        self.input_dim = input_dim or self._infer_input_dim()
        self.input_label = input_label
        self.output_labels = output_labels or {}

    def _infer_input_dim(self) -> Optional[int]:
        """Intenta inferir la dimensión de entrada del primer Linear."""
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                return m.in_features
        return None

    def _sanitize_id(self, text: str) -> str:
        """Convierte un texto en un identificador Mermaid válido."""
        text = re.sub(r"[^a-zA-Z0-9_]", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        return text.upper() or "NODE"

    def _module_label(self, name: str, module: nn.Module) -> str:
        """Etiqueta Mermaid para un módulo."""
        if isinstance(module, nn.Linear):
            return f'{name} (Linear)<br/>{module.in_features} → {module.out_features}'
        if isinstance(module, nn.ReLU):
            return "ReLU"
        if isinstance(module, nn.Sigmoid):
            return "Sigmoid"
        if isinstance(module, nn.Tanh):
            return "Tanh"
        if isinstance(module, nn.Softmax):
            return "Softmax"
        if isinstance(module, nn.Dropout):
            return f"Dropout(p={module.p})"
        if isinstance(module, nn.BatchNorm1d):
            return f"{name} (BatchNorm1d)<br/>features={module.num_features}"
        if isinstance(module, nn.Conv2d):
            return (
                f'{name} (Conv2d)<br/>'
                f'{module.in_channels} → {module.out_channels}, '
                f'k={module.kernel_size}, s={module.stride}'
            )
        return f"{name} ({module.__class__.__name__})"

    def _input_node_label(self) -> str:
        if self.input_dim is None:
            return f'["{self.input_label}"]'
        return f'(["{self.input_label}<br/>dim={self.input_dim}"])'

    def _output_node_label(self, node_name: str, module: nn.Module) -> str:
        """
        Si el usuario ha definido un alias para esta cabeza, úsalo.
        Si no, genera una salida genérica.
        """
        if node_name in self.output_labels:
            out_name, desc = self.output_labels[node_name]
            if isinstance(module, nn.Linear):
                return f'(["{out_name}<br/>({desc})<br/>dim={module.out_features}"])'
            return f'(["{out_name}<br/>({desc})"])'

        if isinstance(module, nn.Linear):
            return f'(["{node_name}<br/>dim={module.out_features}"])'

        return f'(["{node_name}"])'

    def _iter_fx_args(self, arg):
        """Devuelve todos los nodos FX contenidos en args/kwargs anidados."""
        if isinstance(arg, torch.fx.Node):
            yield arg
        elif isinstance(arg, (tuple, list)):
            for x in arg:
                yield from self._iter_fx_args(x)
        elif isinstance(arg, dict):
            for x in arg.values():
                yield from self._iter_fx_args(x)

    def to_mermaid(self) -> str:
        """
        Genera el flujo Mermaid.
        """
        traced = symbolic_trace(self.model)
        modules = dict(self.model.named_modules())

        node_ids = {}
        lines = ["flowchart TD"]

        # Nodo de entrada
        input_id = "IN"
        lines.append(f'    {input_id}{self._input_node_label()}')

        # Recorremos el grafo FX en orden
        for fx_node in traced.graph.nodes:
            if fx_node.op == "placeholder":
                # Conectamos el placeholder al nodo de entrada
                node_ids[fx_node.name] = input_id
                continue

            if fx_node.op == "output":
                # Output puede ser un tuple/list de salidas
                output_sources = list(self._iter_fx_args(fx_node.args))
                if not output_sources:
                    continue

                # Si hay varias salidas, creamos un nodo final por cada una
                for i, src in enumerate(output_sources, start=1):
                    src_id = node_ids.get(src.name)
                    if src_id is None:
                        continue
                    out_id = f"OUT_{i}"
                    lines.append(f'    {out_id}(["Output {i}"])')
                    lines.append(f"    {src_id} --> {out_id}")
                continue

            if fx_node.op == "call_module":
                module = modules[fx_node.target]
                nice_name = fx_node.target.split(".")[-1]
                node_id = self._sanitize_id(fx_node.name)

                label = self._module_label(nice_name, module)
                lines.append(f'    {node_id}["{label}"]')
                node_ids[fx_node.name] = node_id

                # Conectamos entradas
                for arg in self._iter_fx_args(fx_node.args):
                    if arg.name in node_ids:
                        lines.append(f"    {node_ids[arg.name]} --> {node_id}")

                for kwarg in self._iter_fx_args(fx_node.kwargs):
                    if kwarg.name in node_ids:
                        lines.append(f"    {node_ids[kwarg.name]} --> {node_id}")

                continue

            if fx_node.op in ("call_function", "call_method"):
                # Operaciones funcionales: relu, flatten, add, etc.
                fn_name = getattr(fx_node.target, "__name__", str(fx_node.target))
                node_id = self._sanitize_id(fx_node.name)

                if fn_name == "relu":
                    label = "ReLU"
                elif fn_name == "flatten":
                    label = "Flatten"
                elif fn_name == "softmax":
                    label = "Softmax"
                elif fn_name == "sigmoid":
                    label = "Sigmoid"
                elif fn_name == "tanh":
                    label = "Tanh"
                else:
                    label = fn_name

                lines.append(f'    {node_id}["{label}"]')
                node_ids[fx_node.name] = node_id

                for arg in self._iter_fx_args(fx_node.args):
                    if arg.name in node_ids:
                        lines.append(f"    {node_ids[arg.name]} --> {node_id}")

                continue

        # Ahora, generamos salidas bonitas para las cabezas finales tipo Linear si existen
        # (esto mejora mucho el caso de multi-head).
        # Re-traversal para detectar módulos finales.
        for fx_node in traced.graph.nodes:
            if fx_node.op != "call_module":
                continue

            module = modules[fx_node.target]
            if not isinstance(module, nn.Linear):
                continue

            # Si este nodo no alimenta a nadie más, lo tratamos como salida
            # (o si su nombre parece "head").
            users = list(fx_node.users.keys())
            is_terminal = len(users) == 0
            looks_like_head = any(
                key in fx_node.target.lower() for key in ["head", "output", "out", "result"]
            )

            if not (is_terminal or looks_like_head):
                continue

            # Reemplazamos el nodo por una salida más semántica, si hay alias
            if fx_node.target in self.output_labels:
                out_name, desc = self.output_labels[fx_node.target]
                out_id = self._sanitize_id(f"OUT_{fx_node.name}")
                lines.append(f'    {out_id}(["{out_name}<br/>({desc})<br/>dim={module.out_features}"])')
                if fx_node.name in node_ids:
                    lines.append(f"    {node_ids[fx_node.name]} --> {out_id}")
            else:
                # salida genérica
                pass

        return "\n".join(lines)

if __name__ == "__main__":
    from src.models.baseline import BaselineNet
    
    model = BaselineNet(result_mode="classification")

    transcriber = MermaidNNTranscriber(model)
    mermaid_str = transcriber.to_mermaid()

    with open("baseline_model.mmd", "w") as f:
        f.write(mermaid_str)
