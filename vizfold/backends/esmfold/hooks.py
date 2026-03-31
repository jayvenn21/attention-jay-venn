"""
Hook-based extraction of attention weights and hidden states from HuggingFace ESMFold.

HF EsmForProteinFolding does NOT support output_attentions / output_hidden_states.
We capture traces by registering forward hooks on the ESM-2 trunk:

  Attention weights: hook on each EsmSelfAttention module, monkey-patching
                     the forward to force attn_weights to be returned.
  Activations:       hook on each EsmLayer (full transformer block output).

The ESM-2 tokenizer adds <cls> and <eos> tokens, so attention maps are
(seq_len+2, seq_len+2). We slice out the special tokens so stored tensors
are (seq_len, seq_len), matching the FASTA sequence.

Structure module tracing:
  StructureModuleTraceCollector hooks into model.trunk.structure_module.ipa
  to capture IPA attention weights [H, N, N] at each block within each
  recycling iteration, and hooks on trunk.structure_module to capture
  per-recycle backbone positions and single representations.
"""
import re
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import inspect
import torch.nn as nn


class ESMFoldTraceCollector:
    """
    Collects attention weights and/or hidden states from the ESM-2 trunk
    inside HuggingFace EsmForProteinFolding.
    """

    def __init__(
        self,
        want_attention: bool = True,
        want_activations: bool = True,
        layer_indices: Optional[List[int]] = None,
        head_indices: Optional[List[int]] = None,
    ):
        self.want_attention = want_attention
        self.want_activations = want_activations
        self.layer_indices = layer_indices  # None => all
        self.head_indices = head_indices
        self.attention: Dict[str, torch.Tensor] = {}
        self.activations: Dict[str, torch.Tensor] = {}
        self._handles: List[Any] = []
        self._patched_forwards: List[Tuple[nn.Module, Callable]] = []

    def clear(self) -> None:
        self.attention.clear()
        self.activations.clear()

    def _should_store_layer(self, layer_idx: int) -> bool:
        return self.layer_indices is None or layer_idx in self.layer_indices

    def register_hooks(self, esm_model: nn.Module) -> None:
        """
        Register hooks on the ESM-2 trunk (model.esm passed in).

        Targets:
          - encoder.layer[i].attention.self  -> attention weights [B, H, N, N]
          - encoder.layer[i]                 -> activations [B, N, D]

        For HF ESM, EsmSelfAttention.forward() only returns attn_weights when
        the framework's output capturing mechanism requests it. We monkey-patch
        the forward to always emit (output, attn_weights).
        """
        encoder_layers = self._find_encoder_layers(esm_model)
        if not encoder_layers:
            warnings.warn(
                "Could not find encoder.layer ModuleList in ESM trunk. "
                "Trace extraction may not work.",
                UserWarning,
            )
            return

        for layer_idx, layer_module in enumerate(encoder_layers):
            if not self._should_store_layer(layer_idx):
                continue

            if self.want_attention:
                self_attn = self._find_self_attention(layer_module)
                if self_attn is not None:
                    self._patch_and_hook_attention(self_attn, layer_idx)

            if self.want_activations:
                h = layer_module.register_forward_hook(self._make_activation_hook(layer_idx))
                self._handles.append(h)

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        for module, orig_forward in self._patched_forwards:
            module.forward = orig_forward
        self._patched_forwards.clear()

    def _find_encoder_layers(self, esm_model: nn.Module) -> Optional[nn.ModuleList]:
        """Find the nn.ModuleList of transformer layers in the ESM encoder."""
        if hasattr(esm_model, "encoder"):
            enc = esm_model.encoder
            if hasattr(enc, "layer") and isinstance(enc.layer, nn.ModuleList):
                return enc.layer
        for name, module in esm_model.named_modules():
            if isinstance(module, nn.ModuleList) and re.match(r".*encoder.*layer$", name):
                return module
        return None

    def _find_self_attention(self, layer_module: nn.Module) -> Optional[nn.Module]:
        """Find the self-attention submodule inside a transformer layer."""
        if hasattr(layer_module, "attention"):
            attn = layer_module.attention
            if hasattr(attn, "self"):
                return attn.self
            return attn
        return None

    def _patch_and_hook_attention(self, self_attn: nn.Module, layer_idx: int) -> None:
        """
        Monkey-patch EsmSelfAttention.forward to always return attn_weights,
        then register a hook to capture them.

        HF EsmSelfAttention returns (attn_output, attn_weights) from its
        internal attention function. But the outer EsmAttention layer discards
        attn_weights with `attn_output, _ = self.self(...)`. We hook self_attn
        directly to get the full tuple.
        """
        orig_forward = self_attn.forward
        params = list(inspect.signature(orig_forward).parameters)
        # Find output_attentions position by name — robust to signature changes
        oa_pos = params.index("output_attentions") if "output_attentions" in params else -1

        def patched_forward(*args, **kwargs):
            if oa_pos >= 0 and oa_pos < len(args):
                args = args[:oa_pos] + (True,) + args[oa_pos + 1:]
            else:
                kwargs["output_attentions"] = True
            return orig_forward(*args, **kwargs)

        self_attn.forward = patched_forward
        self._patched_forwards.append((self_attn, orig_forward))

        h = self_attn.register_forward_hook(self._make_attention_hook(layer_idx))
        self._handles.append(h)

    def _make_attention_hook(self, layer_idx: int) -> Callable:
        """Hook that captures attn_weights from (attn_output, attn_weights) tuple."""
        def hook(module: nn.Module, inp: Any, out: Any) -> None:
            if not isinstance(out, tuple) or len(out) < 2:
                return
            attn_weights = out[1]
            if attn_weights is None:
                return
            # attn_weights shape: [B, H, N+2, N+2] (includes <cls> and <eos>)
            # Slice out special tokens -> [B, H, N, N]
            if attn_weights.dim() == 4 and attn_weights.shape[-1] >= 3:
                attn_weights = attn_weights[:, :, 1:-1, 1:-1]
            key = f"layer_{layer_idx:03d}"
            self.attention[key] = attn_weights.detach()
        return hook

    def _make_activation_hook(self, layer_idx: int) -> Callable:
        """Hook that captures the transformer layer output (hidden state)."""
        def hook(module: nn.Module, inp: Any, out: Any) -> None:
            h = out[0] if isinstance(out, tuple) else out
            if h is not None and isinstance(h, torch.Tensor) and h.dim() >= 2:
                key = f"layer_{layer_idx:03d}"
                self.activations[key] = h.detach()
        return hook


class StructureModuleTraceCollector:
    """
    Captures IPA attention weights and per-recycling-iteration backbone
    outputs from the ESMFold structure module.

    Hook targets:
      - trunk.structure_module.ipa: monkey-patched to stash the softmax
        attention matrix a [*, H, N, N] before it's consumed internally.
        Fires num_blocks times per recycle (IPA is a single shared module
        reused across all structure module blocks).
      - trunk.structure_module: captures the full output dict per recycle,
        including stacked positions [num_blocks, B, N, 14, 3], frames,
        and the final single representation.
    """

    def __init__(self) -> None:
        self.ipa_attention: Dict[str, torch.Tensor] = {}
        self.backbone_positions: Dict[str, torch.Tensor] = {}
        self.backbone_frames: Dict[str, torch.Tensor] = {}
        self.sm_states: Dict[str, torch.Tensor] = {}
        self._handles: List[Any] = []
        self._patched_forwards: List[Tuple[nn.Module, Callable]] = []
        self._recycle_idx = 0
        self._block_idx = 0

    def clear(self) -> None:
        self.ipa_attention.clear()
        self.backbone_positions.clear()
        self.backbone_frames.clear()
        self.sm_states.clear()
        self._recycle_idx = 0
        self._block_idx = 0

    def register_hooks(self, model: nn.Module) -> None:
        """
        Register hooks on model.trunk.structure_module and its IPA submodule.

        Args:
            model: the full EsmForProteinFolding model (we navigate to trunk).
        """
        trunk = getattr(model, "trunk", None)
        if trunk is None:
            warnings.warn("model.trunk not found; structure module hooks skipped.", UserWarning)
            return
        sm = getattr(trunk, "structure_module", None)
        if sm is None:
            warnings.warn("trunk.structure_module not found; hooks skipped.", UserWarning)
            return

        ipa = getattr(sm, "ipa", None)
        if ipa is not None:
            self._patch_ipa(ipa)

        h = sm.register_forward_hook(self._sm_output_hook)
        self._handles.append(h)

    def _patch_ipa(self, ipa: nn.Module) -> None:
        """
        Monkey-patch IPA forward to stash the softmax attention matrix.

        IPA computes a = softmax(scalar_attn + point_attn + pair_bias + mask)
        but only returns the single-rep update. We intercept a after softmax
        and store it, keyed by recycle_idx and block_idx.

        ipa.softmax is an nn.Softmax module, so we can't replace it with a
        plain function (PyTorch __setattr__ rejects non-Module assignments).
        Instead we wrap it in an nn.Module subclass that captures the output.
        """
        orig_forward = ipa.forward
        orig_softmax_module = ipa.softmax
        collector = self

        class CapturingSoftmax(nn.Module):
            def __init__(self, wrapped: nn.Module):
                super().__init__()
                self.wrapped = wrapped
                self.last_a: Optional[torch.Tensor] = None

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                result = self.wrapped(x)
                self.last_a = result.detach()
                return result

        capturing = CapturingSoftmax(orig_softmax_module)
        ipa.softmax = capturing

        def patched_forward(*args, **kwargs):
            capturing.last_a = None
            out = orig_forward(*args, **kwargs)
            if capturing.last_a is not None:
                key = f"recycle_{collector._recycle_idx:02d}_block_{collector._block_idx:02d}"
                collector.ipa_attention[key] = capturing.last_a
            collector._block_idx += 1
            return out

        ipa.forward = patched_forward
        self._patched_forwards.append((ipa, orig_forward))
        self._orig_softmax = (ipa, orig_softmax_module)

    def _sm_output_hook(self, module: nn.Module, inp: Any, out: Any) -> None:
        """
        Fires once per recycling iteration after the full structure module
        forward (all blocks). Captures backbone positions and states.
        """
        key = f"recycle_{self._recycle_idx:02d}"
        if isinstance(out, dict):
            if "positions" in out:
                self.backbone_positions[key] = out["positions"].detach().cpu()
            if "frames" in out:
                self.backbone_frames[key] = out["frames"].detach().cpu()
            if "single" in out:
                self.sm_states[key] = out["single"].detach().cpu()

        self._recycle_idx += 1
        self._block_idx = 0

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        for module, orig_forward in self._patched_forwards:
            module.forward = orig_forward
        self._patched_forwards.clear()
        if hasattr(self, "_orig_softmax"):
            ipa_mod, orig_sm = self._orig_softmax
            ipa_mod.softmax = orig_sm
            del self._orig_softmax
