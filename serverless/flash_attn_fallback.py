"""
Flash Attention Fallback for LongCat-Video
Monkey-patches flash_attn imports to use PyTorch's scaled_dot_product_attention
"""

import torch
import torch.nn.functional as F


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None, deterministic=False):
    """
    Fallback implementation using PyTorch's scaled_dot_product_attention
    
    Args:
        q, k, v: Query, Key, Value tensors [batch, seqlen, nheads, headdim]
        dropout_p: Dropout probability
        softmax_scale: Scaling factor for attention scores
        causal: Whether to apply causal masking
        Other args: Ignored in fallback
    
    Returns:
        output: Attention output [batch, seqlen, nheads, headdim]
    """
    # Reshape to [batch, nheads, seqlen, headdim] for PyTorch SDPA
    batch, seqlen, nheads, headdim = q.shape
    
    q = q.transpose(1, 2)  # [batch, nheads, seqlen, headdim]
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    # Scale
    if softmax_scale is None:
        softmax_scale = 1.0 / (headdim ** 0.5)
    
    # Use PyTorch's built-in scaled_dot_product_attention (supports flash attention if available)
    try:
        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=dropout_p if deterministic else 0.0,
            is_causal=causal,
            scale=softmax_scale
        )
    except Exception:
        # Manual fallback if SDPA not available
        scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
        
        if causal:
            mask = torch.triu(torch.ones(seqlen, seqlen, device=q.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        
        if dropout_p > 0 and not deterministic:
            attn = F.dropout(attn, p=dropout_p)
        
        output = torch.matmul(attn, v)
    
    # Reshape back to [batch, seqlen, nheads, headdim]
    output = output.transpose(1, 2).contiguous()
    
    return output


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None, deterministic=False, return_attn_probs=False):
    """
    Variable-length fallback (simplified - assumes single sequence for now)
    """
    return flash_attn_func(q, k, v, dropout_p, softmax_scale, causal, window_size, softcap, alibi_slopes, deterministic)


# Create a fake flash_attn module
class FakeFlashAttnModule:
    flash_attn_func = staticmethod(flash_attn_func)
    flash_attn_varlen_func = staticmethod(flash_attn_varlen_func)
    __version__ = "fallback-0.0.0"
    __spec__ = None  # Explicitly set __spec__ to prevent importlib errors


def install_fallback():
    """Install the fallback by adding it to sys.modules"""
    import sys
    import importlib.util
    
    # Create a proper module spec to avoid "flash_attn.__spec__ is not set" error
    fake_module = FakeFlashAttnModule()
    
    # Create a ModuleSpec for the fake module
    spec = importlib.util.spec_from_loader("flash_attn", loader=None)
    fake_module.__spec__ = spec
    
    sys.modules['flash_attn'] = fake_module
    print("⚠️  flash-attn not found - using PyTorch fallback (slower but works)")
