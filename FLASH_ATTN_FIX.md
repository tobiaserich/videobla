# Flash Attention Fallback Fix

## Problems Fixed

### 1. `flash_attn.__spec__ is not set` Error

**Symptom:** Model loading failed with `ValueError: flash_attn.__spec__ is not set`

**Root Cause:** The `diffusers` library checks for flash_attn availability during initialization using `importlib.util.find_spec()`, which requires modules to have a valid `__spec__` attribute. The original fallback module didn't set this attribute properly.

**Solution:**

- Added explicit `__spec__` attribute to `FakeFlashAttnModule` class
- Updated `install_fallback()` to create a proper `ModuleSpec` using `importlib.util.spec_from_loader()`
- Improved handler.py to detect and fix missing `__spec__` on real flash_attn if present

### 2. Tensor Shape Mismatch Error

**Symptom:** Video generation failed with `ValueError: not enough values to unpack (expected 4, got 3)`

**Root Cause:** The fallback `flash_attn_func` only handled 4D tensors `[batch, seqlen, nheads, headdim]`, but LongCat-Video also uses 3D tensors `[total_seqlen, nheads, headdim]` in varlen attention.

**Solution:**

- Added logic to detect tensor dimensions (3D vs 4D)
- Handle 3D tensors by adding batch dimension, processing, then squeezing back
- Maintain backward compatibility with 4D tensors

## Files Modified

### `serverless/flash_attn_fallback.py`

1. Added `__spec__ = None` to `FakeFlashAttnModule` class
2. Updated `install_fallback()` to create proper ModuleSpec
3. Enhanced `flash_attn_func()` to handle both 3D and 4D tensors
4. Added dimension detection and appropriate reshaping logic

### `serverless/handler.py`

1. Improved flash_attn detection using `importlib.util.find_spec()`
2. Added logic to fix `__spec__` on real flash_attn if needed
3. Enhanced error handling with traceback
4. Added explanatory comments

## Technical Details

### ModuleSpec Creation

```python
spec = importlib.util.spec_from_loader("flash_attn", loader=None)
fake_module.__spec__ = spec
```

### Tensor Dimension Handling

```python
if q.dim() == 3:
    # [total_seqlen, nheads, headdim] -> [1, nheads, seqlen, headdim]
    q = q.unsqueeze(0).transpose(1, 2)
    # ... process ...
    output = output.transpose(1, 2).squeeze(0).contiguous()
elif q.dim() == 4:
    # [batch, seqlen, nheads, headdim] -> [batch, nheads, seqlen, headdim]
    q = q.transpose(1, 2)
    # ... process ...
    output = output.transpose(1, 2).contiguous()
```

## Result

✅ Model now loads successfully without flash_attn errors  
✅ Video generation works with the PyTorch fallback  
✅ Compatible with both 3D and 4D attention tensors  
✅ No external flash_attn installation required

## Performance Note

The PyTorch fallback uses `F.scaled_dot_product_attention()` which is optimized but slower than the CUDA-optimized flash_attn. For production use, consider installing the real flash_attn package for better performance.
