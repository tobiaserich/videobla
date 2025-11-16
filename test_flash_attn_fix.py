#!/usr/bin/env python3
"""
Test script to verify the flash_attn fallback fix
"""
import sys
import os

# Add serverless directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "serverless"))

print("=" * 60)
print("Testing flash_attn fallback fix...")
print("=" * 60)

# Import the flash_attn_fallback module
from flash_attn_fallback import install_fallback

# Install the fallback
install_fallback()

# Now test if importlib can find flash_attn without errors
import importlib.util

print("\n1. Testing importlib.util.find_spec('flash_attn')...")
try:
    spec = importlib.util.find_spec("flash_attn")
    print(f"   ✅ find_spec succeeded: {spec}")
    print(f"   - spec is None: {spec is None}")
    if spec is not None:
        print(f"   - spec.name: {spec.name}")
        print(f"   - spec.loader: {spec.loader}")
except ValueError as e:
    print(f"   ❌ ERROR: {e}")
    sys.exit(1)

# Test importing flash_attn
print("\n2. Testing 'import flash_attn'...")
try:
    import flash_attn
    print(f"   ✅ Import succeeded")
    print(f"   - __spec__: {flash_attn.__spec__}")
    print(f"   - __version__: {flash_attn.__version__}")
    print(f"   - has flash_attn_func: {hasattr(flash_attn, 'flash_attn_func')}")
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    sys.exit(1)

# Test the critical path that was failing in diffusers
print("\n3. Simulating diffusers import check...")
try:
    # This is what diffusers does that was causing the error
    pkg_exists = importlib.util.find_spec("flash_attn") is not None
    print(f"   ✅ Check succeeded: pkg_exists = {pkg_exists}")
except ValueError as e:
    print(f"   ❌ ERROR: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ All tests passed! The fix should work.")
print("=" * 60)
