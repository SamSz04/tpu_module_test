import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

# ==========================================
# 1. FP32 硬件数值定义 (Type -> 32-bit Hex)
# ==========================================
HARDCODED_VALUES_FP32 = {
    "qnan": "0x7FC80000",
    "snan": "0xFFA00000",
    "inf": "0x7F800000",
    "-inf": "0xFF800000",
    "zero": "0x0",
    "-zero": "0x80000000",
    "normal": "0x3FC00001",
    "-normal": "0xBFC00000",
    "subnormal": "0x400000",
    "-subnormal": "0x80400000"
}

# ==========================================
# 2. BF16 硬件数值定义 (Type -> 16-bit Hex)
# ==========================================
HARDCODED_VALUES_BF16 = {
    "qnan": "0x7FC0",
    "snan": "0xFFA0",
    "inf": "0x7F80",
    "-inf": "0xFF80",
    "zero": "0x0",
    "-zero": "0x8000",
    "normal": "0x3FC1",
    "-normal": "0xBFC0",
    "subnormal": "0x40",
    "-subnormal": "0x8040"
}


# ==========================================
# 通用工具函数 (支持 FP32 和 BF16)
# ==========================================

def hex_to_jax(hex_str: str, dtype) -> jnp.ndarray:
    """
    通用 Hex 转 JAX Array 函数。
    dtype: jnp.float32 或 jnp.bfloat16
    """
    clean_hex = str(hex_str).strip().lower()
    if clean_hex.startswith('0x'):
        clean_hex = clean_hex[2:]
    if not clean_hex:
        # 空串返回 0
        return jnp.array(0.0, dtype=dtype)

    int_val = int(clean_hex, 16)

    if dtype == jnp.float32:
        # FP32: Hex -> uint32 -> float32
        np_uint = np.array(int_val, dtype=np.uint32)
        jax_uint = jnp.array(np_uint)
        return lax.bitcast_convert_type(jax_uint, jnp.float32)

    elif dtype == jnp.bfloat16:
        # BF16: Hex -> uint16 -> bfloat16
        np_uint = np.array(int_val, dtype=np.uint16)
        jax_uint = jnp.array(np_uint)
        return lax.bitcast_convert_type(jax_uint, jnp.bfloat16)

    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def jax_to_hex(jax_val: jnp.ndarray) -> str:
    """
    通用 JAX Array 转 Hex 函数。
    自动检测输入是 float32 还是 bfloat16。
    """
    dtype = jax_val.dtype

    if dtype == jnp.float32:
        # float32 -> uint32 -> 8位 Hex
        val_uint = lax.bitcast_convert_type(jax_val, jnp.uint32)
        width = 8
    elif dtype == jnp.bfloat16:
        # bfloat16 -> uint16 -> 4位 Hex
        val_uint = lax.bitcast_convert_type(jax_val, jnp.uint16)
        width = 4
    else:
        # 兼容处理
        return str(jax_val)

    try:
        int_val = int(val_uint)
    except TypeError:
        int_val = int(val_uint.item())

    # 格式化为大写 Hex，补齐位数
    return f"0x{int_val:0{width}X}"


def check_tpu():
    devices = jax.devices()
    print(f"Running on devices: {devices}")
    if not any('TPU' in str(d) for d in devices):
        print("WARNING: Not running on TPU!")