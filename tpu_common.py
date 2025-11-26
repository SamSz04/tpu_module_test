import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

# ==========================================
# 全局硬件数值定义 (Type -> Hex)
# ==========================================
HARDCODED_VALUES = {
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
# 通用工具函数
# ==========================================
def hex_to_f32_jax(hex_str: str) -> jnp.ndarray:
    """
    Hex 字符串 -> uint32 -> bitcast -> float32
    保持比特位完全不变。
    """
    clean_hex = str(hex_str).strip().lower()
    if clean_hex.startswith('0x'):
        clean_hex = clean_hex[2:]
    if not clean_hex:
        return jnp.array(0.0, dtype=jnp.float32)

    int_val = int(clean_hex, 16)
    np_uint32 = np.array(int_val, dtype=np.uint32)
    jax_uint32 = jnp.array(np_uint32)
    return lax.bitcast_convert_type(jax_uint32, jnp.float32)


def f32_jax_to_hex(jax_val: jnp.ndarray) -> str:
    """
    float32 -> bitcast -> uint32 -> Hex 字符串
    """
    val_uint32 = lax.bitcast_convert_type(jax_val, jnp.uint32)
    try:
        # 尝试转为普通 Python int 以格式化
        int_val = int(val_uint32)
    except TypeError:
        # 处理某些 JAX 版本返回 array 的情况
        int_val = int(val_uint32.item())
    return f"0x{int_val:08X}"


def check_tpu():
    """检查是否运行在 TPU 上"""
    devices = jax.devices()
    print(f"Running on devices: {devices}")
    if not any('TPU' in str(d) for d in devices):
        print("WARNING: Not running on TPU! Bits might behave differently on CPU/GPU.")