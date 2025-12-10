import jax
import jax.numpy as jnp


def main():
    # print(f"JAX Platform: {jax.lib.xla_bridge.get_backend().platform}")

    # 定义 uint32 类型
    dtype = jnp.uint32

    # ==========================
    # Case 1: 1 / 1 (正常情况)
    # ==========================
    x1 = jnp.array(1, dtype=dtype)
    y1 = jnp.array(1, dtype=dtype)

    q1 = jnp.floor_divide(x1, y1)
    r1 = jnp.remainder(x1, y1)

    print("\nTest: 1 / 1")
    print(f"  Quotient : {q1} (Hex: 0x{int(q1):X})")
    print(f"  Remainder: {r1} (Hex: 0x{int(r1):X})")

    # ==========================
    # Case 2: 1 / 0 (除以零)
    # ==========================
    x2 = jnp.array(1, dtype=dtype)
    y2 = jnp.array(0, dtype=dtype)

    q2 = jnp.floor_divide(x2, y2)
    r2 = jnp.remainder(x2, y2)

    print("\nTest: 1 / 0")
    print(f"  Quotient : {q2} (Hex: 0x{int(q2):X})")
    print(f"  Remainder: {r2} (Hex: 0x{int(r2):X})")

    # ==========================
    # Case 3: 0 / 1 (正常情况)
    # ==========================
    x3 = jnp.array(0, dtype=dtype)
    y3 = jnp.array(1, dtype=dtype)

    q3 = jnp.floor_divide(x3, y3)
    r3 = jnp.remainder(x3, y3)

    print("\nTest: 0 / 1")
    print(f"  Quotient : {q3} (Hex: 0x{int(q3):X})")
    print(f"  Remainder: {r3} (Hex: 0x{int(r3):X})")

    # ==========================
    # Case 4: 0 / 0 (除以零)
    # ==========================
    x4 = jnp.array(0, dtype=dtype)
    y4 = jnp.array(0, dtype=dtype)

    q4 = jnp.floor_divide(x4, y4)
    r4 = jnp.remainder(x4, y4)

    print("\nTest: 0 / 0")
    print(f"  Quotient : {q4} (Hex: 0x{int(q4):X})")
    print(f"  Remainder: {r4} (Hex: 0x{int(r4):X})")


if __name__ == "__main__":
    main()