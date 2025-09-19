import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple

os.environ['JAX_PLATFORM_NAME'] = 'gpu'


class GPUFloatOperationTester:
    def __init__(self, dtype=jnp.float32):
        """
        初始化GPU测试器
        dtype: jnp.float32 or jnp.bfloat16
        """
        self.dtype = dtype

        # 确认在GPU上运行
        devices = jax.devices()
        print(f"JAX devices: {devices}")
        if not any('GPU' in str(d) for d in devices):
            print("WARNING: Not running on GPU!")

        # GPU信息
        print(f"Device count: {jax.device_count()}")
        print(f"Local device count: {jax.local_device_count()}")

    def print_test_values_description(self):
        """打印 trunc 测试值定义：truncate towards zero（朝 0 截断）"""
        if self.dtype == jnp.float32:
            print("\nFP32 测试值定义（trunc：towards zero）：")
            print("- nan: float('nan')")
            print("- inf: float('inf')")
            print("- -inf: float('-inf')")
            print("- zero: 0.0")
            print("- -zero: -0.0")
            print("- normal: 1.5          （→ 1.0）")
            print("- -normal: -1.5        （→ -1.0）")
            print("- positive_fraction: 2.9   （→ 2.0）")
            print("- negative_fraction: -2.9  （→ -2.0）")
            print("- positive_small: 0.9      （→ 0.0）")
            print("- negative_small: -0.9     （→ 0.0）")
            print("- positive_integer: 3.0    （→ 3.0）")
            print("- negative_integer: -3.0   （→ -3.0）")
            print("- positive_half: 2.5       （→ 2.0）")
            print("- negative_half: -2.5      （→ -2.0）")
            print("- large_positive: 999.99   （→ 999.0）")
            print("- large_negative: -999.99  （→ -999.0）")
            print("- subnormal: 1e-40")
            print("- -subnormal: -1e-40")
            print(f"- min_normal: np.finfo(np.float32).tiny （最小正规数 ≈ {np.finfo(np.float32).tiny:.8e}）")
            print("- near_overflow: 1e38 / -near_overflow: -1e38")
            print(f"- max_float: np.finfo(np.float32).max   （最大浮点数 ≈ {np.finfo(np.float32).max:.7e}）")
        else:
            print("\nBF16 测试值定义（trunc：towards zero）：")
            print("- nan: float('nan')")
            print("- inf: float('inf')")
            print("- -inf: float('-inf')")
            print("- zero: 0.0")
            print("- -zero: -0.0")
            print("- normal: 1.5, -normal: -1.5 （→ 1.0 / -1.0）")
            print("- positive_fraction: 2.9, negative_fraction: -2.9 （→ 2.0 / -2.0）")
            print("- positive_small: 0.9, negative_small: -0.9       （→ 0.0 / 0.0）")
            print("- positive_integer: 3.0, negative_integer: -3.0   （→ 3.0 / -3.0）")
            print("- positive_half: 2.5, negative_half: -2.5         （→ 2.0 / -2.0）")
            print("- large_positive: 999.99, large_negative: -999.99  （→ 999.0 / -999.0）")
            print("- subnormal: 1e-40 / -subnormal: -1e-40")
            print("- min_normal: 1.18e-38")
            print("- near_overflow: 3.0e38 / -near_overflow: -3.0e38")
            print("- max_float: 3.39e38")

    def create_test_values(self) -> Dict[str, jnp.ndarray]:
        """创建各种测试值"""
        values = {}

        if self.dtype == jnp.float32:
            # fp32特殊值
            values['nan'] = jnp.array(float('nan'), dtype=self.dtype)
            values['inf'] = jnp.array(float('inf'), dtype=self.dtype)
            values['-inf'] = jnp.array(float('-inf'), dtype=self.dtype)
            values['zero'] = jnp.array(0.0, dtype=self.dtype)
            values['-zero'] = jnp.array(-0.0, dtype=self.dtype)

            # fp32 normal值
            values['normal'] = jnp.array(1.5, dtype=self.dtype)
            values['-normal'] = jnp.array(-1.5, dtype=self.dtype)

            # trunc测试的特殊值 (truncate towards zero)
            values['positive_fraction'] = jnp.array(2.9, dtype=self.dtype)  # -> 2.0
            values['negative_fraction'] = jnp.array(-2.9, dtype=self.dtype)  # -> -2.0
            values['positive_small'] = jnp.array(0.9, dtype=self.dtype)  # -> 0.0
            values['negative_small'] = jnp.array(-0.9, dtype=self.dtype)  # -> 0.0
            values['positive_integer'] = jnp.array(3.0, dtype=self.dtype)  # -> 3.0
            values['negative_integer'] = jnp.array(-3.0, dtype=self.dtype)  # -> -3.0
            values['positive_half'] = jnp.array(2.5, dtype=self.dtype)  # -> 2.0
            values['negative_half'] = jnp.array(-2.5, dtype=self.dtype)  # -> -2.0
            values['large_positive'] = jnp.array(999.99, dtype=self.dtype)  # -> 999.0
            values['large_negative'] = jnp.array(-999.99, dtype=self.dtype)  # -> -999.0

            # fp32 subnormal值
            values['subnormal'] = jnp.array(1e-40, dtype=self.dtype)
            values['-subnormal'] = jnp.array(-1e-40, dtype=self.dtype)

            # fp32 边界值
            values['min_normal'] = jnp.array(np.finfo(np.float32).tiny, dtype=self.dtype)
            values['near_overflow'] = jnp.array(1e38, dtype=self.dtype)
            values['-near_overflow'] = jnp.array(-1e38, dtype=self.dtype)
            values['max_float'] = jnp.array(np.finfo(np.float32).max, dtype=self.dtype)

        elif self.dtype == jnp.bfloat16:
            # bf16特殊值
            values['nan'] = jnp.array(float('nan'), dtype=self.dtype)
            values['inf'] = jnp.array(float('inf'), dtype=self.dtype)
            values['-inf'] = jnp.array(float('-inf'), dtype=self.dtype)
            values['zero'] = jnp.array(0.0, dtype=self.dtype)
            values['-zero'] = jnp.array(-0.0, dtype=self.dtype)

            # bf16 normal值
            values['normal'] = jnp.array(1.5, dtype=self.dtype)
            values['-normal'] = jnp.array(-1.5, dtype=self.dtype)

            # trunc测试的特殊值
            values['positive_fraction'] = jnp.array(2.9, dtype=self.dtype)
            values['negative_fraction'] = jnp.array(-2.9, dtype=self.dtype)
            values['positive_small'] = jnp.array(0.9, dtype=self.dtype)
            values['negative_small'] = jnp.array(-0.9, dtype=self.dtype)
            values['positive_integer'] = jnp.array(3.0, dtype=self.dtype)
            values['negative_integer'] = jnp.array(-3.0, dtype=self.dtype)
            values['positive_half'] = jnp.array(2.5, dtype=self.dtype)
            values['negative_half'] = jnp.array(-2.5, dtype=self.dtype)
            values['large_positive'] = jnp.array(999.99, dtype=self.dtype)
            values['large_negative'] = jnp.array(-999.99, dtype=self.dtype)

            # bf16 subnormal值
            values['subnormal'] = jnp.array(1e-40, dtype=self.dtype)
            values['-subnormal'] = jnp.array(-1e-40, dtype=self.dtype)

            # bf16 边界值
            values['min_normal'] = jnp.array(1.18e-38, dtype=self.dtype)
            values['near_overflow'] = jnp.array(3.0e38, dtype=self.dtype)
            values['-near_overflow'] = jnp.array(-3.0e38, dtype=self.dtype)
            values['max_float'] = jnp.array(3.39e38, dtype=self.dtype)

        return values

    def format_result(self, result: jnp.ndarray) -> str:
        """格式化结果输出"""
        result_scalar = float(result)

        if jnp.isnan(result):
            return "nan"
        elif jnp.isinf(result):
            if result_scalar > 0:
                return "inf"
            else:
                return "-inf"
        elif result_scalar == 0:
            # 检查是否为-0
            if jnp.signbit(result):
                return "-zero"
            else:
                return "zero"
        else:
            # 检查是否为subnormal
            abs_val = abs(result_scalar)
            if self.dtype == jnp.float32:
                if 0 < abs_val < np.finfo(np.float32).tiny:
                    return f"subnormal({result_scalar:.6e})"
            elif self.dtype == jnp.bfloat16:
                if 0 < abs_val < 1.18e-38:
                    return f"subnormal({result_scalar:.6e})"

            return f"{result_scalar:.6e}"

    def test_trunc_operations(self) -> List[Tuple[str, str]]:
        """测试trunc运算 - truncate towards zero"""
        values = self.create_test_values()
        results = []

        # 单操作数测试用例
        test_cases = [
            'nan',
            'inf',
            '-inf',
            'zero',
            '-zero',
            'normal',  # 1.5 -> 1.0
            '-normal',  # -1.5 -> -1.0
            'positive_fraction',  # 2.9 -> 2.0
            'negative_fraction',  # -2.9 -> -2.0
            'positive_small',  # 0.9 -> 0.0
            'negative_small',  # -0.9 -> 0.0
            'positive_integer',  # 3.0 -> 3.0
            'negative_integer',  # -3.0 -> -3.0
            'positive_half',  # 2.5 -> 2.0
            'negative_half',  # -2.5 -> -2.0
            'large_positive',  # 999.99 -> 999.0
            'large_negative',  # -999.99 -> -999.0
            'subnormal',
            '-subnormal',
            'near_overflow',
            '-near_overflow',
            'max_float',
        ]

        # JIT编译的trunc函数
        @jax.jit
        def trunc_op(x):
            # Truncate towards zero
            # JAX的trunc函数直接提供了这个功能
            return jnp.trunc(x)

        for val_name in test_cases:
            if val_name in values:
                val = values[val_name]

                # 执行trunc运算
                result = trunc_op(val)

                # 格式化结果
                result_str = self.format_result(result)
                results.append((val_name, result_str))

        return results

    def print_results(self, results: List[Tuple[str, str]], operation: str = "trunc"):
        """打印结果"""
        dtype_str = "fp32" if self.dtype == jnp.float32 else "bf16"

        print(f"\n{'=' * 60}")
        print(f"GPU Float Operation Test")
        print(f"Dtype: {dtype_str}")
        print(f"Operation: {operation}")
        print(f"{'=' * 60}\n")

        for val, result in results:
            print(f"{operation}({val}) = {result}")


def main():
    """在GPU上运行所有测试"""
    print("Starting GPU Float TRUNC Operation Tests")
    print(f"JAX version: {jax.__version__}")

    # 测试fp32
    print("\n" + "#" * 80)
    print("Testing FP32")
    print("#" * 80)

    fp32_tester = GPUFloatOperationTester(dtype=jnp.float32)
    fp32_tester.print_test_values_description()
    fp32_results = fp32_tester.test_trunc_operations()
    fp32_tester.print_results(fp32_results, "trunc")

    # 测试bf16
    print("\n" + "#" * 80)
    print("Testing BF16")
    print("#" * 80)

    bf16_tester = GPUFloatOperationTester(dtype=jnp.bfloat16)
    bf16_tester.print_test_values_description()
    bf16_results = bf16_tester.test_trunc_operations()
    bf16_tester.print_results(bf16_results, "trunc")

    # 比较fp32和bf16的结果差异
    print("\n" + "=" * 80)
    print("Comparing FP32 vs BF16 Results")
    print("=" * 80)

    fp32_dict = {f"trunc({v})": res for v, res in fp32_results}
    bf16_dict = {f"trunc({v})": res for v, res in bf16_results}

    differences = []
    for key in fp32_dict:
        if key in bf16_dict and fp32_dict[key] != bf16_dict[key]:
            differences.append((key, fp32_dict[key], bf16_dict[key]))

    if differences:
        print("\nDifferences found:")
        for op, fp32_res, bf16_res in differences:
            print(f"{op}:")
            print(f"  FP32: {fp32_res}")
            print(f"  BF16: {bf16_res}")
    else:
        print("\nNo differences between FP32 and BF16 results!")


if __name__ == "__main__":
    main()