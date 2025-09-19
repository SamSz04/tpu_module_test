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
        """打印 round.rtna 测试值定义：四舍五入到最近，恰好 .5 时远离 0 舍入"""
        if self.dtype == jnp.float32:
            print("\nFP32 测试值定义（round.rtna：ties away from zero）：")
            print("- nan / inf / -inf / zero / -zero")
            print("- normal: 1.5       （→ 2.0）")
            print("- -normal: -1.5     （→ -2.0）")
            print("- positive_half: 2.5    （→ 3.0）")
            print("- negative_half: -2.5   （→ -3.0）")
            print("- positive_fraction: 2.3 （→ 2.0）")
            print("- negative_fraction: -2.3（→ -2.0）")
            print("- positive_integer: 3.0  （→ 3.0）")
            print("- negative_integer: -3.0 （→ -3.0）")
            print("- small_positive: 0.4    （→ 0.0）")
            print("- small_negative: -0.4   （→ 0.0）")
            print("- exact_half: 0.5        （→ 1.0）")
            print("- -exact_half: -0.5      （→ -1.0）")
            print("- subnormal: 1e-40 / -subnormal: -1e-40")
            print(f"- min_normal: np.finfo(np.float32).tiny （≈ {np.finfo(np.float32).tiny:.8e}）")
            print("- near_overflow: ±1e38, max_float: np.finfo(np.float32).max")
        else:
            print("\nBF16 测试值定义（round.rtna：ties away from zero）：")
            print("- nan / inf / -inf / zero / -zero")
            print("- normal: 1.5, -normal: -1.5")
            print("- positive_half: 2.5, negative_half: -2.5")
            print("- positive_fraction: 2.3, negative_fraction: -2.3")
            print("- positive_integer: 3.0, negative_integer: -3.0")
            print("- small_positive: 0.4, small_negative: -0.4")
            print("- exact_half: 0.5, -exact_half: -0.5")
            print("- subnormal: 1e-40 / -subnormal: -1e-40")
            print("- min_normal: 1.18e-38, near_overflow: ±3.0e38, max_float: 3.39e38")

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

            # round.rtna测试的特殊值 (round to nearest, ties away from zero)
            values['positive_half'] = jnp.array(2.5, dtype=self.dtype)  # -> 3.0
            values['negative_half'] = jnp.array(-2.5, dtype=self.dtype)  # -> -3.0
            values['positive_fraction'] = jnp.array(2.3, dtype=self.dtype)  # -> 2.0
            values['negative_fraction'] = jnp.array(-2.3, dtype=self.dtype)  # -> -2.0
            values['positive_integer'] = jnp.array(3.0, dtype=self.dtype)  # -> 3.0
            values['negative_integer'] = jnp.array(-3.0, dtype=self.dtype)  # -> -3.0
            values['small_positive'] = jnp.array(0.4, dtype=self.dtype)  # -> 0.0
            values['small_negative'] = jnp.array(-0.4, dtype=self.dtype)  # -> 0.0
            values['exact_half'] = jnp.array(0.5, dtype=self.dtype)  # -> 1.0
            values['-exact_half'] = jnp.array(-0.5, dtype=self.dtype)  # -> -1.0

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

            # round.rtna测试的特殊值
            values['positive_half'] = jnp.array(2.5, dtype=self.dtype)
            values['negative_half'] = jnp.array(-2.5, dtype=self.dtype)
            values['positive_fraction'] = jnp.array(2.3, dtype=self.dtype)
            values['negative_fraction'] = jnp.array(-2.3, dtype=self.dtype)
            values['positive_integer'] = jnp.array(3.0, dtype=self.dtype)
            values['negative_integer'] = jnp.array(-3.0, dtype=self.dtype)
            values['small_positive'] = jnp.array(0.4, dtype=self.dtype)
            values['small_negative'] = jnp.array(-0.4, dtype=self.dtype)
            values['exact_half'] = jnp.array(0.5, dtype=self.dtype)
            values['-exact_half'] = jnp.array(-0.5, dtype=self.dtype)

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

    def test_round_rtna_operations(self) -> List[Tuple[str, str]]:
        """测试round.rtna运算 - round to nearest, ties away from zero"""
        values = self.create_test_values()
        results = []

        # 单操作数测试用例
        test_cases = [
            'nan',
            'inf',
            '-inf',
            'zero',
            '-zero',
            'normal',  # 1.5 -> 2.0
            '-normal',  # -1.5 -> -2.0
            'positive_half',  # 2.5 -> 3.0
            'negative_half',  # -2.5 -> -3.0
            'positive_fraction',  # 2.3 -> 2.0
            'negative_fraction',  # -2.3 -> -2.0
            'positive_integer',  # 3.0 -> 3.0
            'negative_integer',  # -3.0 -> -3.0
            'small_positive',  # 0.4 -> 0.0
            'small_negative',  # -0.4 -> 0.0
            'exact_half',  # 0.5 -> 1.0
            '-exact_half',  # -0.5 -> -1.0
            'subnormal',
            '-subnormal',
            'near_overflow',
            '-near_overflow',
            'max_float',
        ]

        # JIT编译的round.rtna函数
        @jax.jit
        def round_rtna_op(x):
            # Round to nearest, ties away from zero
            # 对于0.5的情况，向远离0的方向舍入
            # JAX的round函数默认是round half to even，我们需要自定义实现

            # 获取符号
            sign = jnp.sign(x)
            abs_x = jnp.abs(x)

            # 计算小数部分
            floor_abs = jnp.floor(abs_x)
            frac = abs_x - floor_abs

            # 如果小数部分 >= 0.5，向上舍入；否则向下舍入
            result_abs = jnp.where(frac >= 0.5, floor_abs + 1, floor_abs)

            # 恢复符号
            return sign * result_abs

        for val_name in test_cases:
            if val_name in values:
                val = values[val_name]

                # 执行round.rtna运算
                result = round_rtna_op(val)

                # 格式化结果
                result_str = self.format_result(result)
                results.append((val_name, result_str))

        return results

    def print_results(self, results: List[Tuple[str, str]], operation: str = "round.rtna"):
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
    print("Starting GPU Float ROUND.RTNA Operation Tests")
    print(f"JAX version: {jax.__version__}")

    # 测试fp32
    print("\n" + "#" * 80)
    print("Testing FP32")
    print("#" * 80)

    fp32_tester = GPUFloatOperationTester(dtype=jnp.float32)
    fp32_tester.print_test_values_description()
    fp32_results = fp32_tester.test_round_rtna_operations()
    fp32_tester.print_results(fp32_results, "round.rtna")

    # 测试bf16
    print("\n" + "#" * 80)
    print("Testing BF16")
    print("#" * 80)

    bf16_tester = GPUFloatOperationTester(dtype=jnp.bfloat16)
    bf16_tester.print_test_values_description()
    bf16_results = bf16_tester.test_round_rtna_operations()
    bf16_tester.print_results(bf16_results, "round.rtna")

    # 比较fp32和bf16的结果差异
    print("\n" + "=" * 80)
    print("Comparing FP32 vs BF16 Results")
    print("=" * 80)

    fp32_dict = {f"round.rtna({v})": res for v, res in fp32_results}
    bf16_dict = {f"round.rtna({v})": res for v, res in bf16_results}

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