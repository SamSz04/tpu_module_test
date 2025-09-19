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
        """打印测试值定义说明"""
        if self.dtype == jnp.float32:
            print("\nFP32 测试值定义:")
            print("- nan: float('nan')")
            print("- inf: float('inf')")
            print("- -inf: float('-inf')")
            print("- zero: 0.0")
            print("- -zero: -0.0")
            print("- normal: 1.5")
            print("- -normal: -1.5")
            print("- below_min: -5.0 (低于最小边界)")
            print("- above_max: 5.0 (高于最大边界)")
            print("- in_range: 0.5 (在范围内)")
            print("- min_bound: -2.0 (最小边界)")
            print("- max_bound: 2.0 (最大边界)")
            print("- subnormal: 1e-40 (次正规数)")
            print("- -subnormal: -1e-40")
            print(f"- min_normal: np.finfo(np.float32).tiny (最小正规数 ≈ {np.finfo(np.float32).tiny:.8e})")
            print("- near_overflow: 1e38 (接近溢出的值)")
            print("- -near_overflow: -1e38")
            print(f"- max_float: np.finfo(np.float32).max (最大浮点数 ≈ {np.finfo(np.float32).max:.7e})")
        else:  # bfloat16
            print("\nBF16 测试值定义:")
            print("- nan: float('nan')")
            print("- inf: float('inf')")
            print("- -inf: float('-inf')")
            print("- zero: 0.0")
            print("- -zero: -0.0")
            print("- normal: 1.5")
            print("- -normal: -1.5")
            print("- below_min: -5.0 (低于最小边界)")
            print("- above_max: 5.0 (高于最大边界)")
            print("- in_range: 0.5 (在范围内)")
            print("- min_bound: -2.0 (最小边界)")
            print("- max_bound: 2.0 (最大边界)")
            print("- subnormal: 1e-40 (次正规数)")
            print("- -subnormal: -1e-40")
            print("- min_normal: 1.18e-38 (最小正规数)")
            print("- near_overflow: 3.0e38 (接近溢出的值)")
            print("- -near_overflow: -3.0e38")
            print("- max_float: 3.39e38 (最大浮点数)")

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

            # clamp测试用的特殊值
            values['below_min'] = jnp.array(-5.0, dtype=self.dtype)
            values['above_max'] = jnp.array(5.0, dtype=self.dtype)
            values['in_range'] = jnp.array(0.5, dtype=self.dtype)

            # clamp边界值
            values['min_bound'] = jnp.array(-2.0, dtype=self.dtype)
            values['max_bound'] = jnp.array(2.0, dtype=self.dtype)

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

            # clamp测试用的特殊值
            values['below_min'] = jnp.array(-5.0, dtype=self.dtype)
            values['above_max'] = jnp.array(5.0, dtype=self.dtype)
            values['in_range'] = jnp.array(0.5, dtype=self.dtype)

            # clamp边界值
            values['min_bound'] = jnp.array(-2.0, dtype=self.dtype)
            values['max_bound'] = jnp.array(2.0, dtype=self.dtype)

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

    def test_clamp_operations(self) -> List[Tuple[str, str, str, str]]:
        """测试clamp运算 - clamp(x, min, max)"""
        values = self.create_test_values()
        results = []

        # clamp测试用例 - (value, min, max)
        test_cases = [
            # 特殊值测试
            ('nan', 'min_bound', 'max_bound'),
            ('inf', 'min_bound', 'max_bound'),
            ('-inf', 'min_bound', 'max_bound'),

            # 正常clamp测试
            ('below_min', 'min_bound', 'max_bound'),  # -5.0 clamp to [-2, 2] -> -2
            ('above_max', 'min_bound', 'max_bound'),  # 5.0 clamp to [-2, 2] -> 2
            ('in_range', 'min_bound', 'max_bound'),  # 0.5 clamp to [-2, 2] -> 0.5

            # 边界值测试
            ('min_bound', 'min_bound', 'max_bound'),  # -2.0 clamp to [-2, 2] -> -2
            ('max_bound', 'min_bound', 'max_bound'),  # 2.0 clamp to [-2, 2] -> 2

            # zero值测试
            ('zero', 'min_bound', 'max_bound'),
            ('-zero', 'min_bound', 'max_bound'),

            # subnormal值测试
            ('subnormal', 'min_bound', 'max_bound'),
            ('-subnormal', 'min_bound', 'max_bound'),

            # 特殊的min/max组合
            ('normal', '-inf', 'inf'),  # 无限制范围
            ('normal', 'nan', 'max_bound'),  # min是nan
            ('normal', 'min_bound', 'nan'),  # max是nan

            # 溢出值测试
            ('near_overflow', 'min_bound', 'max_bound'),
            ('-near_overflow', 'min_bound', 'max_bound'),
        ]

        # JIT编译的clamp函数
        @jax.jit
        def clamp_op(x, min_val, max_val):
            return jnp.clip(x, min_val, max_val)

        for val_name, min_name, max_name in test_cases:
            if val_name in values and min_name in values and max_name in values:
                val = values[val_name]
                min_val = values[min_name]
                max_val = values[max_name]

                # 执行clamp运算
                result = clamp_op(val, min_val, max_val)

                # 格式化结果
                result_str = self.format_result(result)
                results.append((val_name, min_name, max_name, result_str))

        return results

    def print_results(self, results: List[Tuple[str, str, str, str]], operation: str = "clamp"):
        """打印结果"""
        dtype_str = "fp32" if self.dtype == jnp.float32 else "bf16"

        print(f"\n{'=' * 60}")
        print(f"GPU Float Operation Test")
        print(f"Dtype: {dtype_str}")
        print(f"Operation: {operation}")
        print(f"{'=' * 60}\n")

        for val, min_val, max_val, result in results:
            print(f"{operation}({val}, {min_val}, {max_val}) = {result}")


def main():
    """在GPU上运行所有测试"""
    print("Starting GPU Float CLAMP Operation Tests")
    print(f"JAX version: {jax.__version__}")

    # 测试fp32
    print("\n" + "#" * 80)
    print("Testing FP32")
    print("#" * 80)

    fp32_tester = GPUFloatOperationTester(dtype=jnp.float32)
    fp32_tester.print_test_values_description()
    fp32_results = fp32_tester.test_clamp_operations()
    fp32_tester.print_results(fp32_results, "clamp")

    # 测试bf16
    print("\n" + "#" * 80)
    print("Testing BF16")
    print("#" * 80)

    bf16_tester = GPUFloatOperationTester(dtype=jnp.bfloat16)
    bf16_tester.print_test_values_description()
    bf16_results = bf16_tester.test_clamp_operations()
    bf16_tester.print_results(bf16_results, "clamp")

    # 比较fp32和bf16的结果差异
    print("\n" + "=" * 80)
    print("Comparing FP32 vs BF16 Results")
    print("=" * 80)

    fp32_dict = {f"clamp({v}, {m}, {M})": res for v, m, M, res in fp32_results}
    bf16_dict = {f"clamp({v}, {m}, {M})": res for v, m, M, res in bf16_results}

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