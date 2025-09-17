import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple

os.environ['JAX_PLATFORM_NAME'] = 'tpu'


class TPUFloatOperationTester:
    def __init__(self, dtype=jnp.float32):
        """
        初始化TPU测试器
        dtype: jnp.float32 or jnp.bfloat16
        """
        self.dtype = dtype

        # 确认在TPU上运行
        devices = jax.devices()
        print(f"JAX devices: {devices}")
        if not any('TPU' in str(d) for d in devices):
            print("WARNING: Not running on TPU!")

        # TPU信息
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
        """格式化结果输出 - 比较运算返回布尔值"""
        return "True" if bool(result) else "False"

    def test_eq_operations(self) -> List[Tuple[str, str, str]]:
        """测试相等运算"""
        values = self.create_test_values()
        results = []

        # 测试用例
        test_pairs = [
            # NaN comparisons - NaN不等于任何值，包括自己
            ('nan', 'nan'),
            ('nan', 'inf'),
            ('nan', '-inf'),
            ('nan', 'normal'),
            ('nan', 'zero'),

            # Infinity comparisons
            ('inf', 'inf'),
            ('inf', '-inf'),
            ('-inf', '-inf'),
            ('inf', 'normal'),

            # Zero comparisons - 注意+0和-0在IEEE标准中是相等的
            ('zero', 'zero'),
            ('zero', '-zero'),
            ('-zero', '-zero'),

            # Normal value comparisons
            ('normal', 'normal'),
            ('normal', '-normal'),
            ('normal', 'zero'),

            # Subnormal comparisons
            ('subnormal', 'subnormal'),
            ('subnormal', '-subnormal'),
            ('subnormal', 'zero'),

            # Boundary value comparisons
            ('near_overflow', 'near_overflow'),
            ('max_float', 'max_float'),
            ('min_normal', 'min_normal'),
        ]

        # JIT编译的相等比较函数
        @jax.jit
        def eq_op(x, y):
            return jnp.equal(x, y)

        for val1_name, val2_name in test_pairs:
            if val1_name in values and val2_name in values:
                val1 = values[val1_name]
                val2 = values[val2_name]

                # 执行相等比较
                result = eq_op(val1, val2)

                # 格式化结果
                result_str = self.format_result(result)
                results.append((val1_name, val2_name, result_str))

        return results

    def print_results(self, results: List[Tuple[str, str, str]], operation: str = "eq"):
        """打印结果"""
        dtype_str = "fp32" if self.dtype == jnp.float32 else "bf16"

        print(f"\n{'=' * 60}")
        print(f"TPU Float Operation Test")
        print(f"Dtype: {dtype_str}")
        print(f"Operation: {operation}")
        print(f"{'=' * 60}\n")

        for val1, val2, result in results:
            print(f"{val1} {operation} {val2} = {result}")


def main():
    """在TPU上运行所有测试"""
    print("Starting TPU Float EQ Operation Tests")
    print(f"JAX version: {jax.__version__}")

    # 测试fp32
    print("\n" + "#" * 80)
    print("Testing FP32")
    print("#" * 80)

    fp32_tester = TPUFloatOperationTester(dtype=jnp.float32)
    fp32_tester.print_test_values_description()
    fp32_results = fp32_tester.test_eq_operations()
    fp32_tester.print_results(fp32_results, "eq")

    # 测试bf16
    print("\n" + "#" * 80)
    print("Testing BF16")
    print("#" * 80)

    bf16_tester = TPUFloatOperationTester(dtype=jnp.bfloat16)
    bf16_tester.print_test_values_description()
    bf16_results = bf16_tester.test_eq_operations()
    bf16_tester.print_results(bf16_results, "eq")

    # 比较fp32和bf16的结果差异
    print("\n" + "=" * 80)
    print("Comparing FP32 vs BF16 Results")
    print("=" * 80)

    fp32_dict = {f"{v1} eq {v2}": res for v1, v2, res in fp32_results}
    bf16_dict = {f"{v1} eq {v2}": res for v1, v2, res in bf16_results}

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