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

            # bf16 边界值 - 使用已知的bfloat16范围值
            # bfloat16的范围大约是: 最小正数 ≈ 1.18e-38, 最大值 ≈ 3.39e38
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
                # bfloat16的最小正规数大约是1.18e-38
                if 0 < abs_val < 1.18e-38:
                    return f"subnormal({result_scalar:.6e})"

            return f"{result_scalar:.6e}"

    def test_sub_operations(self) -> List[Tuple[str, str, str]]:
        """测试减法运算"""
        values = self.create_test_values()
        results = []

        # 测试用例
        test_pairs = [
            # NaN propagation
            ('nan', 'nan'),
            ('nan', 'inf'),
            ('nan', '-inf'),
            ('nan', 'normal'),
            ('nan', 'zero'),
            ('nan', 'subnormal'),

            # Infinity arithmetic
            ('inf', 'inf'),
            ('inf', '-inf'),
            ('-inf', '-inf'),
            ('inf', 'normal'),
            ('inf', 'zero'),

            # Zero arithmetic
            ('zero', 'zero'),
            ('zero', '-zero'),
            ('-zero', '-zero'),

            # Normal values
            ('normal', 'normal'),
            ('normal', '-normal'),

            # Subnormal handling
            ('subnormal', 'subnormal'),
            ('subnormal', '-subnormal'),
            ('subnormal', 'normal'),
            ('subnormal', 'zero'),

            # Overflow cases
            ('near_overflow', 'near_overflow'),
            ('max_float', 'max_float'),
            ('near_overflow', 'normal'),
        ]

        # JIT编译的减法函数
        @jax.jit
        def sub_op(x, y):
            return x - y

        for val1_name, val2_name in test_pairs:
            if val1_name in values and val2_name in values:
                val1 = values[val1_name]
                val2 = values[val2_name]

                # 执行减法
                result = sub_op(val1, val2)

                # 格式化结果
                result_str = self.format_result(result)
                results.append((val1_name, val2_name, result_str))

        return results

    def print_results(self, results: List[Tuple[str, str, str]], operation: str = "sub"):
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
    print("Starting TPU Float Operation Tests")
    print(f"JAX version: {jax.__version__}")

    # 测试fp32
    print("\n" + "#" * 80)
    print("Testing FP32")
    print("#" * 80)

    fp32_tester = TPUFloatOperationTester(dtype=jnp.float32)
    fp32_results = fp32_tester.test_sub_operations()
    fp32_tester.print_results(fp32_results, "sub")

    # 测试bf16
    print("\n" + "#" * 80)
    print("Testing BF16")
    print("#" * 80)

    bf16_tester = TPUFloatOperationTester(dtype=jnp.bfloat16)
    bf16_results = bf16_tester.test_sub_operations()
    bf16_tester.print_results(bf16_results, "sub")

    # 比较fp32和bf16的结果差异
    print("\n" + "=" * 80)
    print("Comparing FP32 vs BF16 Results")
    print("=" * 80)

    fp32_dict = {f"{v1} sub {v2}": res for v1, v2, res in fp32_results}
    bf16_dict = {f"{v1} sub {v2}": res for v1, v2, res in bf16_results}

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