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
        """打印测试值定义说明（乘法相关）"""
        if self.dtype == jnp.float32:
            print("\nFP32 测试值定义（mul）：")
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
            print("\nBF16 测试值定义（mul）：")
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

    def test_mul_operations(self) -> List[Tuple[str, str, str]]:
        """测试乘法运算"""
        values = self.create_test_values()
        results = []

        # 测试用例（覆盖 NaN 传播、无穷大规则、零的符号、次正规数、溢出/下溢）
        test_pairs = [
            # NaN propagation
            ('nan', 'nan'),
            ('nan', 'inf'),
            ('nan', '-inf'),
            ('nan', 'normal'),
            ('nan', 'zero'),
            ('nan', 'subnormal'),

            # Infinity arithmetic
            ('inf', 'inf'),          # -> inf
            ('inf', '-inf'),         # -> -inf
            ('-inf', '-inf'),        # -> inf
            ('inf', 'normal'),       # -> inf
            ('-inf', 'normal'),      # -> -inf
            ('inf', 'zero'),         # -> nan (IEEE: inf * 0 = NaN)
            ('-inf', 'zero'),        # -> nan
            ('inf', '-zero'),        # -> nan
            ('-inf', '-zero'),       # -> nan

            # Zero arithmetic & sign
            ('zero', 'zero'),        # 0 * 0
            ('zero', '-zero'),       # 0 * (-0)
            ('-zero', '-zero'),      # (-0) * (-0)
            ('zero', 'normal'),      # 0 * x
            ('-zero', 'normal'),     # (-0) * x  -> 可能得到 -0
            ('zero', '-normal'),     # 0 * (-x)  -> 可能得到 -0

            # Normal values & sign
            ('normal', 'normal'),    # 1.5 * 1.5
            ('normal', '-normal'),   # 1.5 * (-1.5)

            # Subnormal handling
            ('subnormal', 'subnormal'),     # 极小 * 极小 -> 可能下溢到 0/±0
            ('subnormal', 'normal'),
            ('-subnormal', 'normal'),

            # Mix with bounds
            ('near_overflow', 'normal'),     # 1e38 * 1.5（fp32 不溢出；bf16 3e38 * 1.5 可能溢出）
            ('near_overflow', 'near_overflow'),  # 1e38 * 1e38 -> inf
            ('max_float', 'normal'),         # max * 1.5 -> inf
        ]

        # JIT编译的乘法函数
        @jax.jit
        def mul_op(x, y):
            return x * y

        for val1_name, val2_name in test_pairs:
            if val1_name in values and val2_name in values:
                val1 = values[val1_name]
                val2 = values[val2_name]

                # 执行乘法
                result = mul_op(val1, val2)

                # 格式化结果
                result_str = self.format_result(result)
                results.append((val1_name, val2_name, result_str))

        return results

    def print_results(self, results: List[Tuple[str, str, str]], operation: str = "mul"):
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
    print("Starting TPU Float MUL Operation Tests")
    print(f"JAX version: {jax.__version__}")

    # 测试fp32
    print("\n" + "#" * 80)
    print("Testing FP32")
    print("#" * 80)

    fp32_tester = TPUFloatOperationTester(dtype=jnp.float32)
    fp32_tester.print_test_values_description()
    fp32_results = fp32_tester.test_mul_operations()
    fp32_tester.print_results(fp32_results, "mul")

    # 测试bf16
    print("\n" + "#" * 80)
    print("Testing BF16")
    print("#" * 80)

    bf16_tester = TPUFloatOperationTester(dtype=jnp.bfloat16)
    bf16_tester.print_test_values_description()
    bf16_results = bf16_tester.test_mul_operations()
    bf16_tester.print_results(bf16_results, "mul")

    # 比较fp32和bf16的结果差异
    print("\n" + "=" * 80)
    print("Comparing FP32 vs BF16 Results")
    print("=" * 80)

    fp32_dict = {f"{v1} mul {v2}": res for v1, v2, res in fp32_results}
    bf16_dict = {f"{v1} mul {v2}": res for v1, v2, res in bf16_results}

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
