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
        """打印 remap 测试值定义：remap(v, in_min, in_max, out_min, out_max)
        公式：out = (v - in_min)/(in_max - in_min) * (out_max - out_min) + out_min"""
        if self.dtype == jnp.float32:
            print("\nFP32 测试值定义（remap：按线性区间映射，支持反向区间与外推）：")
            print("- nan / inf / -inf / zero / -zero / normal / -normal")
            print("- in_value_1: 0.5   （位于输入区间内）")
            print("- in_value_2: 1.0   （位于输入区间内）")
            print("- in_value_3: 2.5   （超出输入上界，用于外推）")
            print("- in_min: 0.0, in_max: 2.0        （输入区间）")
            print("- out_min: -1.0, out_max: 1.0     （输出区间）")
            print("- subnormal: 1e-40 / -subnormal: -1e-40")
            print(f"- min_normal: np.finfo(np.float32).tiny （最小正规数 ≈ {np.finfo(np.float32).tiny:.8e})")
            print("- near_overflow: ±1e38, max_float: np.finfo(np.float32).max")
        else:
            print("\nBF16 测试值定义（remap：按线性区间映射，支持反向区间与外推）：")
            print("- nan / inf / -inf / zero / -zero / normal / -normal")
            print("- in_value_1: 0.5")
            print("- in_value_2: 1.0")
            print("- in_value_3: 2.5   （超出输入上界，用于外推）")
            print("- in_min: 0.0, in_max: 2.0")
            print("- out_min: -1.0, out_max: 1.0")
            print("- subnormal: 1e-40 / -subnormal: -1e-40")
            print("- min_normal: 1.18e-38")
            print("- near_overflow: ±3.0e38, max_float: 3.39e38")

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

            # remap测试用的特殊值
            # remap通常用于将一个范围映射到另一个范围
            values['in_value_1'] = jnp.array(0.5, dtype=self.dtype)
            values['in_value_2'] = jnp.array(1.0, dtype=self.dtype)
            values['in_value_3'] = jnp.array(2.5, dtype=self.dtype)

            # 输入范围
            values['in_min'] = jnp.array(0.0, dtype=self.dtype)
            values['in_max'] = jnp.array(2.0, dtype=self.dtype)

            # 输出范围
            values['out_min'] = jnp.array(-1.0, dtype=self.dtype)
            values['out_max'] = jnp.array(1.0, dtype=self.dtype)

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

            # remap测试用的特殊值
            values['in_value_1'] = jnp.array(0.5, dtype=self.dtype)
            values['in_value_2'] = jnp.array(1.0, dtype=self.dtype)
            values['in_value_3'] = jnp.array(2.5, dtype=self.dtype)

            # 输入范围
            values['in_min'] = jnp.array(0.0, dtype=self.dtype)
            values['in_max'] = jnp.array(2.0, dtype=self.dtype)

            # 输出范围
            values['out_min'] = jnp.array(-1.0, dtype=self.dtype)
            values['out_max'] = jnp.array(1.0, dtype=self.dtype)

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

    def test_remap_operations(self) -> List[Tuple[str, str, str, str, str, str]]:
        """测试remap运算 - remap(value, in_min, in_max, out_min, out_max)"""
        values = self.create_test_values()
        results = []

        # remap测试用例 - (value, in_min, in_max, out_min, out_max)
        test_cases = [
            # 正常映射测试
            ('in_value_1', 'in_min', 'in_max', 'out_min', 'out_max'),  # 0.5 in [0,2] -> -0.5 in [-1,1]
            ('in_value_2', 'in_min', 'in_max', 'out_min', 'out_max'),  # 1.0 in [0,2] -> 0.0 in [-1,1]
            ('in_min', 'in_min', 'in_max', 'out_min', 'out_max'),  # 0.0 in [0,2] -> -1.0 in [-1,1]
            ('in_max', 'in_min', 'in_max', 'out_min', 'out_max'),  # 2.0 in [0,2] -> 1.0 in [-1,1]

            # 超出范围的值（外推）
            ('in_value_3', 'in_min', 'in_max', 'out_min', 'out_max'),  # 2.5 in [0,2] -> 1.5 in [-1,1]
            ('-normal', 'in_min', 'in_max', 'out_min', 'out_max'),  # -1.5 in [0,2] -> -2.5 in [-1,1]

            # 反向映射
            ('normal', 'in_max', 'in_min', 'out_min', 'out_max'),  # 反向输入范围
            ('normal', 'in_min', 'in_max', 'out_max', 'out_min'),  # 反向输出范围

            # 特殊值测试
            ('nan', 'in_min', 'in_max', 'out_min', 'out_max'),
            ('inf', 'in_min', 'in_max', 'out_min', 'out_max'),
            ('-inf', 'in_min', 'in_max', 'out_min', 'out_max'),
            ('zero', 'in_min', 'in_max', 'out_min', 'out_max'),
            ('-zero', 'in_min', 'in_max', 'out_min', 'out_max'),

            # 边界包含特殊值
            ('normal', 'nan', 'in_max', 'out_min', 'out_max'),
            ('normal', 'in_min', 'nan', 'out_min', 'out_max'),
            ('normal', 'in_min', 'in_max', 'nan', 'out_max'),
            ('normal', 'in_min', 'in_max', 'out_min', 'nan'),

            # 相同的输入输出范围
            ('normal', 'in_min', 'in_max', 'in_min', 'in_max'),

            # subnormal值测试
            ('subnormal', 'in_min', 'in_max', 'out_min', 'out_max'),
            ('-subnormal', 'in_min', 'in_max', 'out_min', 'out_max'),
        ]

        # JIT编译的remap函数
        @jax.jit
        def remap_op(value, in_min, in_max, out_min, out_max):
            # remap公式: out = (value - in_min) / (in_max - in_min) * (out_max - out_min) + out_min
            # 处理除零情况
            in_range = in_max - in_min
            out_range = out_max - out_min

            # 避免除零
            normalized = jnp.where(
                in_range != 0,
                (value - in_min) / in_range,
                jnp.zeros_like(value)
            )

            return normalized * out_range + out_min

        for val_name, in_min_name, in_max_name, out_min_name, out_max_name in test_cases:
            if all(name in values for name in [val_name, in_min_name, in_max_name, out_min_name, out_max_name]):
                val = values[val_name]
                in_min = values[in_min_name]
                in_max = values[in_max_name]
                out_min = values[out_min_name]
                out_max = values[out_max_name]

                # 执行remap运算
                result = remap_op(val, in_min, in_max, out_min, out_max)

                # 格式化结果
                result_str = self.format_result(result)
                results.append((val_name, in_min_name, in_max_name, out_min_name, out_max_name, result_str))

        return results

    def print_results(self, results: List[Tuple[str, str, str, str, str, str]], operation: str = "remap"):
        """打印结果"""
        dtype_str = "fp32" if self.dtype == jnp.float32 else "bf16"

        print(f"\n{'=' * 60}")
        print(f"GPU Float Operation Test")
        print(f"Dtype: {dtype_str}")
        print(f"Operation: {operation}")
        print(f"{'=' * 60}\n")

        for val, in_min, in_max, out_min, out_max, result in results:
            print(f"{operation}({val}, [{in_min}, {in_max}] -> [{out_min}, {out_max}]) = {result}")


def main():
    """在GPU上运行所有测试"""
    print("Starting GPU Float REMAP Operation Tests")
    print(f"JAX version: {jax.__version__}")

    # 测试fp32
    print("\n" + "#" * 80)
    print("Testing FP32")
    print("#" * 80)

    fp32_tester = GPUFloatOperationTester(dtype=jnp.float32)
    fp32_tester.print_test_values_description()
    fp32_results = fp32_tester.test_remap_operations()
    fp32_tester.print_results(fp32_results, "remap")

    # 测试bf16
    print("\n" + "#" * 80)
    print("Testing BF16")
    print("#" * 80)

    bf16_tester = GPUFloatOperationTester(dtype=jnp.bfloat16)
    bf16_tester.print_test_values_description()
    bf16_results = bf16_tester.test_remap_operations()
    bf16_tester.print_results(bf16_results, "remap")

    # 比较fp32和bf16的结果差异
    print("\n" + "=" * 80)
    print("Comparing FP32 vs BF16 Results")
    print("=" * 80)

    fp32_dict = {f"remap({v}, [{im}, {iM}] -> [{om}, {oM}])": res
                 for v, im, iM, om, oM, res in fp32_results}
    bf16_dict = {f"remap({v}, [{im}, {iM}] -> [{om}, {oM}])": res
                 for v, im, iM, om, oM, res in bf16_results}

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