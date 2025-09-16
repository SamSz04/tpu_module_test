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

            # clamps测试用的特殊值（对称clamp）
            values['large_positive'] = jnp.array(5.0, dtype=self.dtype)
            values['large_negative'] = jnp.array(-5.0, dtype=self.dtype)
            values['small_positive'] = jnp.array(0.5, dtype=self.dtype)
            values['small_negative'] = jnp.array(-0.5, dtype=self.dtype)

            # clamps的对称边界值（使用不同的值来测试对称性）
            values['sym_bound_1'] = jnp.array(2.0, dtype=self.dtype)
            values['sym_bound_2'] = jnp.array(3.0, dtype=self.dtype)
            values['sym_bound_3'] = jnp.array(4.0, dtype=self.dtype)

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

            # clamps测试用的特殊值
            values['large_positive'] = jnp.array(5.0, dtype=self.dtype)
            values['large_negative'] = jnp.array(-5.0, dtype=self.dtype)
            values['small_positive'] = jnp.array(0.5, dtype=self.dtype)
            values['small_negative'] = jnp.array(-0.5, dtype=self.dtype)

            # clamps的对称边界值
            values['sym_bound_1'] = jnp.array(2.0, dtype=self.dtype)
            values['sym_bound_2'] = jnp.array(3.0, dtype=self.dtype)
            values['sym_bound_3'] = jnp.array(4.0, dtype=self.dtype)

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

    def test_clamps_operations(self) -> List[Tuple[str, str, str, str]]:
        """测试clamps运算 - clamps(x, bound1, bound2) 使用对称边界"""
        values = self.create_test_values()
        results = []

        # clamps测试用例 - (value, bound1, bound2)
        # clamps会创建对称边界 [-max(|bound1|, |bound2|), max(|bound1|, |bound2|)]
        test_cases = [
            # 特殊值测试
            ('nan', 'sym_bound_1', 'sym_bound_2'),
            ('inf', 'sym_bound_1', 'sym_bound_2'),
            ('-inf', 'sym_bound_1', 'sym_bound_2'),

            # 正常clamps测试
            ('large_positive', 'sym_bound_1', 'sym_bound_2'),  # 5.0 clamps with bounds 2,3 -> 3.0
            ('large_negative', 'sym_bound_1', 'sym_bound_2'),  # -5.0 clamps with bounds 2,3 -> -3.0
            ('small_positive', 'sym_bound_1', 'sym_bound_2'),  # 0.5 remains 0.5
            ('small_negative', 'sym_bound_1', 'sym_bound_2'),  # -0.5 remains -0.5

            # 不同边界组合
            ('large_positive', 'sym_bound_2', 'sym_bound_1'),  # 测试参数顺序
            ('large_negative', 'sym_bound_3', 'sym_bound_1'),  # 更大的边界

            # 边界值测试
            ('sym_bound_2', 'sym_bound_1', 'sym_bound_2'),  # 值等于其中一个边界
            ('-normal', 'normal', 'sym_bound_1'),  # 混合边界

            # zero值测试
            ('zero', 'sym_bound_1', 'sym_bound_2'),
            ('-zero', 'sym_bound_1', 'sym_bound_2'),

            # subnormal值测试
            ('subnormal', 'sym_bound_1', 'sym_bound_2'),
            ('-subnormal', 'sym_bound_1', 'sym_bound_2'),

            # 特殊的边界值
            ('normal', 'nan', 'sym_bound_1'),
            ('normal', 'sym_bound_1', 'nan'),
            ('normal', 'inf', 'sym_bound_1'),
            ('normal', 'zero', 'sym_bound_1'),

            # 溢出值测试
            ('near_overflow', 'sym_bound_1', 'sym_bound_2'),
            ('-near_overflow', 'sym_bound_1', 'sym_bound_2'),
        ]

        # JIT编译的clamps函数
        @jax.jit
        def clamps_op(x, bound1, bound2):
            # clamps: 使用对称边界进行clamp
            # 取两个边界绝对值的最大值作为对称边界
            max_abs_bound = jnp.maximum(jnp.abs(bound1), jnp.abs(bound2))
            return jnp.clip(x, -max_abs_bound, max_abs_bound)

        for val_name, bound1_name, bound2_name in test_cases:
            if val_name in values and bound1_name in values and bound2_name in values:
                val = values[val_name]
                bound1 = values[bound1_name]
                bound2 = values[bound2_name]

                # 执行clamps运算
                result = clamps_op(val, bound1, bound2)

                # 格式化结果
                result_str = self.format_result(result)
                results.append((val_name, bound1_name, bound2_name, result_str))

        return results

    def print_results(self, results: List[Tuple[str, str, str, str]], operation: str = "clamps"):
        """打印结果"""
        dtype_str = "fp32" if self.dtype == jnp.float32 else "bf16"

        print(f"\n{'=' * 60}")
        print(f"TPU Float Operation Test")
        print(f"Dtype: {dtype_str}")
        print(f"Operation: {operation}")
        print(f"{'=' * 60}\n")

        for val, bound1, bound2, result in results:
            print(f"{operation}({val}, {bound1}, {bound2}) = {result}")


def main():
    """在TPU上运行所有测试"""
    print("Starting TPU Float CLAMPS Operation Tests")
    print(f"JAX version: {jax.__version__}")

    # 测试fp32
    print("\n" + "#" * 80)
    print("Testing FP32")
    print("#" * 80)

    fp32_tester = TPUFloatOperationTester(dtype=jnp.float32)
    fp32_results = fp32_tester.test_clamps_operations()
    fp32_tester.print_results(fp32_results, "clamps")

    # 测试bf16
    print("\n" + "#" * 80)
    print("Testing BF16")
    print("#" * 80)

    bf16_tester = TPUFloatOperationTester(dtype=jnp.bfloat16)
    bf16_results = bf16_tester.test_clamps_operations()
    bf16_tester.print_results(bf16_results, "clamps")

    # 比较fp32和bf16的结果差异
    print("\n" + "=" * 80)
    print("Comparing FP32 vs BF16 Results")
    print("=" * 80)

    fp32_dict = {f"clamps({v}, {b1}, {b2})": res for v, b1, b2, res in fp32_results}
    bf16_dict = {f"clamps({v}, {b1}, {b2})": res for v, b1, b2, res in bf16_results}

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