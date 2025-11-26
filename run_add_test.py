import os
import pandas as pd
import jax
import jax.numpy as jnp
from typing import Dict
import tpu_common

# 设置 JAX 运行平台
os.environ['JAX_PLATFORM_NAME'] = 'tpu'


class TPUAddTester:
    def __init__(self, mode='fp32'):
        tpu_common.check_tpu()
        self.value_registry: Dict[str, jnp.ndarray] = {}
        self.mode = mode

        # 根据模式选择对应的 Type 和 Dtype
        if mode == 'bf16':
            self.dtype = jnp.bfloat16
            self.definitions = tpu_common.HARDCODED_VALUES_BF16
            print("Mode: BF16 (bfloat16)")
        else:
            self.dtype = jnp.float32
            self.definitions = tpu_common.HARDCODED_VALUES_FP32
            print("Mode: FP32 (float32)")

        self._initialize_registry()

    def _initialize_registry(self):
        print("Initializing value registry...")
        for name, hex_val in self.definitions.items():
            # 调用通用的 hex_to_jax，传入 self.dtype
            self.value_registry[name] = tpu_common.hex_to_jax(hex_val, self.dtype)
        print(f"Registered {len(self.value_registry)} values.")

    @staticmethod
    @jax.jit
    def run_kernel(x, y):
        """
        定义算子: Add
        JAX 的加法算子会自动处理 dtype (fp32+fp32 或 bf16+bf16)
        """
        return x + y

    def run_tests(self, input_file: str, output_file: str):
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' not found.")
            return

        print(f"Reading tests from {input_file}...")

        try:
            # 根据你提供的 CSV 格式，Headers 实际上在第2行 (index=1)
            # 使用 engine='openpyxl' 读取 xlsx
            df = pd.read_excel(input_file, header=1, engine='openpyxl')
        except ImportError:
            print("Error: Library 'openpyxl' is missing. Please run: pip install openpyxl")
            return
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            return

        # 简单的列名检查，防止读取错误的 sheet 或 header 偏移
        required_cols = ['x', 'y', 'exp_out']
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: Columns {required_cols} not found. Found: {df.columns}")
            print("Check if the header is correctly located at Row 2 (index 1).")
            return

        # 预热 JIT
        print("Compiling JAX kernel...")
        if not self.value_registry:
            print("Error: Value registry is empty.")
            return
        dummy = list(self.value_registry.values())[0]
        self.run_kernel(dummy, dummy)

        test_outs = []
        result_flags = []

        print(f"Processing {len(df)} test cases...")

        for idx, row in df.iterrows():
            x_name = row.get('x')
            y_name = row.get('y')

            # 跳过空行
            if pd.isna(x_name) or pd.isna(y_name):
                test_outs.append("")
                result_flags.append("")
                continue

            x_name = str(x_name).strip()
            y_name = str(y_name).strip()

            # 检查输入是否在注册表中
            if x_name not in self.value_registry:
                test_outs.append("ERROR_UNKNOWN_X")
                result_flags.append("Input Error")
                continue
            if y_name not in self.value_registry:
                test_outs.append("ERROR_UNKNOWN_Y")
                result_flags.append("Input Error")
                continue

            # 1. 运行 TPU 运算
            val_x = self.value_registry[x_name]
            val_y = self.value_registry[y_name]
            res = self.run_kernel(val_x, val_y)
            res.block_until_ready()

            # 2. 转换结果为 Hex
            actual_hex = tpu_common.jax_to_hex(res)
            test_outs.append(actual_hex)

            # 3. 验证逻辑 (填充 result 列)
            exp_hex = row.get('exp_out')
            if pd.isna(exp_hex):
                # 如果没有预期值，则不标记
                result_flags.append("")
            else:
                exp_hex = str(exp_hex).strip()

                # 归一化对比 (忽略大小写和 0x 前缀)
                def normalize(h):
                    return str(h).lower().replace('0x', '').replace(' ', '')

                if normalize(actual_hex) != normalize(exp_hex):
                    # 发现不一致，标记 MISMATCH
                    result_flags.append(f"MISMATCH (Exp: {exp_hex})")
                else:
                    # 一致，留空 (或者你可以改成 "PASS")
                    result_flags.append("")

        # 回填结果到 DataFrame
        df['test_out'] = test_outs
        df['result'] = result_flags

        # 保存为新的 Excel 文件
        try:
            df.to_excel(output_file, index=False)
            print(f"\nSuccess! Results saved to: {output_file}")
        except Exception as e:
            print(f"Error saving output file: {e}")

        # 打印简要统计
        mismatches = [r for r in result_flags if "MISMATCH" in str(r)]
        print(f"Summary:")
        print(f"  Total Test Cases: {len(test_outs)}")
        print(f"  Mismatches Found: {len(mismatches)}")
        if mismatches:
            print("  > Please check the 'result' column in the output file for details.")


if __name__ == "__main__":
    # 实例化两个测试器，分别测试 FP32 和 BF16
    tester_fp32 = TPUAddTester(mode='fp32')
    tester_fp32.run_tests("fp32_add.xlsx", "fp32_add_result.xlsx")

    tester_bf16 = TPUAddTester(mode='bf16')
    tester_bf16.run_tests("bf16_add.xlsx", "bf16_add_result.xlsx")