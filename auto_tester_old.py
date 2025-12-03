import os
import glob
import re
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()


# ==========================================
# 1. 通用工具：Hex 与 JAX 转换
# ==========================================
def hex_to_jax(hex_str, dtype):
    """通用 Hex -> JAX Array 转换"""
    clean_hex = str(hex_str).strip().lower()
    if pd.isna(hex_str) or not clean_hex or clean_hex == 'nan':
        return jnp.array(0, dtype=dtype)

    if clean_hex.startswith('0x'): clean_hex = clean_hex[2:]

    # 移除可能存在的空格 (如 "FFFF FFFF")
    clean_hex = clean_hex.replace(' ', '')

    int_val = int(clean_hex, 16)

    # 根据 dtype 决定 bitcast 的源类型
    if dtype == jnp.float32:
        return lax.bitcast_convert_type(jnp.array(int_val, dtype=jnp.uint32), jnp.float32)
    elif dtype == jnp.bfloat16:
        return lax.bitcast_convert_type(jnp.array(int_val, dtype=jnp.uint16), jnp.bfloat16)
    elif dtype == jnp.int32:
        return jnp.array(int_val, dtype=jnp.int32)  # 直接当整数
    elif dtype == jnp.uint32:
        return jnp.array(int_val, dtype=jnp.uint32)
    elif dtype == jnp.int16:
        return jnp.array(int_val, dtype=jnp.int16)
    elif dtype == jnp.uint16:
        return jnp.array(int_val, dtype=jnp.uint16)
    # ... 其他整数类型按需添加
    return jnp.array(int_val, dtype=dtype)


def jax_to_hex(jax_val):
    """通用 JAX Array -> Hex 转换"""
    # 如果是 bool (比较运算结果)，直接返回 1 或 0
    if jax_val.dtype == jnp.bool_:
        return "0x1" if jax_val else "0x0"

    # Bitcast 回无符号整数以获取 Hex
    if jax_val.dtype == jnp.float32:
        val_uint = lax.bitcast_convert_type(jax_val, jnp.uint32)
        width = 8
    elif jax_val.dtype == jnp.bfloat16:
        val_uint = lax.bitcast_convert_type(jax_val, jnp.uint16)
        width = 4
    elif jax_val.dtype in [jnp.int32, jnp.uint32]:
        val_uint = lax.bitcast_convert_type(jax_val, jnp.uint32)
        width = 8
    elif jax_val.dtype in [jnp.int16, jnp.uint16]:
        val_uint = lax.bitcast_convert_type(jax_val, jnp.uint16)
        width = 4
    elif jax_val.dtype == jnp.int8 or jax_val.dtype == jnp.uint8:
        val_uint = lax.bitcast_convert_type(jax_val, jnp.uint8)
        width = 2
    else:
        return str(jax_val)

    try:
        int_val = int(val_uint)
    except:
        int_val = int(val_uint.item())
    return f"0x{int_val:0{width}X}"


# ==========================================
# 2. 算子注册表 (Op Registry)
#    通过 Regex 匹配文件名 -> JAX 函数
# ==========================================
OP_REGISTRY = [
    # --- 算术 ---
    (r".*_add.*", lambda x, y: x + y),
    (r".*_sub.*", lambda x, y: x - y),
    (r".*_mul.*", lambda x, y: x * y),
    (r".*_div.*", lambda x, y: x / y),  # 注意：整数除法可能需要 jnp.floor_divide
    (r".*_min.*", jnp.minimum),
    (r".*_max.*", jnp.maximum),

    # --- 位运算 ---
    (r".*_and\..*", lambda x, y: jnp.bitwise_and(x, y)),  # 排除 andn
    (r".*_or.*", lambda x, y: jnp.bitwise_or(x, y)),
    (r".*_xor.*", lambda x, y: jnp.bitwise_xor(x, y)),

    # --- 比较 (结果通常为 0/1) ---
    (r".*_eq.*", lambda x, y: x == y),
    (r".*_ne.*", lambda x, y: x != y),
    (r".*_gt.*", lambda x, y: x > y),
    (r".*_lt.*", lambda x, y: x < y),
    (r".*_ge.*", lambda x, y: x >= y),
    (r".*_le.*", lambda x, y: x <= y),

    # --- 转换 (需细化，这里演示) ---
    (r".*f32_to_bf16.*", lambda x: lax.convert_element_type(x, jnp.bfloat16)),
    (r".*bf16_to_f32.*", lambda x: lax.convert_element_type(x, jnp.float32)),

    # --- 取整 ---
    (r".*_ceil.*", jnp.ceil),
    (r".*_floor.*", jnp.floor),
    (r".*_trunc.*", jnp.trunc),
]


def get_op_from_filename(filename):
    """根据文件名匹配算子"""
    fname = os.path.basename(filename).lower()
    for pattern, func in OP_REGISTRY:
        if re.match(pattern, fname):
            return func
    return None


def get_dtype_from_filename(filename):
    """猜测主要数据类型"""
    fname = os.path.basename(filename).lower()
    if 'f32' in fname: return jnp.float32
    if 'bf16' in fname: return jnp.bfloat16
    if 'u32' in fname: return jnp.uint32
    if 's32' in fname: return jnp.int32
    if 'u16' in fname: return jnp.uint16
    if 's16' in fname: return jnp.int16
    return jnp.float32  # 默认


# ==========================================
# 3. 智能 CSV 加载器
# ==========================================
def parse_test_file(filepath):
    """
    智能解析 CSV。
    1. 寻找包含 'type' 和 'val' 的区域作为 define_table
    2. 寻找包含 'test', 'x', 'exp_out' 的区域作为 test_table
    """
    try:
        # 读取整个文件为 list of lists，不做任何解析
        df_raw = pd.read_csv(filepath, header=None, engine='python')
    except Exception as e:
        logger.error(f"  [Error] Read failed: {e}")
        return None, None

    # 寻找 Header 行号
    def_start_row = -1
    test_start_row = -1

    for i, row in df_raw.iterrows():
        row_str = " ".join([str(x).lower() for x in row.values])
        if 'type' in row_str and 'val' in row_str:
            def_start_row = i
        if ('test' in row_str or 'x' in row_str) and 'exp_out' in row_str:
            # 优先找 test 这一行，通常下一行是 x, y
            # 如果这行同时包含 x 和 exp_out，那它就是 header
            test_start_row = i

    if def_start_row == -1 or test_start_row == -1:
        logger.warning(f"  [Skip] Could not identify headers in {os.path.basename(filepath)}")
        return None, None

    # 读取 Definitions
    # 假设 definition table 在 test table 之前
    # 截取两个 header 之间的部分
    df_def = df_raw.iloc[def_start_row + 1: test_start_row].copy()
    # 设置列名 (假设 header 行的结构)
    def_header = df_raw.iloc[def_start_row].values
    # 简单的列名清理
    df_def.columns = [str(x).strip() for x in def_header]

    # 读取 Tests
    df_test = df_raw.iloc[test_start_row + 1:].copy()
    test_header = df_raw.iloc[test_start_row].values
    df_test.columns = [str(x).strip() for x in test_header]

    return df_def, df_test


# ==========================================
# 4. 主执行逻辑
# ==========================================
def run_single_file(filepath, output_dir):
    filename = os.path.basename(filepath)
    logger.info(f"Processing {filename}...")

    # 1. 确定算子和类型
    op_func = get_op_from_filename(filename)
    if not op_func:
        logger.warning(f"  [Skip] No operator matched for {filename}")
        return

    dtype = get_dtype_from_filename(filename)

    # 2. 解析文件
    df_def, df_test = parse_test_file(filepath)
    if df_def is None: return

    # 3. 构建值注册表
    value_registry = {}
    # 找到 type 和 val 列
    type_col = next((c for c in df_def.columns if 'type' in c.lower()), None)
    val_col = next((c for c in df_def.columns if 'val' in c.lower()), None)

    if not type_col or not val_col:
        logger.warning("  [Skip] Cannot find 'type' or 'val' columns")
        return

    for _, row in df_def.iterrows():
        if pd.notna(row[type_col]) and pd.notna(row[val_col]):
            name = str(row[type_col]).strip()
            hex_val = str(row[val_col]).strip()
            value_registry[name] = hex_to_jax(hex_val, dtype)

    # 4. 运行测试
    # 编译算子
    jit_op = jax.jit(op_func)

    # 准备结果列
    test_outs = []
    results = []

    # 识别输入列 (x, y)
    x_col = next((c for c in df_test.columns if c.strip() == 'x'), None)
    y_col = next((c for c in df_test.columns if c.strip() == 'y'), None)
    exp_col = next((c for c in df_test.columns if 'exp_out' in c), None)

    if not x_col or not exp_col:
        logger.warning("  [Skip] Cannot find 'x' or 'exp_out' columns")
        return

    is_binary = (y_col is not None)

    for idx, row in df_test.iterrows():
        x_name = row[x_col]
        if pd.isna(x_name):
            test_outs.append("")
            results.append("")
            continue

        x_name = str(x_name).strip()
        if x_name not in value_registry:
            test_outs.append("ERR_UNK_X")
            results.append("Error")
            continue

        val_x = value_registry[x_name]

        try:
            if is_binary:
                y_name = row[y_col]
                y_name = str(y_name).strip()
                if y_name not in value_registry:
                    test_outs.append("ERR_UNK_Y")
                    results.append("Error")
                    continue
                val_y = value_registry[y_name]
                res = jit_op(val_x, val_y)
            else:
                # 一元运算
                res = jit_op(val_x)

            res.block_until_ready()
            hex_res = jax_to_hex(res)
            test_outs.append(hex_res)

            # 对比
            exp_hex = str(row[exp_col]).strip()
            # 简单的归一化对比
            norm_act = hex_res.lower().replace('0x', '')
            norm_exp = exp_hex.lower().replace('0x', '')

            if norm_act == norm_exp:
                results.append("PASS")
            else:
                results.append(f"FAIL (Exp: {exp_hex})")

        except Exception as e:
            test_outs.append(f"ERR_EXEC: {str(e)}")
            results.append("Error")

    # 5. 保存结果
    df_test['test_out'] = test_outs
    df_test['result'] = results

    out_path = os.path.join(output_dir, f"RES_{filename}")
    df_test.to_csv(out_path, index=False)

    # 统计
    pass_cnt = results.count("PASS")
    total_cnt = len([r for r in results if r])
    logger.info(f"  -> Saved to {out_path} (Pass: {pass_cnt}/{total_cnt})")


def main():
    input_dir = "./input_csvs"  # 你的148个csv放这里
    output_dir = "./test_results"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = glob.glob(os.path.join(input_dir, "*.csv"))
    print(f"Found {len(files)} files.")

    for f in files:
        run_single_file(f, output_dir)


if __name__ == "__main__":
    main()