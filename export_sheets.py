import os
import pandas as pd


def extract_all_sheets_to_csv(excel_path, output_folder="input_csvs_251223"):
    """
    将 Excel 中的所有 Sheet 导出为单独的 CSV 文件。
    保持原始结构（不添加 Header，不添加 Index）。
    """

    # 1. 检查文件是否存在
    if not os.path.exists(excel_path):
        print(f"错误: 找不到文件 {excel_path}")
        return

    # 2. 创建输出目录
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建输出目录: {output_folder}")

    print(f"正在读取 Excel 文件: {excel_path} ...")

    try:
        # 使用 ExcelFile 对象以提高性能（不用一次性把所有数据读入内存）
        xls = pd.ExcelFile(excel_path)
    except Exception as e:
        print(f"读取 Excel 失败: {e}")
        return

    sheet_names = xls.sheet_names
    total_sheets = len(sheet_names)
    print(f"共发现 {total_sheets} 个 Sheet，准备导出...")

    # 3. 循环导出
    for i, sheet_name in enumerate(sheet_names):
        try:
            # header=None: 确保不把第一行当作列名，保留 Excel 原始布局
            # index_col=None: 不使用某一列作为索引
            df = pd.read_excel(xls, sheet_name=sheet_name, header=None)

            # 清理文件名 (去除首尾空格)
            safe_name = sheet_name.strip()
            csv_filename = f"{safe_name}.csv"
            output_path = os.path.join(output_folder, csv_filename)

            # header=False, index=False: 导出纯数据，不加 Pandas 生成的序号
            df.to_csv(output_path, index=False, header=False)

            print(f"[{i + 1}/{total_sheets}] 已导出: {csv_filename}")

        except Exception as e:
            print(f"[{i + 1}/{total_sheets}] 导出 {sheet_name} 失败: {e}")

    print("\n" + "=" * 30)
    print(f"导出完成！所有 CSV 文件保存在 '{output_folder}' 目录下。")
    print("=" * 30)


if __name__ == "__main__":
    # 修改这里为你的文件名
    # EXCEL_FILE = "DIP_AIC_LIB_Test_Point_251205.xlsx"
    EXCEL_FILE = "AIC_LIB_test_point_v2.xlsx"

    extract_all_sheets_to_csv(EXCEL_FILE)