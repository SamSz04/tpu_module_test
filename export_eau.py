import os
import pandas as pd
import re


def sanitize_filename(name):
    """清洗文件名，防止非法字符"""
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    return name.strip()


def export_sheets(excel_path, output_folder="input_eau_251223"):
    if not os.path.exists(excel_path):
        print(f"Error: File {excel_path} not found.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")

    print(f"Reading {excel_path}...")
    try:
        # 读取所有 Sheet，header=0 表示第一行是列名 (Xin, CmodelOut, TPUOut)
        xls = pd.read_excel(excel_path, sheet_name=None, header=0)
    except Exception as e:
        print(f"Read failed: {e}")
        return

    print(f"Found {len(xls)} sheets.")

    for sheet_name, df in xls.items():
        safe_name = sanitize_filename(sheet_name)
        csv_name = f"{safe_name}.csv"
        out_path = os.path.join(output_folder, csv_name)

        # 保存 CSV，保留表头
        df.to_csv(out_path, index=False)
        print(f"  -> Exported: {csv_name} ({len(df)} rows)")

    print("\nDone! All files in:", output_folder)


if __name__ == "__main__":
    # 你的文件名
    EXCEL_FILE = "EAU_test_data2.xlsx"
    export_sheets(EXCEL_FILE)