#!/usr/bin/env python3
"""
批量将DAT文件转换为CSV格式
"""

import os
import sys
from pathlib import Path
from dat_to_csv import dat_to_csv


def batch_convert(directory='.', pattern='*.dat'):
    """
    批量转换目录中的所有DAT文件
    
    Args:
        directory: 目录路径
        pattern: 文件匹配模式
    """
    directory = Path(directory)
    dat_files = list(directory.glob(pattern))
    
    if not dat_files:
        print(f"在 {directory} 中没有找到匹配 {pattern} 的文件")
        return
    
    print(f"找到 {len(dat_files)} 个DAT文件")
    print("-" * 60)
    
    success_count = 0
    fail_count = 0
    
    for dat_file in dat_files:
        try:
            print(f"\n处理: {dat_file.name}")
            dat_to_csv(str(dat_file))
            success_count += 1
        except Exception as e:
            print(f"  错误: {e}")
            fail_count += 1
    
    print("\n" + "=" * 60)
    print(f"转换完成: 成功 {success_count} 个, 失败 {fail_count} 个")


def main():
    """主函数"""
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = '.'
    
    if not os.path.exists(directory):
        print(f"错误: 目录不存在: {directory}")
        sys.exit(1)
    
    batch_convert(directory)


if __name__ == "__main__":
    main()
