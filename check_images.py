#!/usr/bin/env python3
"""
检查 images/ 文件夹下的图片是否受损
支持格式：JPG, JPEG, PNG, GIF, BMP, WEBP, TIFF
"""

import os
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("正在安装 Pillow...")
    os.system(f"{sys.executable} -m pip install Pillow --break-system-packages -q")
    from PIL import Image


SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}


def check_image(filepath: Path) -> tuple[bool, str]:
    """
    检查单张图片是否受损。
    返回 (是否正常, 说明信息)
    """
    try:
        with Image.open(filepath) as img:
            img.verify()  # 验证文件完整性
        # verify() 后需重新打开才能加载像素数据
        with Image.open(filepath) as img:
            img.load()
        return True, "正常"
    except Exception as e:
        return False, str(e)


def check_images_folder(folder: str = "images") -> None:
    folder_path = Path(folder)

    if not folder_path.exists():
        print(f"❌ 文件夹不存在：{folder_path.resolve()}")
        sys.exit(1)

    image_files = [
        f for f in sorted(folder_path.rglob("*"))
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_files:
        print(f"⚠️  在 {folder_path} 中未找到支持的图片文件")
        return

    print(f"📂 扫描目录：{folder_path.resolve()}")
    print(f"📸 共找到 {len(image_files)} 张图片\n")
    print(f"{'文件路径':<50} {'状态':<8} 详情")
    print("-" * 90)

    ok_count = 0
    damaged_files = []

    for filepath in image_files:
        ok, msg = check_image(filepath)
        rel_path = str(filepath)
        if ok:
            ok_count += 1
            status = "✅ 正常"
            print(f"{rel_path:<50} {status}")
        else:
            damaged_files.append((rel_path, msg))
            status = "❌ 受损"
            print(f"{rel_path:<50} {status}  {msg}")

    print("-" * 90)
    print(f"\n📊 检查结果：{ok_count} 正常 / {len(damaged_files)} 受损 / {len(image_files)} 总计")

    if damaged_files:
        print("\n🔴 受损文件列表：")
        for path, reason in damaged_files:
            print(f"  • {path}")
            print(f"    原因：{reason}")
        sys.exit(2)  # 以非零退出码表示存在受损文件
    else:
        print("\n🎉 所有图片均完好无损！")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "images"
    check_images_folder(folder)