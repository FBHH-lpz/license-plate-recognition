# make_bold_templates.py
import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont


def generate_bold_hanzi():
    # 1. 准备目录
    save_dir = "templates_bold"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 2. 准备字符和对应的文件名
    # 包含了常见的广东车牌汉字
    chars = [
        ('广', 'guang'), ('州', 'zhou'),
        ('东', 'dong'), ('莞', 'guan'),
        ('佛', 'fo'), ('山', 'shan')
    ]

    # 3. 设置字体 (Windows默认黑体路径，Mac/Linux请自行替换)
    # 黑体 (SimHei) 是最接近车牌的字体
    font_path = "C:/Windows/Fonts/simhei.ttf"

    # 如果找不到字体，尝试使用系统默认的 arial unicode 或者提示用户
    if not os.path.exists(font_path):
        print(f"⚠️ 找不到字体文件: {font_path}")
        print("请修改代码中的 font_path 为你电脑上的支持中文的 .ttf 字体路径")
        return

    # 设置字号 (稍大一点，方便后续缩小)
    font_size = 100
    font = ImageFont.truetype(font_path, font_size)

    print(f"开始生成加粗模板 -> {save_dir}/ ...")

    for char, filename in chars:
        # --- A. 使用 PIL 绘制基础字形 ---
        # 创建大一点的画布，防止字出界
        pil_img = Image.new("L", (120, 120), 0)  # 0=黑色背景
        draw = ImageDraw.Draw(pil_img)

        # 居中绘制 (白色字)
        bbox = draw.textbbox((0, 0), char, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (120 - text_w) // 2
        y = (120 - text_h) // 2 - 10  # 稍微往上提一点，因为汉字通常偏下

        draw.text((x, y), char, font=font, fill=255)

        # 转为 OpenCV 格式
        cv_img = np.array(pil_img)

        # --- B. 核心：形态学膨胀 (人工加粗) ---
        # 如果觉得还不够粗，可以把 kernel 改成 (5, 5) 或 (7, 7)
        kernel = np.ones((5, 5), np.uint8)
        cv_img = cv2.dilate(cv_img, kernel, iterations=1)

        # --- C. 裁剪与统一尺寸 ---
        coords = cv2.findNonZero(cv_img)
        x, y, w, h = cv2.boundingRect(coords)

        # 裁剪出最小外接矩形
        crop = cv_img[y:y + h, x:x + w]

        # 强制缩放到标准模板大小 (30x60)
        final_img = cv2.resize(crop, (30, 60))

        # 二值化确保纯黑纯白
        _, final_img = cv2.threshold(final_img, 127, 255, cv2.THRESH_BINARY)

        # --- D. 保存 ---
        save_path = os.path.join(save_dir, f"{filename}.jpg")
        cv2.imwrite(save_path, final_img)
        print(f"已生成: {char} -> {save_path}")

    print("✅ 模板生成完毕！")


if __name__ == "__main__":
    generate_bold_hanzi()