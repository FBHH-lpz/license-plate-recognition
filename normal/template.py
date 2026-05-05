# create_chinese_only.py
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 1. 准备文件夹
TEMPLATE_DIR = "templates_cn"
if not os.path.exists(TEMPLATE_DIR):
    os.makedirs(TEMPLATE_DIR)


def create_chinese_char(char, font_path="simhei.ttf"):
    # 创建黑色背景 (与原始代码的 np.zeros 保持一致)
    img_pil = Image.new("L", (100, 100), 0)
    draw = ImageDraw.Draw(img_pil)

    try:
        # 如果没有 simhei.ttf，请改为你的系统字体路径
        # Windows 都在 C:/Windows/Fonts/simhei.ttf
        font = ImageFont.truetype("simhei.ttf", 80)
    except:
        print("⚠️ 找不到 simhei.ttf，汉字将无法生成！请检查字体路径。")
        return None

    # 居中绘制白色文字 (fill=255) -> 这样出来就是黑底白字
    bbox = draw.textbbox((0, 0), char, font=font)
    w_text, h_text = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((100 - w_text) / 2, (100 - h_text) / 2 - 10), char, font=font, fill=255)

    # 转为 OpenCV 格式
    img_cv = np.array(img_pil)

    # 裁剪
    coords = cv2.findNonZero(img_cv)
    if coords is None: return None
    x, y, w, h = cv2.boundingRect(coords)

    # 缩放到 30x60 (与原始代码一致)
    img_crop = img_cv[y:y + h, x:x + w]
    img_final = cv2.resize(img_crop, (30, 60))

    return img_final


# 你需要的汉字列表
chinese_list = {
    '广': 'guang',
    '州': 'zhou',
    '东': 'dong',
    '莞': 'guan',
    '佛': 'fo',
    '山': 'shan'
}

print(">>> 开始生成汉字模板...")
for char, filename in chinese_list.items():
    img = create_chinese_char(char)
    if img is not None:
        save_path = os.path.join(TEMPLATE_DIR, f"{filename}.jpg")
        cv2.imwrite(save_path, img)
        print(f"  已生成: {char} -> {save_path}")

print("✅ 汉字生成完毕！")