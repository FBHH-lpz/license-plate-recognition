import cv2
import numpy as np
import os


# ==========================================
# 第一部分：生成字符模板 (保留增强版)
# ==========================================
def generate_templates():
    templates = {}
    characters = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"

    print(">>> 正在生成增强版字符模板库...")

    for char in characters:
        img = np.zeros((60, 30), dtype=np.uint8)
        cv2.putText(img, char, (3, 45), cv2.FONT_HERSHEY_PLAIN, 2, 255, 4)

        coords = cv2.findNonZero(img)
        if coords is None:
            continue
        x, y, w, h = cv2.boundingRect(coords)
        img_crop = img[y:y + h, x:x + w]
        img_final = cv2.resize(img_crop, (30, 60))
        templates[char] = img_final

    return templates


# ==========================================
# 第二部分：核心处理类 (简化定位+增强调试)
# ==========================================
class LicensePlateRecognizer:
    def __init__(self):
        self.templates = generate_templates()
        self.debug = True  # 强制开启调试，显示所有中间步骤

    def preprocess(self, img):
        """简化预处理：保留核心步骤，降低过度处理"""
        # 固定缩放（统一缩放到800宽，保证一致性）
        height, width = img.shape[:2]
        scale = 800 / width
        img_resized = cv2.resize(img, (800, int(height * scale)))

        # 灰度+高斯模糊+边缘检测（简化参数）
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        sobel = cv2.Sobel(blurred, cv2.CV_8U, 1, 0, ksize=3)

        # 改用固定阈值二值化（更稳定，避免OTSU适配问题）
        ret, binary = cv2.threshold(sobel, 80, 255, cv2.THRESH_BINARY)

        # 简化形态学闭运算（固定核尺寸，保证车牌轮廓保留）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 调试：显示预处理后的边缘图
        if self.debug:
            cv2.imshow("1. Preprocessed Edge", closed)

        return img_resized, closed

    def locate_plate(self, img_original, img_morph):
        """大幅简化车牌定位：降低筛选门槛，优先检测"""
        contours, _ = cv2.findContours(img_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 调试：显示所有检测到的轮廓
        img_contours = img_original.copy()
        cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
        if self.debug:
            cv2.imshow("2. All Contours", img_contours)

        candidates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = w / float(h)
            area = w * h

            # 极度宽松的筛选条件：只要是窄长型轮廓就保留
            if (area > 500) and (1.0 < ratio < 6.0):  # 宽高比1-6，面积>500
                candidates.append((x, y, w, h, area))

        if not candidates:
            print("❌ 未检测到任何窄长型轮廓（车牌候选）")
            return None

        # 按面积排序，取前5个候选框（避免只取第一个漏掉车牌）
        candidates.sort(key=lambda x: x[4], reverse=True)
        # 调试：显示前5个候选框
        img_candidates = img_original.copy()
        for i, (x, y, w, h, _) in enumerate(candidates[:5]):
            cv2.rectangle(img_candidates, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img_candidates, str(i + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if self.debug:
            cv2.imshow("3. Top 5 Candidates", img_candidates)

        # 优先选第一个候选框（面积最大）
        best = candidates[0]
        x, y, w, h, _ = best

        # 加大padding，确保车牌完整
        pad = 15
        plate_img = img_original[max(0, y - pad):y + h + pad,
        max(0, x - pad):x + w + pad]

        if self.debug:
            cv2.imshow("4. Selected Plate ROI", plate_img)

        return plate_img

    def segment_chars(self, plate_img):
        """简化字符分割：保留核心逻辑"""
        if plate_img is None or plate_img.size == 0:
            return []

        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

        # 裁剪边框（固定比例）
        h, w = binary.shape
        binary = binary[int(h * 0.1):int(h * 0.9), int(w * 0.05):int(w * 0.95)]

        # 轻微膨胀
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilated = cv2.dilate(binary, kernel, iterations=1)

        if self.debug:
            cv2.imshow("5. Plate Binary", dilated)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        char_imgs = []
        # 宽松的字符筛选
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h > binary.shape[0] * 0.2 and w > 5:  # 高度>20%，宽度>5
                char_imgs.append((x, binary[y:y + h, x:x + w]))

        char_imgs.sort(key=lambda x: x[0])
        # 调试：显示分割的字符
        if self.debug and char_imgs:
            char_show = np.hstack([cv2.resize(c[1], (30, 60)) for c in char_imgs])
            cv2.imshow("6. Segmented Chars", char_show)

        return [c[1] for c in char_imgs]

    def recognize(self, char_imgs):
        """保留字符识别逻辑"""
        result_string = ""
        correction_map = {
            'S': '3', 'Z': '2', 'D': '0', 'O': '0', 'I': '1',
            'B': '8', 'N': '8', 'Q': '0', 'L': '1', 'G': '6',
            'R': '8', 'T': '7', 'P': '9'
        }

        print("\n--- 字符识别详情 ---")
        for i, char_img in enumerate(char_imgs):
            char_resized = cv2.resize(char_img, (30, 60))

            best_score = -1
            best_char = "?"

            for char_key, template_img in self.templates.items():
                res = cv2.matchTemplate(char_resized, template_img, cv2.TM_CCOEFF_NORMED)
                _, score, _, _ = cv2.minMaxLoc(res)

                if score > best_score:
                    best_score = score
                    best_char = char_key

            if best_score < 0.3:  # 降低置信度阈值
                print(f"位置{i}: 识别为 [{best_char}] (置信度 {best_score:.2f}) -> 跳过")
                continue

            original_char = best_char
            if best_char in correction_map:
                best_char = correction_map[best_char]

            print(f"位置{i}: 识别为 [{original_char}] -> 修正为 [{best_char}] (置信度: {best_score:.2f})")
            result_string += best_char

        return result_string


# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # 请确认图片路径正确！
    image_paths = [
        "images/normal/1.PNG",  # 第一张图：P53283
        "images/normal/2.jpg",  # 第二张图：216556
        # "images/normal/3.jpg",  # 第三张图：P53283
        # "images/normal/4.jpg",  # 第四张图：P53283
        # "images/normal/5.jpg",  # 第五张图：P53283
    ]

    # 检查路径是否存在
    for path in image_paths:
        if not os.path.exists(path):
            print(f"\n❌ 错误：文件不存在 -> {path}")
            print("   请检查图片路径是否正确，建议使用绝对路径测试，例如：F:/Project/.../1.PNG")

    recognizer = LicensePlateRecognizer()

    for img_path in image_paths:
        if not os.path.exists(img_path):
            continue

        print(f"\n{'=' * 40}\n处理图片：{img_path}\n{'=' * 40}")
        original_img = cv2.imread(img_path)
        if original_img is None:
            print(f"❌ 无法读取图片：{img_path}（可能格式错误）")
            continue

        # 核心处理流程
        resized, morph = recognizer.preprocess(original_img)
        plate_img = recognizer.locate_plate(resized, morph)

        if plate_img is not None:
            char_list = recognizer.segment_chars(plate_img)
            print(f"✅ 成功定位到车牌，分割出 {len(char_list)} 个字符区域")

            if len(char_list) > 0:
                plate_number = recognizer.recognize(char_list)
                print("\n" + "=" * 30)
                print(f"最终识别结果: {plate_number}")
                print("=" * 30)
            else:
                print("❌ 未分割出有效字符")
        else:
            print("❌ 未能定位到车牌")

    # 等待按键关闭所有窗口（重要！）
    print("\n\n按任意键关闭所有调试窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()