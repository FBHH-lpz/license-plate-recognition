import cv2
import numpy as np
import os


# ==========================================
# 第一部分：字符模板（保留）
# ==========================================
def generate_templates():
    templates = {}
    characters = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"

    print(">>> 正在生成字符模板库...")
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
# 第二部分：核心处理类（定位全优化）
# ==========================================
class LicensePlateRecognizer:
    def __init__(self):
        self.templates = generate_templates()
        self.debug = True

    def preprocess_enhanced(self, img):
        """增强预处理：边缘+颜色双特征提取"""
        # 1. 固定缩放（统一尺寸）
        height, width = img.shape[:2]
        scale = 800 / width
        img_resized = cv2.resize(img, (800, int(height * scale)))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        # 2. 多尺度边缘检测（解决单一边缘提取不足）
        sobel_x = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
        laplacian = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
        edges = cv2.bitwise_or(sobel_x, laplacian)  # 融合边缘
        ret, binary_edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)

        # 3. 颜色掩码（针对广州白底黑字车牌：提取白色区域）
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        # 白色HSV范围：低饱和度、高亮度
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 50, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        # 形态学开运算去除白色噪点
        kernel_white = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_white_clean = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel_white)

        # 4. 融合边缘+颜色特征（双保险）
        morph_combined = cv2.bitwise_and(binary_edges, mask_white_clean)
        # 闭运算连接断裂轮廓
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        morph_final = cv2.morphologyEx(morph_combined, cv2.MORPH_CLOSE, kernel_close)

        # 调试窗口
        if self.debug:
            cv2.imshow("1. Enhanced Edges", edges)
            cv2.imshow("2. White Mask", mask_white_clean)
            cv2.imshow("3. Combined Morph", morph_final)

        return img_resized, morph_final

    def fit_plate_contour(self, cnt, img_shape):
        """精准拟合车牌轮廓：多边形近似+倾斜校正"""
        # 多边形近似（保留轮廓主要特征）
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # 转换为矩形（支持倾斜）
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        # 计算矩形的x/y/w/h（兼容倾斜）
        x, y, w, h = cv2.boundingRect(box)
        # 确保宽>高（处理旋转）
        if w < h:
            w, h = h, w
            x, y = box[1][0], box[1][1]

        return x, y, w, h, box

    def locate_plate_dual_strategy(self, img_original, img_morph):
        """双策略定位：轮廓定位 + 颜色区域补充"""
        contours, _ = cv2.findContours(img_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_contours = img_original.copy()
        cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
        if self.debug:
            cv2.imshow("4. All Contours", img_contours)

        # 策略1：轮廓筛选（宽松条件）
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300:  # 过滤极小噪点
                continue
            x, y, w, h, box = self.fit_plate_contour(cnt, img_original.shape)
            ratio = w / float(h)
            if 1.5 < ratio < 6.0:  # 宽高比1.5-6
                candidates.append((x, y, w, h, area, box))

        # 策略2：颜色区域补充（如果轮廓筛选无结果）
        if not candidates:
            print("⚠️  轮廓筛选无结果，启用颜色区域补充...")
            # 提取白色区域的最大连通域
            gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
            ret, binary_white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            contours_white, _ = cv2.findContours(binary_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours_white:
                cnt_white = max(contours_white, key=cv2.contourArea)
                area = cv2.contourArea(cnt_white)
                if area > 1000:
                    x, y, w, h, box = self.fit_plate_contour(cnt_white, img_original.shape)
                    candidates.append((x, y, w, h, area, box))

        if not candidates:
            print("❌ 双策略定位均失败")
            return None

        # 选面积最大的候选框
        candidates.sort(key=lambda x: x[4], reverse=True)
        best = candidates[0]
        x, y, w, h, _, box = best

        # 动态padding（按车牌高度的20%）
        pad = int(h * 0.2)
        # 确保padding不超出图片范围
        x_start = max(0, x - pad)
        y_start = max(0, y - pad)
        x_end = min(img_original.shape[1], x + w + pad)
        y_end = min(img_original.shape[0], y + h + pad)

        # 提取完整车牌区域
        plate_img = img_original[y_start:y_end, x_start:x_end]

        # 调试：显示最终框选的车牌（红色框+多边形）
        img_plate_box = img_original.copy()
        cv2.rectangle(img_plate_box, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
        cv2.drawContours(img_plate_box, [box], 0, (255, 0, 0), 2)
        if self.debug:
            cv2.imshow("5. Final Plate Box", img_plate_box)
            cv2.imshow("6. Extracted Plate", plate_img)

        return plate_img

    def segment_chars(self, plate_img):
        """简化字符分割（保留核心）"""
        if plate_img is None or plate_img.size == 0:
            return []
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
        h, w = binary.shape
        binary = binary[int(h * 0.1):int(h * 0.9), int(w * 0.05):int(w * 0.95)]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilated = cv2.dilate(binary, kernel, iterations=1)
        if self.debug:
            cv2.imshow("7. Plate Binary", dilated)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        char_imgs = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h > binary.shape[0] * 0.2 and w > 5:
                char_imgs.append((x, binary[y:y + h, x:x + w]))
        char_imgs.sort(key=lambda x: x[0])
        if self.debug and char_imgs:
            char_show = np.hstack([cv2.resize(c[1], (30, 60)) for c in char_imgs])
            cv2.imshow("8. Segmented Chars", char_show)
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
            if best_score < 0.3:
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
    # 替换为你的图片绝对路径！
    image_paths = [
        "images/normal/1.PNG",  # 第一张图：P53283
        "images/normal/2.jpg",  # 第二张图：216556
        "images/normal/3.jpg",  # 第三张图：P53283
        # "images/normal/4.jpg",  # 第四张图：P53283
        # "images/normal/5.jpg",  # 第五张图：P53283
    ]

    # 路径校验
    for path in image_paths:
        if not os.path.exists(path):
            print(f"\n❌ 错误：文件不存在 -> {path}")

    recognizer = LicensePlateRecognizer()

    for img_path in image_paths:
        if not os.path.exists(img_path):
            continue
        print(f"\n{'=' * 40}\n处理图片：{img_path}\n{'=' * 40}")
        original_img = cv2.imread(img_path)
        if original_img is None:
            print(f"❌ 无法读取图片：{img_path}")
            continue

        # 核心流程：增强预处理 + 双策略定位
        resized, morph = recognizer.preprocess_enhanced(original_img)
        plate_img = recognizer.locate_plate_dual_strategy(resized, morph)

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

    print("\n\n按任意键关闭所有调试窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()