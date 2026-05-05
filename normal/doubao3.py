import cv2
import numpy as np
import os


# ==========================================
# 第一部分：字符模板（完全保留）
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
# 第二部分：核心处理类（修复选错问题，保留上一版优势）
# ==========================================
class LicensePlateRecognizer:
    def __init__(self):
        self.templates = generate_templates()
        self.debug = True

    def preprocess_enhanced(self, img):
        """完全保留上一版的增强预处理"""
        height, width = img.shape[:2]
        scale = 800 / width
        img_resized = cv2.resize(img, (800, int(height * scale)))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        sobel_x = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
        laplacian = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
        edges = cv2.bitwise_or(sobel_x, laplacian)
        ret, binary_edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)

        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 50, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        kernel_white = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_white_clean = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel_white)

        morph_combined = cv2.bitwise_and(binary_edges, mask_white_clean)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        morph_final = cv2.morphologyEx(morph_combined, cv2.MORPH_CLOSE, kernel_close)

        if self.debug:
            cv2.imshow("1. Enhanced Edges", edges)
            cv2.imshow("2. White Mask", mask_white_clean)
            cv2.imshow("3. Combined Morph", morph_final)

        return img_resized, morph_final, mask_white_clean  # 新增返回干净的白色掩码

    def white_pixel_ratio(self, roi, mask_white, x, y, w, h):
        """新增：计算候选框内白色像素占比（车牌的核心特征）"""
        # 提取候选框对应的白色掩码区域
        mask_roi = mask_white[y:y + h, x:x + w]
        if mask_roi.size == 0:
            return 0.0
        # 计算白色像素占比
        white_pixels = np.sum(mask_roi == 255)
        total_pixels = mask_roi.shape[0] * mask_roi.shape[1]
        ratio = white_pixels / total_pixels
        return ratio

    def check_char_texture(self, roi):
        """优化：扩大ROI后计算纹理，避免漏判"""
        if roi is None or roi.size == 0:
            return 0.0
        # 先轻微放大ROI（避免候选框太小漏算字符）
        h, w = roi.shape[:2]
        roi_enlarged = cv2.resize(roi, (int(w * 1.2), int(h * 1.2)))

        gray = cv2.cvtColor(roi_enlarged, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        # 优化：同时计算水平+垂直纹理（更精准）
        h_bin, w_bin = binary.shape
        texture_score = 0
        # 水平纹理
        for row in range(h_bin):
            texture_score += np.sum(np.abs(np.diff(binary[row, :]))) / 255
        # 垂直纹理
        for col in range(w_bin):
            texture_score += np.sum(np.abs(np.diff(binary[:, col]))) / 255

        texture_score = texture_score / (h_bin * w_bin * 2)  # 归一化
        return texture_score

    def fit_plate_contour(self, cnt, img_shape):
        """保留上一版的轮廓拟合"""
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        x, y, w, h = cv2.boundingRect(box)
        if w < h:
            w, h = h, w
            x, y = box[1][0], box[1][1]
        return x, y, w, h, box

    def locate_plate_dual_strategy(self, img_original, img_morph, mask_white):
        """保留双策略核心，新增白色占比校验"""
        contours, _ = cv2.findContours(img_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_contours = img_original.copy()
        cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
        if self.debug:
            cv2.imshow("4. All Contours", img_contours)

        # 第一步：筛选候选框（保留上一版逻辑）
        raw_candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300:
                continue
            x, y, w, h, box = self.fit_plate_contour(cnt, img_original.shape)
            ratio = w / float(h)
            if 1.5 < ratio < 6.0:
                raw_candidates.append((x, y, w, h, area, box))

        # 第二步：双重校验（纹理+白色占比，核心修复！）
        valid_candidates = []
        for (x, y, w, h, area, box) in raw_candidates:
            roi = img_original[y:y + h, x:x + w]
            # 校验1：字符纹理（优化后）
            texture_score = self.check_char_texture(roi)
            # 校验2：白色像素占比（新增，车牌占比>0.5）
            white_ratio = self.white_pixel_ratio(roi, mask_white, x, y, w, h)

            # 只有同时满足：纹理>0.04 + 白色占比>0.5，才保留
            if texture_score > 0.04 and white_ratio > 0.5:
                valid_candidates.append((x, y, w, h, area, box, texture_score, white_ratio))
                # 调试标注：纹理分数+白色占比
                if self.debug:
                    cv2.putText(img_contours, f"T:{texture_score:.2f} W:{white_ratio:.2f}",
                                (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # 第三步：颜色区域补充（仅当有效候选框为空时）
        if not valid_candidates:
            print("⚠️  无有效候选框，启用颜色区域补充...")
            gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
            ret, binary_white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            contours_white, _ = cv2.findContours(binary_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours_white:
                cnt_white = max(contours_white, key=cv2.contourArea)
                area = cv2.contourArea(cnt_white)
                if area > 1000:
                    x, y, w, h, box = self.fit_plate_contour(cnt_white, img_original.shape)
                    roi = img_original[y:y + h, x:x + w]
                    texture_score = self.check_char_texture(roi)
                    white_ratio = self.white_pixel_ratio(roi, mask_white, x, y, w, h)
                    valid_candidates.append((x, y, w, h, area, box, texture_score, white_ratio))

        # 第四步：优化排序（纹理0.6 + 白色占比0.3 + 面积0.1）
        if not valid_candidates:
            print("❌ 双策略定位均失败")
            return None

        # 权重调整：优先纹理和白色占比（干扰框这两个值都低）
        valid_candidates.sort(key=lambda x: (x[6] * 0.6 + x[7] * 0.3 + x[4] / 10000 * 0.1), reverse=True)
        best = valid_candidates[0]
        x, y, w, h, _, box, _, _ = best

        # 动态padding（保留）
        pad = int(h * 0.2)
        x_start = max(0, x - pad)
        y_start = max(0, y - pad)
        x_end = min(img_original.shape[1], x + w + pad)
        y_end = min(img_original.shape[0], y + h + pad)
        plate_img = img_original[y_start:y_end, x_start:x_end]

        # 调试可视化（标注关键分数）
        img_plate_box = img_original.copy()
        cv2.rectangle(img_plate_box, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
        cv2.drawContours(img_plate_box, [box], 0, (255, 0, 0), 2)
        cv2.putText(img_plate_box, f"T:{best[6]:.2f} W:{best[7]:.2f}", (x_start, y_start - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if self.debug:
            cv2.imshow("5. Final Plate Box (Fixed)", img_plate_box)
            cv2.imshow("6. Extracted Plate", plate_img)

        return plate_img

    def segment_chars(self, plate_img):
        """完全保留上一版的字符分割"""
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
        """完全保留上一版的字符识别"""
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
# 主程序（仅修改预处理返回值）
# ==========================================
if __name__ == "__main__":
    # 替换为你的图片绝对路径
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

        # 仅修改：预处理返回值增加mask_white
        resized, morph, mask_white = recognizer.preprocess_enhanced(original_img)
        plate_img = recognizer.locate_plate_dual_strategy(resized, morph, mask_white)

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