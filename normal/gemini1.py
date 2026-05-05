import cv2
import numpy as np
import os


# ==========================================
# 第一部分：字符模板 (更新：加入广州汉字)
# ==========================================
def generate_templates():
    templates = {}
    # 【改动】加入 '广', '洲' 以便识别
    characters = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ广洲"

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
# 第二部分：核心处理类 (定位用旧版，分割用新版)
# ==========================================
class LicensePlateRecognizer:
    def __init__(self):
        self.templates = generate_templates()
        self.debug = True

    # ---------------------------------------------------------
    # 🟢 模块 A：辅助函数 (来自第一版代码，用于增强定位)
    # ---------------------------------------------------------
    def white_pixel_ratio(self, roi, mask_white, x, y, w, h):
        """计算候选框内白色像素占比"""
        mask_roi = mask_white[y:y + h, x:x + w]
        if mask_roi.size == 0: return 0.0
        white_pixels = np.sum(mask_roi == 255)
        total_pixels = mask_roi.shape[0] * mask_roi.shape[1]
        return white_pixels / total_pixels

    def check_char_texture(self, roi):
        """计算纹理密度"""
        if roi is None or roi.size == 0: return 0.0
        h, w = roi.shape[:2]
        roi_enlarged = cv2.resize(roi, (int(w * 1.2), int(h * 1.2)))
        gray = cv2.cvtColor(roi_enlarged, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        h_bin, w_bin = binary.shape
        texture_score = 0
        # 水平 + 垂直 差分
        for row in range(h_bin):
            texture_score += np.sum(np.abs(np.diff(binary[row, :]))) / 255
        for col in range(w_bin):
            texture_score += np.sum(np.abs(np.diff(binary[:, col]))) / 255

        return texture_score / (h_bin * w_bin * 2)

    def fit_plate_contour(self, cnt, img_shape):
        """轮廓拟合"""
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        x, y, w, h = cv2.boundingRect(box)
        if w < h: w, h = h, w
        return x, y, w, h, box

    # ---------------------------------------------------------
    # 🟢 模块 B：预处理 & 定位 (完全复刻第一版代码，最强定位)
    # ---------------------------------------------------------
    def preprocess_enhanced(self, img):
        height, width = img.shape[:2]
        scale = 800 / width
        img_resized = cv2.resize(img, (800, int(height * scale)))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        sobel_x = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
        laplacian = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
        edges = cv2.bitwise_or(sobel_x, laplacian)
        _, binary_edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)

        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        mask_white = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 50, 255]))
        kernel_white = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_white_clean = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel_white)

        morph_combined = cv2.bitwise_and(binary_edges, mask_white_clean)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        morph_final = cv2.morphologyEx(morph_combined, cv2.MORPH_CLOSE, kernel_close)

        if self.debug:
            cv2.imshow("1. Enhanced Edges", edges)
            cv2.imshow("2. White Mask", mask_white_clean)
            cv2.imshow("3. Combined Morph", morph_final)

        return img_resized, morph_final, mask_white_clean

    def locate_plate_dual_strategy(self, img_original, img_morph, mask_white):
        """(来自第一版) 包含双重校验和颜色补救的定位逻辑"""
        contours, _ = cv2.findContours(img_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 1. 筛选
        raw_candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300: continue
            x, y, w, h, box = self.fit_plate_contour(cnt, img_original.shape)
            if 1.5 < w / float(h) < 6.0:
                raw_candidates.append((x, y, w, h, area, box))

        # 2. 校验 (纹理+颜色)
        valid_candidates = []
        for (x, y, w, h, area, box) in raw_candidates:
            roi = img_original[y:y + h, x:x + w]
            texture = self.check_char_texture(roi)
            white_r = self.white_pixel_ratio(roi, mask_white, x, y, w, h)

            # 严格门槛
            if texture > 0.04 and white_r > 0.5:
                valid_candidates.append((x, y, w, h, area, box, texture, white_r))

        # 3. 补救 (如果没有有效框，尝试直接用白色掩码找)
        if not valid_candidates:
            gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
            _, binary_white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            cnts_white, _ = cv2.findContours(binary_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts_white:
                cnt_max = max(cnts_white, key=cv2.contourArea)
                if cv2.contourArea(cnt_max) > 1000:
                    x, y, w, h, box = self.fit_plate_contour(cnt_max, img_original.shape)
                    roi = img_original[y:y + h, x:x + w]
                    valid_candidates.append((x, y, w, h, cv2.contourArea(cnt_max), box,
                                             self.check_char_texture(roi),
                                             self.white_pixel_ratio(roi, mask_white, x, y, w, h)))

        if not valid_candidates: return None

        # 4. 排序 (纹理权重最高)
        valid_candidates.sort(key=lambda x: (x[6] * 0.6 + x[7] * 0.3 + x[4] / 10000 * 0.1), reverse=True)
        best = valid_candidates[0]
        x, y, w, h, _, box, _, _ = best

        # 5. 提取
        pad = int(h * 0.2)
        plate_img = img_original[max(0, y - pad):min(img_original.shape[0], y + h + pad),
        max(0, x - pad):min(img_original.shape[1], x + w + pad)]

        if self.debug:
            temp = img_original.copy()
            cv2.drawContours(temp, [box], 0, (0, 255, 0), 2)
            cv2.imshow("5. Plate Location", temp)
            cv2.imshow("6. Plate Extracted", plate_img)

        return plate_img

    # ---------------------------------------------------------
    # 🟢 模块 C：字符分割 (完全采用 Step 28 最新逻辑)
    # ---------------------------------------------------------
    def segment_chars(self, plate_img):
        """Step 28: 42%分割 + C位优先(上) + 完美逻辑(下)"""
        if plate_img is None or plate_img.size == 0: return []

        # 1. 预处理 (去边框)
        h, w = plate_img.shape[:2]
        plate_img = plate_img[3:h - 3, 3:w - 3]
        h, w = plate_img.shape[:2]
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        split_y = int(h * 0.42)  # 42% 黄金分割
        final_chars = []

        # === 上层：Step 9 C位优先逻辑 ===
        top_gray = gray[0:split_y, :]
        top_bin = cv2.adaptiveThreshold(top_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 13, 10)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        top_bin = cv2.morphologyEx(top_bin, cv2.MORPH_OPEN, kernel)  # 去噪
        top_bin = cv2.morphologyEx(top_bin, cv2.MORPH_CLOSE, kernel)  # 连接

        cnts_top, _ = cv2.findContours(top_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        top_candidates = []
        for cnt in cnts_top:
            x, y, w_c, h_c = cv2.boundingRect(cnt)
            cx = x + w_c / 2

            if cx > w * 0.85 or cx < w * 0.05: continue  # 左右禁区
            if w_c / h_c > 1.3 or w_c / h_c < 0.1: continue  # 比例过滤
            if h_c < split_y * 0.2: continue  # 尺寸过滤

            roi = top_bin[y:y + h_c, x:x + w_c]
            if cv2.countNonZero(roi) / (w_c * h_c) > 0.95: continue  # 密度过滤

            top_candidates.append((x, y, w_c, h_c, top_bin[y:y + h_c, x:x + w_c]))

        # C位排序：只取离中间最近的2个
        if len(top_candidates) > 2:
            top_candidates.sort(key=lambda c: abs((c[0] + c[2] / 2) - w / 2))
            top_candidates = top_candidates[:2]
        top_candidates.sort(key=lambda c: c[0])  # 还原左右顺序

        for item in top_candidates: final_chars.append(item[4])

        # === 下层：Step 21 完美逻辑 ===
        bot_gray = gray[split_y:, :]
        bot_bin = cv2.adaptiveThreshold(bot_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 51, 15)
        cnts_bot, _ = cv2.findContours(bot_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bot_items = []
        for cnt in cnts_bot:
            x, y, w_c, h_c = cv2.boundingRect(cnt)
            if h_c < (h - split_y) * 0.35: continue
            if w_c / h_c > 2.0: continue
            bot_items.append((x, y, w_c, h_c, bot_bin[y:y + h_c, x:x + w_c]))

        bot_items.sort(key=lambda item: item[0])
        for item in bot_items: final_chars.append(item[4])

        # 统一输出
        resized_chars = []
        for img in final_chars:
            if img.shape[0] > 0 and img.shape[1] > 0:
                resized_chars.append(cv2.resize(img, (30, 60)))

                # =======================
                # 🛠️ 替换 Step 29 中的调试代码块
                # =======================
                if self.debug:
                    # 1. 转回彩色，准备画图
                    vis_debug = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

                    # 2. 画出黄色的 42% 分割线
                    cv2.line(vis_debug, (0, split_y), (w, split_y), (0, 255, 255), 2)

                    # 3. 画出上层选中的框 (红色) - 需要还原坐标
                    # 注意：我们在 candidates 里存的是局部坐标，不需要换算，因为就是在整图上画的
                    for item in top_candidates:  # 复用刚刚计算出的 top_candidates
                        x, y, w_c, h_c, _ = item
                        cv2.rectangle(vis_debug, (x, y), (x + w_c, y + h_c), (0, 0, 255), 2)
                        # 标个注 T
                        cv2.putText(vis_debug, "Top", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    # 4. 画出下层选中的框 (绿色)
                    for item in bot_items:  # 复用刚刚计算出的 bot_items
                        x, y, w_c, h_c, _ = item
                        # 下层的 y 已经是基于全图的了（因为bot_bin也是切片），哎不对
                        # 等等！注意！之前的逻辑里：
                        # bot_gray = gray[split_y:, :]
                        # 所以 findContours 出来的 y 是相对于 split_y 的！
                        # 我们画在 vis_debug (全图) 上时，y 需要加上 split_y

                        real_y = y + split_y
                        cv2.rectangle(vis_debug, (x, real_y), (x + w_c, real_y + h_c), (0, 255, 0), 2)

                    cv2.imshow("7. Segment Logic (Debug Box)", vis_debug)

                    if resized_chars:
                        show_img = np.hstack(resized_chars)
                        cv2.imshow("8. Final Chars", show_img)

        return resized_chars

    def recognize(self, char_imgs):
        """简单的模板匹配识别"""
        result = ""
        for char_img in char_imgs:
            best_score = -1
            best_char = "?"
            for char, template in self.templates.items():
                score = cv2.matchTemplate(char_img, template, cv2.TM_CCOEFF_NORMED)[0][0]
                if score > best_score:
                    best_score = score
                    best_char = char

            # 简单修正
            if best_char == 'D' and best_score < 0.6: best_char = '0'
            result += best_char
        return result


# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # ⚠️ 请确保这里是你的图片路径
    img_path = "images/normal/4.jpg"

    if os.path.exists(img_path):
        print(f"处理图片: {img_path}")
        recognizer = LicensePlateRecognizer()

        # 1. 预处理
        resized, morph, mask = recognizer.preprocess_enhanced(cv2.imread(img_path))

        # 2. 定位 (旧版逻辑：强力定位)
        plate = recognizer.locate_plate_dual_strategy(resized, morph, mask)

        if plate is not None:
            # 3. 分割 (新版逻辑：精准切割)
            chars = recognizer.segment_chars(plate)

            if chars:
                # 4. 识别
                text = recognizer.recognize(chars)
                print(f"\n✅ 最终结果: {text}")
            else:
                print("❌ 分割失败")
        else:
            print("❌ 定位失败")

        print("\n按任意键退出...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("❌ 找不到图片")