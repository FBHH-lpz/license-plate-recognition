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
        """Step 30: 完整修复版 (含螺丝钉过滤 + 强力胶水 + 下层数字找回)"""
        if plate_img is None or plate_img.size == 0: return []

        # 1. 预处理 (去边框)
        h, w = plate_img.shape[:2]
        plate_img = plate_img[3:h - 3, 3:w - 3]
        h, w = plate_img.shape[:2]
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        split_y = int(h * 0.42)  # 42% 黄金分割
        final_chars = []

        # ==========================================
        # 🟢 PART 1: 上层处理 (汉字/省份)
        # ==========================================
        top_gray = gray[0:split_y, :]

        # 【优化点1】C=8，配合胶水，既不过分细也不过分粗
        top_bin = cv2.adaptiveThreshold(top_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 13, 8)

        # 【优化点2】强力胶水：防止“莞”字破碎
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        top_bin = cv2.morphologyEx(top_bin, cv2.MORPH_OPEN, kernel_small)  # 去噪
        kernel_glue = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        top_bin = cv2.morphologyEx(top_bin, cv2.MORPH_CLOSE, kernel_glue)  # 粘合

        cnts_top, _ = cv2.findContours(top_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        top_candidates = []

        for cnt in cnts_top:
            x, y, w_c, h_c = cv2.boundingRect(cnt)
            cx = x + w_c / 2

            # 1. 左右位置 (放宽到 5%-95%)
            if cx > w * 0.85 or cx < w * 0.05: continue

            # 2. 宽高比
            if w_c / h_c > 1.3 or w_c / h_c < 0.1: continue

            # 3. 高度硬性过滤 (提高门槛，杀掉矮小的螺丝)
            if h_c < split_y * 0.35: continue

            # 4. 【核心】实心度过滤 (杀掉黑乎乎的螺丝)
            roi = top_bin[y:y + h_c, x:x + w_c]
            pixel_density = cv2.countNonZero(roi) / (w_c * h_c)
            if pixel_density > 0.80: continue  # 太实心了，肯定是螺丝

            top_candidates.append((x, y, w_c, h_c, top_bin[y:y + h_c, x:x + w_c]))

        # C位排序：取离中心最近的2个
        if len(top_candidates) > 2:
            top_candidates.sort(key=lambda c: abs((c[0] + c[2] / 2) - w / 2))
            top_candidates = top_candidates[:2]

        top_candidates.sort(key=lambda c: c[0])  # 还原左右顺序
        for item in top_candidates: final_chars.append(item[4])

        # ==========================================
        # 🔵 PART 2: 下层处理 (民主投票版 - 专治长条和套娃)
        # ==========================================
        bot_gray = gray[split_y:, :]
        bot_bin = cv2.adaptiveThreshold(bot_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 51, 15)
        cnts_bot, _ = cv2.findContours(bot_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 1. 海选：先把所有可能是字的都抓进来
        raw_items = []
        for cnt in cnts_bot:
            x, y, w_c, h_c = cv2.boundingRect(cnt)

            # 基础过滤：太矮的(小于35%)肯定是噪点
            if h_c < (h - split_y) * 0.35: continue
            # 基础过滤：太宽的肯定不是数字
            if w_c / h_c > 2.0: continue

            raw_items.append({'rect': (x, y, w_c, h_c), 'roi': bot_bin[y:y + h_c, x:x + w_c]})

        bot_items = []
        if raw_items:
            # ---------------------------------------------------
            # 🚩 核心逻辑：高度民主投票
            # ---------------------------------------------------
            # 计算所有候选框的高度中位数 (代表"正常数字"的高度)
            heights = [item['rect'][3] for item in raw_items]
            heights.sort()
            median_h = heights[len(heights) // 2]

            # 第一轮清洗：踢掉身高异常的
            clean_candidates = []
            for item in raw_items:
                h_curr = item['rect'][3]

                # 🚫 杀手锏：如果比中位数高出 20% (1.2倍)，说明它是边框或者长条！
                # 正常的数字高度误差通常不会超过 5-10%
                if h_curr > median_h * 1.2: continue

                # 🚫 顺便踢掉太矮的 (小于 70%)
                if h_curr < median_h * 0.7: continue

                clean_candidates.append(item)

            # ---------------------------------------------------
            # 🚩 第二轮清洗：内斗去重 (解决 Box-in-Box)
            # ---------------------------------------------------
            keep_indices = set(range(len(clean_candidates)))
            for i in range(len(clean_candidates)):
                for j in range(len(clean_candidates)):
                    if i == j: continue

                    xi, yi, wi, hi = clean_candidates[i]['rect']
                    xj, yj, wj, hj = clean_candidates[j]['rect']

                    # 计算重叠面积
                    xx1 = max(xi, xj)
                    yy1 = max(yi, yj)
                    xx2 = min(xi + wi, xj + wj)
                    yy2 = min(yi + hi, yj + hj)
                    w_inter = max(0, xx2 - xx1)
                    h_inter = max(0, yy2 - yy1)
                    inter_area = w_inter * h_inter

                    # 如果重叠严重，删掉面积大的那个（保留精细的内部框）
                    min_area = min(wi * hi, wj * hj)
                    if inter_area > min_area * 0.8:
                        if wi * hi > wj * hj:
                            if i in keep_indices: keep_indices.remove(i)
                        else:
                            if j in keep_indices: keep_indices.remove(j)

            # 收集幸存者
            for i in keep_indices:
                rect = clean_candidates[i]['rect']
                roi = clean_candidates[i]['roi']
                bot_items.append((rect[0], rect[1], rect[2], rect[3], roi))

        bot_items.sort(key=lambda item: item[0])  # 别忘了按从左到右排序
        for item in bot_items: final_chars.append(item[4])

        # ==========================================
        # 🏁 统一输出 & 调试
        # ==========================================
        resized_chars = []
        for img in final_chars:
            if img.shape[0] > 0 and img.shape[1] > 0:
                resized_chars.append(cv2.resize(img, (30, 60)))

        if self.debug:
            vis_debug = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.line(vis_debug, (0, split_y), (w, split_y), (0, 255, 255), 2)

            # 画上层红框
            for item in top_candidates:
                x, y, w_c, h_c, _ = item
                cv2.rectangle(vis_debug, (x, y), (x + w_c, y + h_c), (0, 0, 255), 2)
                cv2.putText(vis_debug, "Top", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # 画下层绿框
            for item in bot_items:  # 现在 bot_items 肯定存在了
                x, y, w_c, h_c, _ = item
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