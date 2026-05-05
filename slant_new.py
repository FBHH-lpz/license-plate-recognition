import cv2
import numpy as np
import os

# ==========================================
# 第一部分：混合字符模板库 (基底：slant_new.py)
# ==========================================
def generate_templates():
    templates = {}

    # ---------------------------------------
    # 🟢 PART 1: 数字和字母 (使用最佳实践：Hershey Plain + 厚度4)
    # ---------------------------------------
    alphanum = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"

    # print(">>> 正在生成数字/字母模板...")
    for char in alphanum:
        img = np.zeros((60, 30), dtype=np.uint8)
        # 核心参数：(3, 45), FONT_HERSHEY_PLAIN, 2.0, 4
        cv2.putText(img, char, (3, 45), cv2.FONT_HERSHEY_PLAIN, 2, 255, 4)

        coords = cv2.findNonZero(img)
        if coords is None:
            continue
        x, y, w, h = cv2.boundingRect(coords)
        img_crop = img[y:y + h, x:x + w]
        img_final = cv2.resize(img_crop, (30, 60))
        templates[char] = img_final

    # ---------------------------------------
    # 🔵 PART 2: 汉字 (从外部文件夹加载加粗版)
    # ---------------------------------------
    template_dir = "templates_bold"
    cn_map = {
        'guang': '广', 'zhou': '州', 'dong': '东',
        'guan': '莞', 'fo': '佛', 'shan': '山'
    }

    if os.path.exists(template_dir):
        for filename in os.listdir(template_dir):
            name_no_ext = os.path.splitext(filename)[0]
            if name_no_ext in cn_map:
                char = cn_map[name_no_ext]
                path = os.path.join(template_dir, filename)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # 确保黑底白字
                    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                    img = cv2.resize(img, (30, 60))
                    templates[char] = img
    else:
        print("⚠️ 警告：templates_bold 文件夹未找到，汉字识别将失效")

    print(f">>> 模板库就绪，共 {len(templates)} 个字符")
    return templates


# ==========================================
# 第二部分：核心处理类
# ==========================================
class LicensePlateRecognizer:
    def __init__(self):
        self.templates = generate_templates()
        self.debug = True

    # ---------------------------------------------------------
    # 🟢 模块 A：辅助函数 (通用)
    # ---------------------------------------------------------
    def white_pixel_ratio(self, roi, mask_white, x, y, w, h):
        mask_roi = mask_white[y:y + h, x:x + w]
        if mask_roi.size == 0: return 0.0
        return np.sum(mask_roi == 255) / (mask_roi.shape[0] * mask_roi.shape[1])

    def check_char_texture(self, roi):
        if roi is None or roi.size == 0: return 0.0
        h, w = roi.shape[:2]
        roi_enlarged = cv2.resize(roi, (int(w * 1.2), int(h * 1.2)))
        gray = cv2.cvtColor(roi_enlarged, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        h_bin, w_bin = binary.shape
        texture_score = 0
        for row in range(h_bin): texture_score += np.sum(np.abs(np.diff(binary[row, :]))) / 255
        for col in range(w_bin): texture_score += np.sum(np.abs(np.diff(binary[:, col]))) / 255
        return texture_score / (h_bin * w_bin * 2)

    def fit_plate_contour(self, cnt, img_shape):
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        x, y, w, h = cv2.boundingRect(box)
        if w < h: w, h = h, w
        return x, y, w, h, box

    # ---------------------------------------------------------
    # 🟢 模块 B：预处理 & 定位 (保留 slant_new 的 Unwarp 逻辑)
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
            cv2.imshow("1. Morph Combined", morph_final)

        return img_resized, morph_final, mask_white_clean

    def unwarp_plate(self, img, box):
        """ 透视变换矫正 (slant_new 特有功能，保留) """
        pts = box.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

        h, w = warped.shape[:2]
        if h > w: # 如果竖起来了，转一下
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

        return warped

    def locate_plate_dual_strategy(self, img_original, img_morph, mask_white):
        contours, _ = cv2.findContours(img_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300: continue
            x, y, w, h, box = self.fit_plate_contour(cnt, img_original.shape)
            if 1.5 < w / float(h) < 6.0:
                raw_candidates.append((x, y, w, h, area, box))

        valid_candidates = []
        for (x, y, w, h, area, box) in raw_candidates:
            roi = img_original[y:y + h, x:x + w]
            texture = self.check_char_texture(roi)
            white_r = self.white_pixel_ratio(roi, mask_white, x, y, w, h)

            if texture > 0.04 and white_r > 0.5:
                valid_candidates.append((x, y, w, h, area, box, texture, white_r))

        # 补救策略
        if not valid_candidates:
            gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
            _, binary_white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            cnts_white, _ = cv2.findContours(binary_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts_white:
                cnt_max = max(cnts_white, key=cv2.contourArea)
                if cv2.contourArea(cnt_max) > 1000:
                    x, y, w, h, box = self.fit_plate_contour(cnt_max, img_original.shape)
                    roi = img_original[y:y + h, x:x + w]
                    valid_candidates.append((x, y, w, h, cv2.contourArea(cnt_max), box, 0, 0))

        if not valid_candidates: return None

        # 排序取最优
        valid_candidates.sort(key=lambda x: (x[6] * 0.6 + x[7] * 0.3 + x[4] / 10000 * 0.1), reverse=True)
        best = valid_candidates[0]
        box = best[5]

        # 【关键】调用 Unwarp 进行矫正
        plate_img = self.unwarp_plate(img_original, box)

        if self.debug:
            temp = img_original.copy()
            cv2.drawContours(temp, [box], 0, (0, 255, 0), 2)
            cv2.imshow("5. Plate Location Box", temp)
            cv2.imshow("6. Plate Unwarped", plate_img)

        return plate_img

    # ---------------------------------------------------------
    # 🟢 模块 C：字符分割 (【替换为 optimize.py 的改进版】)
    # ---------------------------------------------------------
    def segment_chars(self, plate_img):
        """
        融合版分割：
        1. 包含 '剃头' (Anti-Glue) 逻辑，切断上方粘连。
        2. 包含 'C位优先' 逻辑，防止边缘噪点被当成汉字。
        3. 包含 丰富的调试可视化。
        """
        if plate_img is None or plate_img.size == 0: return [], []

        # 1. 预处理 (去边框)
        h, w = plate_img.shape[:2]
        plate_img = plate_img[3:h - 3, 3:w - 3]
        h, w = plate_img.shape[:2]
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        split_y = int(h * 0.42)

        # 准备返回容器
        top_imgs_raw = []
        bot_imgs_raw = []

        # =========================================================
        # 🟢 PART 1: 上层处理 (Anti-Glue + C-Sort)
        # =========================================================
        top_gray = gray[0:split_y, :]

        # 二值化
        top_bin = cv2.adaptiveThreshold(top_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 13, 10)

        # --- 【核心】剃头操作 ---
        safe_zone = int(split_y * 0.15)
        top_bin[0:safe_zone, :] = 0

        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        top_bin = cv2.morphologyEx(top_bin, cv2.MORPH_OPEN, kernel)
        top_bin = cv2.morphologyEx(top_bin, cv2.MORPH_CLOSE, kernel)

        cnts_top, _ = cv2.findContours(top_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 调试容器
        debug_raw_contours = []
        top_candidates = []

        for cnt in cnts_top:
            x, y, w_c, h_c = cv2.boundingRect(cnt)
            cx = x + w_c / 2
            debug_raw_contours.append((x, y, w_c, h_c)) # 记录所有轮廓

            # --- 筛选逻辑 ---
            if cx > w * 0.85 or cx < w * 0.15: continue
            if w_c / h_c > 1.5 or w_c / h_c < 0.1: continue
            if h_c < split_y * 0.2: continue

            roi = top_bin[y:y + h_c, x:x + w_c]
            if cv2.countNonZero(roi) / (w_c * h_c) > 0.95: continue

            top_candidates.append((x, y, w_c, h_c, roi))

        # 【核心】C位排序逻辑
        if len(top_candidates) > 2:
            # 优先选择离中心最近的
            top_candidates.sort(key=lambda c: abs((c[0] + c[2] / 2) - w / 2))
            top_candidates = top_candidates[:2]

        # 选定后按 X 轴重新排序
        top_candidates.sort(key=lambda c: c[0])

        for item in top_candidates:
            top_imgs_raw.append(item[4])

        # =========================================================
        # 🔵 PART 2: 下层处理 (保持稳定逻辑)
        # =========================================================
        bot_gray = gray[split_y:, :]
        bot_bin = cv2.adaptiveThreshold(bot_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 51, 15)
        cnts_bot, _ = cv2.findContours(bot_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_items = []
        for cnt in cnts_bot:
            x, y, w_c, h_c = cv2.boundingRect(cnt)
            if h_c < (h - split_y) * 0.35: continue
            if w_c / h_c > 2.0: continue
            raw_items.append({'rect': (x, y, w_c, h_c), 'roi': bot_bin[y:y + h_c, x:x + w_c]})

        bot_items = []
        if raw_items:
            # 高度中位数过滤
            heights = [item['rect'][3] for item in raw_items]
            heights.sort()
            median_h = heights[len(heights) // 2]
            clean_candidates = [item for item in raw_items if median_h * 0.7 < item['rect'][3] < median_h * 1.2]

            # 去重叠
            keep_indices = set(range(len(clean_candidates)))
            for i in range(len(clean_candidates)):
                for j in range(len(clean_candidates)):
                    if i == j: continue
                    xi, yi, wi, hi = clean_candidates[i]['rect']
                    xj, yj, wj, hj = clean_candidates[j]['rect']
                    xx1 = max(xi, xj); yy1 = max(yi, yj)
                    xx2 = min(xi + wi, xj + wj); yy2 = min(yi + hi, yj + hj)
                    w_inter = max(0, xx2 - xx1); h_inter = max(0, yy2 - yy1)
                    if (w_inter * h_inter) > min(wi * hi, wj * hj) * 0.8:
                        if wi * hi > wj * hj: keep_indices.discard(i)
                        else: keep_indices.discard(j)

            for i in keep_indices:
                rect = clean_candidates[i]['rect']
                bot_items.append((rect[0], rect[1], rect[2], rect[3], clean_candidates[i]['roi']))

        bot_items.sort(key=lambda item: item[0])
        for item in bot_items: bot_imgs_raw.append(item[4])

        # =========================================================
        # 🏁 输出与调试可视化
        # =========================================================
        top_resized = [cv2.resize(img, (30, 60)) for img in top_imgs_raw]
        bot_resized = [cv2.resize(img, (30, 60)) for img in bot_imgs_raw]

        if self.debug:
            vis_debug = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.line(vis_debug, (0, split_y), (w, split_y), (0, 255, 255), 2)
            cv2.line(vis_debug, (0, safe_zone), (w, safe_zone), (255, 0, 255), 1)

            # 蓝色：所有原始轮廓
            for item in debug_raw_contours:
                x, y, w_c, h_c = item
                cv2.rectangle(vis_debug, (x, y), (x + w_c, y + h_c), (255, 0, 0), 1)

            # 绿色：最终入选者
            for item in top_candidates:
                x, y, w_c, h_c, _ = item
                cv2.rectangle(vis_debug, (x, y), (x + w_c, y + h_c), (0, 255, 0), 2)

            for item in bot_items:
                x, y, w_c, h_c = item[:4]
                cv2.rectangle(vis_debug, (x, y + split_y), (x + w_c, y + h_c + split_y), (0, 255, 0), 2)

            cv2.imshow("7. Segment Razor (Blue=Raw)", vis_debug)

        return top_resized, bot_resized

    # ---------------------------------------------------------
    # 🟢 模块 D: 识别 (【替换为 optimize.py 的改进版】)
    # ---------------------------------------------------------
    def recognize_smart(self, top_imgs, bot_imgs):
        """
        融合版识别：
        1. 带有像素级拼图调试功能。
        2. 兼容 BGR/Grayscale 输入。
        """
        # === PART 1: 上层汉字 ===
        top_candidates = "广州东莞佛山"
        raw_top_results = []
        debug_top_visuals = [] # 可视化容器

        print("\n--- 🔍 上层汉字识别详情 ---")

        for idx, img in enumerate(top_imgs):
            if img is None or img.size == 0: continue

            # V1 核心逻辑：确保二值化
            if len(img.shape) == 2:
                _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            else:
                img_bin = img

            img_resize = cv2.resize(img_bin, (30, 60))
            best_score = -1
            best_char = "?"
            debug_scores = []

            for char in top_candidates:
                if char not in self.templates: continue
                score = cv2.matchTemplate(img_resize, self.templates[char], cv2.TM_CCOEFF_NORMED)[0][0]
                debug_scores.append((char, score))
                if score > best_score:
                    best_score = score
                    best_char = char

            debug_scores.sort(key=lambda x: x[1], reverse=True)
            print(f"  位置[{idx}] 识别为 '{best_char}' | Top3: {[(x[0], f'{x[1]:.2f}') for x in debug_scores[:3]]}")

            if best_score > 0.4:
                raw_top_results.append(best_char)

            # --- 🛠️ 调试可视化生成 ---
            vis_char = cv2.cvtColor(img_resize, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(vis_char, (0, 0), (29, 59), (100, 100, 100), 1)
            cv2.putText(vis_char, best_char, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            score_color = (0, 255, 0) if best_score > 0.6 else (0, 0, 255)
            cv2.putText(vis_char, f"{best_score:.2f}", (2, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.35, score_color, 1)
            debug_top_visuals.append(vis_char)

        # 展示上层调试拼图
        if debug_top_visuals:
            top_montage = np.hstack(debug_top_visuals)
            h, w = top_montage.shape[:2]
            top_montage_large = cv2.resize(top_montage, (w * 4, h * 4), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("DEBUG: Top Row Inputs (Pixel View)", top_montage_large)

        # 城市补全逻辑
        top_raw_str = "".join(raw_top_results)
        city_map = {'广': '广州', '佛': '佛山', '山': '佛山', '东': '东莞', '莞': '东莞'}
        final_top = top_raw_str
        for key, full_name in city_map.items():
            if key in top_raw_str:
                final_top = full_name
                break
        if not final_top and raw_top_results:
            final_top = top_raw_str

        # === PART 2: 下层数字 ===
        bot_text = ""
        chars_all = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"

        for i, img in enumerate(bot_imgs):
            img_resize = cv2.resize(img, (30, 60))
            best_char = "?"
            best_score = -1

            for char in chars_all:
                if char not in self.templates: continue
                score = cv2.matchTemplate(img_resize, self.templates[char], cv2.TM_CCOEFF_NORMED)[0][0]
                if score > best_score:
                    best_score = score
                    best_char = char

            if best_char in ['6', '8', 'B', 'G']:
                best_char = self.refine_char(img_resize, best_char)
            if best_char == 'D' and i > 1: best_char = '0'

            bot_text += best_char

        return final_top + bot_text

    def refine_char(self, char_img, candidate):
        """ 万能修正器 (带Debug打印) """
        h, w = 60, 30
        img = cv2.resize(char_img, (w, h))
        if len(img.shape) == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 自动纠正底色
        border_pixels = np.concatenate([img_bin[0, :], img_bin[-1, :], img_bin[:, 0], img_bin[:, -1]])
        if cv2.countNonZero(border_pixels) > border_pixels.size / 2:
            img_bin = cv2.bitwise_not(img_bin)

        # 数洞洞
        contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        hole_count = 0
        if hierarchy is not None:
            for i in range(len(contours)):
                if hierarchy[0][i][3] != -1: # 是内部轮廓
                    if cv2.contourArea(contours[i]) > 10:
                        hole_count += 1

        print(f"DEBUG Fix: 原字符='{candidate}' | 洞数量={hole_count}")

        if hole_count >= 2:
            if candidate == 'B': return 'B'
            return '8'
        if hole_count == 1:
            return '6'
        if hole_count == 0:
            if candidate == 'G': return 'G'
            if candidate == '6': return '6'
            roi_neck = img_bin[12:22, 18:30]
            if cv2.countNonZero(roi_neck) / roi_neck.size > 0.45:
                return '8'
            else:
                return '6'
        return candidate


# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    img_path = "images/slant/2.jpg"

    if os.path.exists(img_path):
        print(f"处理图片: {img_path}")
        recognizer = LicensePlateRecognizer()

        # 1. 预处理
        resized, morph, mask = recognizer.preprocess_enhanced(cv2.imread(img_path))

        # 2. 定位 (会调用 unwarp_plate 矫正倾斜)
        plate = recognizer.locate_plate_dual_strategy(resized, morph, mask)

        if plate is not None:
            # 3. 分割 (使用剃头+C位逻辑)
            top_imgs, bot_imgs = recognizer.segment_chars(plate)

            if top_imgs or bot_imgs:
                # 4. 识别 (带像素可视化拼图)
                text = recognizer.recognize_smart(top_imgs, bot_imgs)
                print(f"\n✅ 最终结果: {text}")
            else:
                print("❌ 分割失败")
        else:
            print("❌ 定位失败")

        cv2.waitKey(0)
        cv2.destroyAllWindows()