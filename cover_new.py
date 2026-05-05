import cv2
import numpy as np
import os


# ==========================================
# 第一部分：字符模板 (混合字符模板库)
# ==========================================
def generate_templates():
    templates = {}

    # ---------------------------------------
    # 🟢 PART 1: 数字和字母
    # ---------------------------------------
    alphanum = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"

    for char in alphanum:
        img = np.zeros((60, 30), dtype=np.uint8)
        cv2.putText(img, char, (3, 45), cv2.FONT_HERSHEY_PLAIN, 2, 255, 4)
        coords = cv2.findNonZero(img)
        if coords is None: continue
        x, y, w, h = cv2.boundingRect(coords)
        img_crop = img[y:y + h, x:x + w]
        img_final = cv2.resize(img_crop, (30, 60))
        templates[char] = img_final

    # ---------------------------------------
    # 🔵 PART 2: 汉字 (优先尝试两个文件夹)
    # ---------------------------------------
    template_dir = "templates_cn"
    if os.path.exists("templates_bold"):
        template_dir = "templates_bold"

    cn_map = {'guang': '广', 'zhou': '州', 'dong': '东', 'guan': '莞', 'fo': '佛', 'shan': '山'}

    if os.path.exists(template_dir):
        for filename in os.listdir(template_dir):
            name_no_ext = os.path.splitext(filename)[0]
            if name_no_ext in cn_map:
                char = cn_map[name_no_ext]
                path = os.path.join(template_dir, filename)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                    img = cv2.resize(img, (30, 60))
                    templates[char] = img
    else:
        print(f"⚠️ 警告：未找到模板文件夹 ({template_dir})，汉字无法识别！")

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
    # 🟢 模块 A：辅助函数
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
        texture_score = (np.sum(np.abs(np.diff(binary, axis=0))) + np.sum(np.abs(np.diff(binary, axis=1)))) / 255
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
    # 🟢 模块 B：预处理 & 定位
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
            cv2.imshow("3. Combined Morph", morph_final)

        return img_resized, morph_final, mask_white_clean

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

        valid_candidates.sort(key=lambda x: (x[6] * 0.6 + x[7] * 0.3 + x[4] / 10000 * 0.1), reverse=True)
        best = valid_candidates[0]
        box = best[5]

        plate_img = self.unwarp_plate(img_original, box.astype("float32"))

        if self.debug:
            temp = img_original.copy()
            cv2.drawContours(temp, [box], 0, (0, 255, 0), 2)
            cv2.imshow("5. Plate Location (Box)", temp)
            cv2.imshow("6. Plate Unwarped", plate_img)

        return plate_img

    # ---------------------------------------------------------
    # 🟢 模块 C：字符分割 (Anti-Glue)
    # ---------------------------------------------------------
    def segment_chars(self, plate_img):
        if plate_img is None or plate_img.size == 0: return [], []

        h, w = plate_img.shape[:2]
        plate_img = plate_img[3:h - 3, 3:w - 3]
        h, w = plate_img.shape[:2]
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        split_y = int(h * 0.42)

        top_imgs_raw = []
        bot_imgs_raw = []

        # === PART 1: 上层处理 ===
        top_gray = gray[0:split_y, :]
        top_bin = cv2.adaptiveThreshold(top_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 13, 10)

        # 剃头 (Anti-Glue)
        safe_zone = int(split_y * 0.25)
        top_bin[0:safe_zone, :] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        top_bin = cv2.morphologyEx(top_bin, cv2.MORPH_OPEN, kernel)
        top_bin = cv2.morphologyEx(top_bin, cv2.MORPH_CLOSE, kernel)

        cnts_top, _ = cv2.findContours(top_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        top_candidates = []

        for cnt in cnts_top:
            x, y, w_c, h_c = cv2.boundingRect(cnt)
            cx = x + w_c / 2
            if cx > w * 0.85 or cx < w * 0.15: continue
            if w_c / h_c > 1.5 or w_c / h_c < 0.1: continue
            if h_c < split_y * 0.2: continue

            roi = top_bin[y:y + h_c, x:x + w_c]
            if cv2.countNonZero(roi) / (w_c * h_c) > 0.95: continue

            top_candidates.append((x, y, w_c, h_c, roi))

        if len(top_candidates) > 2:
            top_candidates.sort(key=lambda c: abs((c[0] + c[2] / 2) - w / 2))
            top_candidates = top_candidates[:2]
        top_candidates.sort(key=lambda c: c[0])
        for item in top_candidates: top_imgs_raw.append(item[4])

        # === PART 2: 下层处理 ===
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
            heights = [item['rect'][3] for item in raw_items]
            heights.sort()
            median_h = heights[len(heights) // 2]
            clean_candidates = [item for item in raw_items if median_h * 0.7 < item['rect'][3] < median_h * 1.2]

            keep_indices = set(range(len(clean_candidates)))
            for i in range(len(clean_candidates)):
                for j in range(len(clean_candidates)):
                    if i == j: continue
                    xi, yi, wi, hi = clean_candidates[i]['rect']
                    xj, yj, wj, hj = clean_candidates[j]['rect']
                    xx1, yy1 = max(xi, xj), max(yi, yj)
                    xx2, yy2 = min(xi + wi, xj + wj), min(yi + hi, yj + hj)
                    if max(0, xx2 - xx1) * max(0, yy2 - yy1) > min(wi * hi, wj * hj) * 0.8:
                        if wi * hi > wj * hj:
                            keep_indices.discard(i)
                        else:
                            keep_indices.discard(j)

            for i in keep_indices:
                rect = clean_candidates[i]['rect']
                bot_items.append((rect[0], rect[1], rect[2], rect[3], clean_candidates[i]['roi']))

        bot_items.sort(key=lambda item: item[0])
        for item in bot_items: bot_imgs_raw.append(item[4])

        top_resized = [cv2.resize(img, (30, 60)) for img in top_imgs_raw]
        bot_resized = [cv2.resize(img, (30, 60)) for img in bot_imgs_raw]

        if self.debug:
            vis_debug = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.line(vis_debug, (0, split_y), (w, split_y), (0, 255, 255), 2)
            cv2.line(vis_debug, (0, safe_zone), (w, safe_zone), (255, 0, 255), 1)
            cv2.imshow("7. Segment Razor (Anti-Glue)", vis_debug)

        return top_resized, bot_resized

    # ---------------------------------------------------------
    # 🟢 模块 D: 识别 (逻辑更新：上层至少识别1个就算正常)
    # ---------------------------------------------------------
    def recognize_smart(self, top_imgs, bot_imgs):
        top_text = ""
        bot_text = ""

        total_chars_found = 0
        suspicious_objects = 0

        MIN_SCORE = 0.45
        OCCLUSION_SCORE = 0.35

        print("\n--- 🔍 开始遮挡与识别分析 ---")
        debug_top_visuals = []

        # === PART 1: 上层汉字 (更新逻辑) ===
        top_candidates = "广州东莞佛山"
        top_valid_count = 0  # 统计上层有效识别个数
        top_raw_chars = []  # 暂存识别结果

        for i, img in enumerate(top_imgs):
            if len(img.shape) == 2:
                _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            else:
                img_bin = img

            img_resize = cv2.resize(img_bin, (30, 60))
            best_score = -1
            best_char = "?"

            for char in top_candidates:
                if char not in self.templates: continue
                score = cv2.matchTemplate(img_resize, self.templates[char], cv2.TM_CCOEFF_NORMED)[0][0]
                if score > best_score:
                    best_score = score
                    best_char = char

            # 暂存结果，不立刻判断遮挡
            if best_score > MIN_SCORE:
                top_valid_count += 1
                top_raw_chars.append(best_char)
                print(f"  上层[{i}]: '{best_char}' (分:{best_score:.2f}) -> ✅ 识别成功")
            else:
                top_raw_chars.append("?")
                print(f"  上层[{i}]: 未知 (分:{best_score:.2f}) -> ❓ 待定")

            # 可视化调试
            vis_char = cv2.cvtColor(img_resize, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(vis_char, (0, 0), (29, 59), (100, 100, 100), 1)
            cv2.putText(vis_char, best_char, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            debug_top_visuals.append(vis_char)

        # ⚡️【核心修改】上层遮挡判定逻辑 ⚡️
        # 只要上层能认出至少 1 个字，就认为“没有遮挡”，只是识别不全。
        # 只有当图片有切割结果(top_imgs不为空)，但一个都认不出来时，才算遮挡。
        if len(top_imgs) > 0 and top_valid_count == 0:
            print("  ⚠️ 上层警告: 所有字符均无法识别 -> 判定为疑似遮挡")
            suspicious_objects += 1
        else:
            print(f"  ℹ️ 上层状态: 识别出 {top_valid_count} 个字符 -> 正常")

        top_text = "".join(top_raw_chars)
        total_chars_found += top_valid_count

        if self.debug and debug_top_visuals:
            top_montage = np.hstack(debug_top_visuals)
            h, w = top_montage.shape[:2]
            top_montage_large = cv2.resize(top_montage, (w * 4, h * 4), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("DEBUG: Top Pixels (Merged)", top_montage_large)

        # 城市修正 (如果只认出一个，比如“广?”，尝试补全)
        city_map = {'广': '广州', '佛': '佛山', '山': '佛山', '东': '东莞', '莞': '东莞'}
        final_top = top_text
        for k, v in city_map.items():
            if k in top_text:
                final_top = v
                break

        # === PART 2: 下层数字 ===
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

            if best_char in ['6', '8', 'B', 'G'] and best_score > MIN_SCORE:
                best_char = self.refine_char(img_resize, best_char)
            if best_char == 'D' and i > 1: best_char = '0'

            if best_score > MIN_SCORE:
                bot_text += best_char
                total_chars_found += 1
                print(f"  下层[{i}]: '{best_char}' (分:{best_score:.2f}) -> ✅ 正常")
            elif best_score < OCCLUSION_SCORE:
                suspicious_objects += 1
                bot_text += "?"
                print(f"  下层[{i}]: 未知物体 (分:{best_score:.2f}) -> ⚠️ 疑似遮挡/污渍")
            else:
                bot_text += "?"

        final_text = final_top + bot_text

        # === PART 3: 诊断报告 ===
        print(f"--- 📊 诊断结果 ---")
        print(f"  识别结果: {final_text}")
        print(f"  有效字符: {total_chars_found}, 可疑异物: {suspicious_objects}")

        status = "正常"
        if suspicious_objects >= 1:
            status = "【警告】车牌存在局部遮挡/污渍"
        elif total_chars_found < 3:  # 放宽标准，只要能认出3个以上就算勉强能看
            status = "【严重】车牌大面积遮挡或残缺"

        print(f"✅ 最终结论: {status}")
        return f"{final_text} | {status}"

    def refine_char(self, char_img, candidate):
        h, w = 60, 30
        img = cv2.resize(char_img, (w, h))
        if len(img.shape) == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        border = np.concatenate([img_bin[0, :], img_bin[-1, :], img_bin[:, 0], img_bin[:, -1]])
        if cv2.countNonZero(border) > border.size / 2: img_bin = cv2.bitwise_not(img_bin)

        contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        hole_count = 0
        if hierarchy is not None:
            for i in range(len(contours)):
                if hierarchy[0][i][3] != -1 and cv2.contourArea(contours[i]) > 10:
                    hole_count += 1

        print(f"DEBUG Fix: '{candidate}' -> holes={hole_count}")
        if hole_count >= 2: return 'B' if candidate == 'B' else '8'
        if hole_count == 1: return '6'
        if hole_count == 0:
            if candidate == 'G': return 'G'
            roi_neck = img_bin[12:22, 18:30]
            return '8' if cv2.countNonZero(roi_neck) / roi_neck.size > 0.45 else '6'
        return candidate

    # ---------------------------------------------------------
    # 🟢 模块 E: 透视变换辅助
    # ---------------------------------------------------------
    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def unwarp_plate(self, img, box):
        rect = self.order_points(box)
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
        if warped.shape[0] > warped.shape[1]:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        return warped


# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    img_path = "images/cover/1.jpg"

    if os.path.exists(img_path):
        print(f"处理图片: {img_path}")
        recognizer = LicensePlateRecognizer()
        resized, morph, mask = recognizer.preprocess_enhanced(cv2.imread(img_path))

        plate = recognizer.locate_plate_dual_strategy(resized, morph, mask)

        if plate is not None:
            top_imgs, bot_imgs = recognizer.segment_chars(plate)

            if top_imgs or bot_imgs:
                text = recognizer.recognize_smart(top_imgs, bot_imgs)
                print(f"\n✅ 最终结果: {text}")
            else:
                print("❌ 分割失败")
        else:
            print("❌ 定位失败")

    cv2.waitKey(0)
    cv2.destroyAllWindows()