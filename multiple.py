import cv2
import numpy as np
import os


# ==========================================
# 第一部分：混合字符模板库
# ==========================================
def generate_templates():
    templates = {}

    # --- PART 1: 数字/字母 (OpenCV 绘制) ---
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

    # --- PART 2: 汉字 (从 templates_cn 文件夹加载) ---
    template_dir = "templates_cn"
    cn_map = {'guang': '广', 'zhou': '州', 'dong': '东', 'guan': '莞', 'fo': '佛', 'shan': '山'}

    if os.path.exists(template_dir):
        for filename in os.listdir(template_dir):
            name_no_ext = os.path.splitext(filename)[0]
            if name_no_ext in cn_map:
                char = cn_map[name_no_ext]
                path = os.path.join(template_dir, filename)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # 简单二值化，保证和 putText 的风格一致
                    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                    img = cv2.resize(img, (30, 60))
                    templates[char] = img
    else:
        print("⚠️ 警告: 未找到 templates_cn 文件夹，仅能识别数字")

    print(f">>> 模板库加载完毕，共 {len(templates)} 个字符")
    return templates


# ==========================================
# 第二部分：核心处理类
# ==========================================
class LicensePlateRecognizer:
    def __init__(self):
        self.templates = generate_templates()
        self.debug = True  # 开启调试窗口

    # ---------------------------------------------------------
    # 🟢 模块 A: 基础工具 (NMS, 透视变换, 纹理检查)
    # ---------------------------------------------------------
    def order_points(self, pts):
        """对四个角点排序：左上, 右上, 右下, 左下"""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def unwarp_plate(self, img, box):
        """透视变换：把歪斜的框拉直"""
        rect = self.order_points(box)
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0], [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

        # 防止竖向长条 (旋转90度修正)
        if maxHeight > maxWidth:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        return warped

    def nms(self, boxes, overlapThresh=0.2):
        """非极大值抑制：去除重叠框"""
        if len(boxes) == 0: return []
        pick = []
        # boxes结构: (x, y, w, h, score, box_points)
        x1 = np.array([b[0] for b in boxes])
        y1 = np.array([b[1] for b in boxes])
        x2 = np.array([b[0] + b[2] for b in boxes])
        y2 = np.array([b[1] + b[3] for b in boxes])
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.arange(len(boxes))

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[0]  # 取当前分数最高的
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[1:]])
            yy1 = np.maximum(y1[i], y1[idxs[1:]])
            xx2 = np.minimum(x2[i], x2[idxs[1:]])
            yy2 = np.minimum(y2[i], y2[idxs[1:]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlap = (w * h) / area[idxs[1:]]

            delete_idxs = np.where(overlap > overlapThresh)[0] + 1
            idxs = np.delete(idxs, np.concatenate(([0], delete_idxs)))

        return [boxes[i] for i in pick]

    def check_char_texture(self, roi):
        if roi is None or roi.size == 0: return 0.0
        h, w = roi.shape[:2]
        roi_enlarged = cv2.resize(roi, (int(w * 1.2), int(h * 1.2)))
        gray = cv2.cvtColor(roi_enlarged, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        h_bin, w_bin = binary.shape
        score = (np.sum(np.abs(np.diff(binary, axis=0))) + np.sum(np.abs(np.diff(binary, axis=1)))) / 255
        return score / (h_bin * w_bin * 2)

    def white_pixel_ratio(self, roi, mask_white, x, y, w, h):
        mask_roi = mask_white[y:y + h, x:x + w]
        if mask_roi.size == 0: return 0.0
        return np.sum(mask_roi == 255) / mask_roi.size

    def fit_plate_contour(self, cnt, img_shape):
        rect = cv2.minAreaRect(cnt)
        box = np.int32(cv2.boxPoints(rect))
        x, y, w, h = cv2.boundingRect(box)
        if w < h: w, h = h, w
        return x, y, w, h, box

    # ---------------------------------------------------------
    # 🟢 模块 B: 预处理 & 定位 (多车牌版)
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
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        morph_combined = cv2.bitwise_and(binary_edges, mask_white)
        morph_final = cv2.morphologyEx(morph_combined, cv2.MORPH_CLOSE, np.ones((15, 5), np.uint8))

        return img_resized, morph_final, mask_white

    def locate_plates_multi(self, img_original, img_morph, mask_white):
        """多车牌定位 + 透视矫正"""
        contours, _ = cv2.findContours(img_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_candidates = []

        # 1. 初筛
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300: continue
            x, y, w, h, box = self.fit_plate_contour(cnt, img_original.shape)
            # 放宽比例以适应斜视
            if 1.2 < w / h < 7.0:
                raw_candidates.append((x, y, w, h, area, box))

        # 2. 精筛 & 打分
        valid_candidates = []
        for (x, y, w, h, area, box) in raw_candidates:
            roi = img_original[y:y + h, x:x + w]
            texture = self.check_char_texture(roi)
            white_r = self.white_pixel_ratio(roi, mask_white, x, y, w, h)

            # 综合打分公式
            score = texture * 0.6 + white_r * 0.3 + (area / 10000) * 0.1

            # 门槛
            if texture > 0.03 and white_r > 0.4:
                valid_candidates.append((x, y, w, h, score, box))

        if not valid_candidates: return []

        # 3. 排序 & NMS去重
        valid_candidates.sort(key=lambda x: x[4], reverse=True)
        clean_candidates = self.nms(valid_candidates, overlapThresh=0.2)

        # 4. 提取与矫正
        results = []
        for item in clean_candidates:
            box = item[5]  # 取出 box 坐标 (4个点)
            plate_img = self.unwarp_plate(img_original, box.astype("float32"))
            results.append(plate_img)

            if self.debug:
                cv2.drawContours(img_original, [box], 0, (0, 255, 0), 2)

        if self.debug:
            cv2.imshow("Detection Result", img_original)

        return results

    # ---------------------------------------------------------
    # 🟢 模块 C: 混合分割 (上V1 + 下V5)
    # ---------------------------------------------------------
    def segment_chars(self, plate_img):
        if plate_img is None or plate_img.size == 0: return [], []
        h, w = plate_img.shape[:2]
        plate_img = plate_img[3:h - 3, 3:w - 3]  # 去边
        h, w = plate_img.shape[:2]
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        split_y = int(h * 0.42)

        top_imgs_raw, bot_imgs_raw = [], []

        # --- Top (V1 Logic) ---
        top_gray = gray[0:split_y, :]
        # V1 参数：阈值 (13, 10)，胶水 (2, 2)
        top_bin = cv2.adaptiveThreshold(top_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 10)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        top_bin = cv2.morphologyEx(top_bin, cv2.MORPH_CLOSE, kernel)

        cnts_top, _ = cv2.findContours(top_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        top_candidates = []
        for cnt in cnts_top:
            x, y, w_c, h_c = cv2.boundingRect(cnt)
            cx = x + w_c / 2
            if cx > w * 0.85 or cx < w * 0.05: continue
            if not (0.1 < w_c / h_c < 1.3): continue
            if h_c < split_y * 0.2: continue  # V1 低门槛
            roi = top_bin[y:y + h_c, x:x + w_c]
            if cv2.countNonZero(roi) / (w_c * h_c) > 0.95: continue
            top_candidates.append((x, roi))

        if len(top_candidates) > 2:  # 简单的中间优先
            top_candidates.sort(key=lambda c: abs((c[0] + c[1].shape[1] / 2) - w / 2))
            top_candidates = top_candidates[:2]
        top_candidates.sort(key=lambda c: c[0])
        top_imgs_raw = [c[1] for c in top_candidates]

        # --- Bot (V5 Logic) ---
        bot_gray = gray[split_y:, :]
        bot_bin = cv2.adaptiveThreshold(bot_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 15)
        cnts_bot, _ = cv2.findContours(bot_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_items = []
        for cnt in cnts_bot:
            x, y, w_c, h_c = cv2.boundingRect(cnt)
            if h_c < (h - split_y) * 0.35: continue  # V5 高门槛
            if w_c / h_c > 2.0: continue
            raw_items.append({'rect': (x, y, w_c, h_c), 'roi': bot_bin[y:y + h_c, x:x + w_c]})

        # V5 投票去重
        bot_items = []
        if raw_items:
            heights = sorted([i['rect'][3] for i in raw_items])
            median_h = heights[len(heights) // 2]
            clean = [i for i in raw_items if median_h * 0.7 < i['rect'][3] < median_h * 1.2]

            # IOU 去重简化版
            clean.sort(key=lambda i: i['rect'][0])
            if clean:
                bot_items.append(clean[0])
                for i in range(1, len(clean)):
                    prev = bot_items[-1]['rect']
                    curr = clean[i]['rect']
                    if curr[0] < prev[0] + prev[2] * 0.8:  # 重叠严重
                        if curr[2] * curr[3] > prev[2] * prev[3]:  # 取大的
                            bot_items.pop()
                            bot_items.append(clean[i])
                    else:
                        bot_items.append(clean[i])

        bot_imgs_raw = [i['roi'] for i in bot_items]
        return top_imgs_raw, bot_imgs_raw

    # ---------------------------------------------------------
    # 🟢 模块 D: 智能识别 (带打分与修正)
    # ---------------------------------------------------------
    def recognize_smart(self, top_imgs, bot_imgs):
        # --- Top (V1 暴力匹配 + 打分) ---
        top_candidates = "广洲东莞佛山"
        raw_top_results = []

        print(f"\n  [汉字层] 检测到 {len(top_imgs)} 个字符")
        for idx, img in enumerate(top_imgs):
            if len(img.shape) == 2: _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            img_resize = cv2.resize(img, (30, 60))

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
            print(
                f"    字符{idx + 1}: 识别为 '{best_char}' (分:{best_score:.2f}) | Top3: {[(x[0], f'{x[1]:.2f}') for x in debug_scores[:3]]}")
            if best_score > 0.4: raw_top_results.append(best_char)

        top_str = "".join(raw_top_results)
        city_map = {'广': '广州', '佛': '佛山', '山': '佛山', '东': '东莞', '莞': '东莞'}
        final_top = top_str
        for k, v in city_map.items():
            if k in top_str: final_top = v; break
        if not final_top and top_str: final_top = top_str

        # --- Bot (V1 匹配 + V5 拓扑修正) ---
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

            # V5 拓扑修正
            orig = best_char
            if best_char in ['6', '8', 'B', 'G']:
                best_char = self.refine_char(img_resize, best_char)
            if orig != best_char: print(f"    [修正] 下层字符{i + 1}: {orig} -> {best_char}")

            if best_char == 'D' and i > 1: best_char = '0'  # V1 经验
            bot_text += best_char

        return final_top + bot_text

    def refine_char(self, char_img, candidate):
        """数洞洞修正法"""
        h, w = char_img.shape[:2]
        if len(char_img.shape) == 3: char_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
        _, img_bin = cv2.threshold(char_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if cv2.countNonZero(img_bin) > img_bin.size / 2: img_bin = cv2.bitwise_not(img_bin)

        contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        hole_count = 0
        if hierarchy is not None:
            for i in range(len(contours)):
                if hierarchy[0][i][3] != -1 and cv2.contourArea(contours[i]) > 10:
                    hole_count += 1

        if hole_count >= 2: return 'B' if candidate == 'B' else '8'
        if hole_count == 1: return '6'
        if hole_count == 0:
            if candidate == 'G': return 'G'
            roi_neck = img_bin[12:22, 18:30]
            if cv2.countNonZero(roi_neck) / roi_neck.size > 0.45: return '8'
            return '6'
        return candidate


# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # ⚠️ 修改这里：你的多车测试图片路径
    img_path = "images/many.PNG"

    if os.path.exists(img_path):
        print(f"\n========================================")
        print(f"🚀 开始处理: {img_path}")
        print(f"========================================\n")

        recognizer = LicensePlateRecognizer()

        # 1. 预处理
        img_raw = cv2.imread(img_path)
        resized, morph, mask = recognizer.preprocess_enhanced(img_raw)

        # 2. 定位 (多车牌 + 矫正)
        plates = recognizer.locate_plates_multi(resized, morph, mask)

        print(f"🔍 检测到 {len(plates)} 个潜在车牌区域")

        for i, plate in enumerate(plates):
            print(f"\n----------------------------------------")
            print(f"🚗 正在分析第 {i + 1} 个车牌...")

            # 3. 分割
            top_imgs, bot_imgs = recognizer.segment_chars(plate)

            if top_imgs or bot_imgs:
                # 4. 识别
                text = recognizer.recognize_smart(top_imgs, bot_imgs)
                print(f"✅ 最终结果: {text}")

                cv2.imshow(f"Plate_{i + 1}", plate)
            else:
                print("❌ 分割失败 (可能不是车牌)")

        print("\n按任意键退出...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"❌ 找不到图片: {img_path}")