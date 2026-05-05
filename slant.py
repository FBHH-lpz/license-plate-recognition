import cv2
import numpy as np
import os


# ==========================================
# 第一部分：字符模板 (更新：加入广州汉字)
# ==========================================
# ==========================================
# 第一部分：字符模板 (修改版：从文件夹加载)
# ==========================================
# ==========================================
# 第一部分：混合字符模板库
# ==========================================
def generate_templates():
    templates = {}

    # ---------------------------------------
    # 🟢 PART 1: 数字和字母 (保持原始代码逻辑)
    # ---------------------------------------
    # 这种线框字体对处理后的数字匹配度很好
    alphanum = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"

    # print(">>> 正在生成数字/字母模板 (原始方法)...")
    for char in alphanum:
        # 原始代码的参数：黑底，(3,45)坐标，HERSHEY_PLAIN字体，字号2，厚度4
        img = np.zeros((60, 30), dtype=np.uint8)
        cv2.putText(img, char, (3, 45), cv2.FONT_HERSHEY_PLAIN, 2, 255, 4)

        # 裁剪掉多余黑边，再拉伸回 30x60
        coords = cv2.findNonZero(img)
        if coords is None:
            continue
        x, y, w, h = cv2.boundingRect(coords)
        img_crop = img[y:y + h, x:x + w]
        img_final = cv2.resize(img_crop, (30, 60))
        templates[char] = img_final

    # ---------------------------------------
    # 🔵 PART 2: 汉字 (从文件夹加载)
    # ---------------------------------------
    template_dir = "templates_cn"  # 对应第一步生成的文件夹名

    # 汉字文件名映射表
    cn_map = {
        'guang': '广',
        'zhou': '州',
        'dong': '东',
        'guan': '莞',
        'fo': '佛',
        'shan': '山'
    }

    if os.path.exists(template_dir):
        # print(">>> 正在加载汉字模板...")
        for filename in os.listdir(template_dir):
            name_no_ext = os.path.splitext(filename)[0]
            if name_no_ext in cn_map:
                char = cn_map[name_no_ext]
                path = os.path.join(template_dir, filename)

                # 读取图片并转为单通道灰度
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

                # 确保是黑底白字 (因为上面的 putText 产生的是黑底白字)
                # 我们的生成脚本已经是黑底白字了，但为了保险起见，加个阈值处理
                if img is not None:
                    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                    img = cv2.resize(img, (30, 60))  # 强制统一尺寸
                    templates[char] = img
    else:
        print("⚠️ 警告：未找到 templates_cn 文件夹，汉字无法识别！")

    print(f">>> 模板库就绪，共 {len(templates)} 个字符")
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
        # 注意：这里我们主要需要 box (四个角的坐标)
        _, _, _, _, _, box, _, _ = best

        # =========================================================
        # 🆕 修改：不再直接切图，而是进行透视矫正
        # =========================================================
        # box 是 float 类型，unwarp_plate 需要 float32
        plate_img = self.unwarp_plate(img_original, box.astype("float32"))

        # 稍微放大一点点(Padding)可能更好，但透视变换后通常不需要太多Padding
        # 如果需要，可以在 resize 之后做

        if self.debug:
            temp = img_original.copy()
            cv2.drawContours(temp, [box], 0, (0, 255, 0), 2)
            cv2.imshow("5. Plate Location (Rotated)", temp)
            cv2.imshow("6. Plate Unwarped", plate_img)  # 看看是不是变直了

        return plate_img

    # ---------------------------------------------------------
    # 🟢 模块 C：字符分割 (完全采用 Step 28 最新逻辑)
    # ---------------------------------------------------------
    def segment_chars(self, plate_img):
        """
        完美缝合版：
        - 上层 (Top): 采用 V1 逻辑 (低门槛、小胶水)，确保抓到汉字。
        - 下层 (Bot): 采用 V5 逻辑 (自适应、去重、投票)，确保数字精准。
        """
        if plate_img is None or plate_img.size == 0: return [], []

        # 1. 预处理 (去边框)
        h, w = plate_img.shape[:2]
        plate_img = plate_img[3:h - 3, 3:w - 3]
        h, w = plate_img.shape[:2]
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        split_y = int(h * 0.42)  # 42% 黄金分割

        # 准备返回容器
        top_imgs_raw = []
        bot_imgs_raw = []

        # =========================================================
        # 🟢 PART 1: 上层处理 (完全回归 gemini1.py 的逻辑)
        # =========================================================
        top_gray = gray[0:split_y, :]

        # 【关键回退1】阈值参数改回 (13, 10)，画面更干净
        top_bin = cv2.adaptiveThreshold(top_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 13, 10)

        # 【关键回退2】胶水改回 (2, 2)，防止汉字粘成一坨
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        top_bin = cv2.morphologyEx(top_bin, cv2.MORPH_OPEN, kernel)
        top_bin = cv2.morphologyEx(top_bin, cv2.MORPH_CLOSE, kernel)

        cnts_top, _ = cv2.findContours(top_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        top_candidates = []

        for cnt in cnts_top:
            x, y, w_c, h_c = cv2.boundingRect(cnt)
            cx = x + w_c / 2

            # 筛选逻辑
            if cx > w * 0.85 or cx < w * 0.05: continue
            if w_c / h_c > 1.3 or w_c / h_c < 0.1: continue

            # 【关键回退3】高度门槛降回 0.2 (20%)，防止漏掉小字
            if h_c < split_y * 0.2: continue

            roi = top_bin[y:y + h_c, x:x + w_c]

            # 【关键回退4】密度容忍度升回 0.95
            if cv2.countNonZero(roi) / (w_c * h_c) > 0.95: continue

            # 注意：gemini1 存的是 top_bin，这对于汉字匹配其实更好
            top_candidates.append((x, y, w_c, h_c, roi))

        # C位排序逻辑 (取离中间最近的2个)
        if len(top_candidates) > 2:
            top_candidates.sort(key=lambda c: abs((c[0] + c[2] / 2) - w / 2))
            top_candidates = top_candidates[:2]
        top_candidates.sort(key=lambda c: c[0])

        for item in top_candidates:
            top_imgs_raw.append(item[4])

        # =========================================================
        # 🔵 PART 2: 下层处理 (保持 gemini5.py 的逻辑，适合数字)
        # =========================================================
        bot_gray = gray[split_y:, :]
        bot_bin = cv2.adaptiveThreshold(bot_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 51, 15)
        cnts_bot, _ = cv2.findContours(bot_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_items = []
        for cnt in cnts_bot:
            x, y, w_c, h_c = cv2.boundingRect(cnt)
            if h_c < (h - split_y) * 0.35: continue  # 数字依然需要严格的高度限制
            if w_c / h_c > 2.0: continue
            raw_items.append({'rect': (x, y, w_c, h_c), 'roi': bot_bin[y:y + h_c, x:x + w_c]})

        bot_items = []
        if raw_items:
            # 民主投票 + 内斗去重 (V5 的精华)
            heights = [item['rect'][3] for item in raw_items]
            heights.sort()
            median_h = heights[len(heights) // 2]

            clean_candidates = []
            for item in raw_items:
                h_curr = item['rect'][3]
                if h_curr > median_h * 1.2: continue
                if h_curr < median_h * 0.7: continue
                clean_candidates.append(item)

            keep_indices = set(range(len(clean_candidates)))
            for i in range(len(clean_candidates)):
                for j in range(len(clean_candidates)):
                    if i == j: continue
                    xi, yi, wi, hi = clean_candidates[i]['rect']
                    xj, yj, wj, hj = clean_candidates[j]['rect']
                    # IOU 计算
                    xx1 = max(xi, xj);
                    yy1 = max(yi, yj)
                    xx2 = min(xi + wi, xj + wj);
                    yy2 = min(yi + hi, yj + hj)
                    w_inter = max(0, xx2 - xx1);
                    h_inter = max(0, yy2 - yy1)
                    if (w_inter * h_inter) > min(wi * hi, wj * hj) * 0.8:
                        if wi * hi > wj * hj:
                            if i in keep_indices: keep_indices.remove(i)
                        else:
                            if j in keep_indices: keep_indices.remove(j)

            for i in keep_indices:
                rect = clean_candidates[i]['rect']
                bot_items.append((rect[0], rect[1], rect[2], rect[3], clean_candidates[i]['roi']))

        bot_items.sort(key=lambda item: item[0])
        for item in bot_items: bot_imgs_raw.append(item[4])

        # =========================================================
        # 🏁 输出与调试
        # =========================================================
        # 统一缩放，注意：这里不需要再做 adaptiveThreshold，因为上面已经是二值图了
        top_resized = [cv2.resize(img, (30, 60)) for img in top_imgs_raw]
        bot_resized = [cv2.resize(img, (30, 60)) for img in bot_imgs_raw]

        if self.debug:
            vis_debug = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.line(vis_debug, (0, split_y), (w, split_y), (0, 255, 255), 2)

            # 画上层框 (红色)
            for item in top_candidates:
                x, y, w_c, h_c, _ = item
                cv2.rectangle(vis_debug, (x, y), (x + w_c, y + h_c), (0, 0, 255), 2)
                cv2.putText(vis_debug, "Top(V1)", (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # 画下层框 (绿色)
            for item in bot_items:
                x, y, w_c, h_c, _ = item
                cv2.rectangle(vis_debug, (x, y + split_y), (x + w_c, y + h_c + split_y), (0, 255, 0), 2)
                cv2.putText(vis_debug, "Bot(V5)", (x, y + split_y + h_c + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 255, 0), 1)

            cv2.imshow("7. Segment Hybrid (V1_Top + V5_Bot)", vis_debug)

        return top_resized, bot_resized

        # ---------------------------------------------------------
        # 🟢 模块 D: 识别 (无预处理 + 带打分显示)
        # ---------------------------------------------------------

    def recognize_smart(self, top_imgs, bot_imgs):
        """
        最终融合版：
        1. 逻辑：回归 V1 (直接缩放 -> 匹配)，不搞复杂的二值化，保护汉字。
        2. 监控：保留了得分显示 (Top 3)，方便你看它到底像谁。
        """

        # === PART 1: 上层汉字 (回归 V1 逻辑 + 恢复打分) ===
        top_candidates = "广洲东莞佛山"
        raw_top_results = []

        print("\n--- 🔍 上层汉字识别详情 ---")

        for idx, img in enumerate(top_imgs):
            if img is None or img.size == 0: continue

            # 【V1 核心】：不做任何二值化/腐蚀/反转，直接暴力拉伸
            # 这样能最大程度保留汉字原本的模样
            if len(img.shape) == 2:
                # 稍微转成二值方便匹配，但阈值设得中庸一点，不要太激进
                _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            else:
                img_bin = img

            img_resize = cv2.resize(img_bin, (30, 60))

            best_score = -1
            best_char = "?"

            # 用来存这一轮所有候选字的得分，方便打印
            debug_scores = []

            for char in top_candidates:
                if char not in self.templates: continue

                # 模板直接用
                score = cv2.matchTemplate(img_resize, self.templates[char], cv2.TM_CCOEFF_NORMED)[0][0]

                # 记录得分
                debug_scores.append((char, score))

                if score > best_score:
                    best_score = score
                    best_char = char

            # 🟢【恢复功能】打印前三名
            debug_scores.sort(key=lambda x: x[1], reverse=True)
            print(
                f"  位置[{idx}] 识别为 '{best_char}' | Top3 候选: {[(x[0], f'{x[1]:.2f}') for x in debug_scores[:3]]}")

            if best_score > 0.4:
                raw_top_results.append(best_char)
            else:
                print(f"  ⚠️ 警告: 最高分仅 {best_score:.2f}，可能不是字")

        # 城市补全逻辑
        top_raw_str = "".join(raw_top_results)
        city_map = {'广': '广州', '佛': '佛山', '山': '佛山', '东': '东莞', '莞': '东莞'}
        final_top = top_raw_str
        for key, full_name in city_map.items():
            if key in top_raw_str:
                final_top = full_name
                break
        if not final_top and raw_top_results: final_top = top_raw_str

        # === PART 2: 下层数字 (V1 逻辑 + V5 修正) ===
        bot_text = ""
        chars_all = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"

        # print("\n--- 🔍 下层数字识别详情 ---") # 如果觉得太吵可以注释掉

        for i, img in enumerate(bot_imgs):
            img_resize = cv2.resize(img, (30, 60))

            best_char = "?"
            best_score = -1
            # debug_scores_bot = [] # 下层一般不需要看那么细，只要看最终结果

            for char in chars_all:
                if char not in self.templates: continue
                score = cv2.matchTemplate(img_resize, self.templates[char], cv2.TM_CCOEFF_NORMED)[0][0]
                # debug_scores_bot.append((char, score))
                if score > best_score:
                    best_score = score
                    best_char = char

            # 【V5 核心】：只有遇到这几个倒霉蛋时，才启动“数洞洞”逻辑
            original_char = best_char
            if best_char in ['6', '8', 'B', 'G']:
                best_char = self.refine_char(img_resize, best_char)

            # V1 的经验规则
            if best_char == 'D' and i > 1: best_char = '0'

            # 如果发生了修正，打印出来看看
            if original_char != best_char:
                print(f"  位置[{i}] 修正: '{original_char}' -> '{best_char}' (通过拓扑分析)")

            bot_text += best_char

        return final_top + bot_text



    def refine_char(self, char_img, candidate):
        """
        【万能修正器】区分 6, 8, G, B
        依据：拓扑学（洞的数量）+ 像素密度
        """
        # 1. 统一尺寸 & 二值化
        h, w = 60, 30
        img = cv2.resize(char_img, (w, h))
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Otsu 阈值
        _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 2. 自动纠正底色 (确保是黑底白字)
        border_pixels = np.concatenate([
            img_bin[0, :], img_bin[-1, :], img_bin[:, 0], img_bin[:, -1]
        ])
        if cv2.countNonZero(border_pixels) > border_pixels.size / 2:
            img_bin = cv2.bitwise_not(img_bin)

        # 3. 数洞洞 (Eular Number)
        contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        hole_count = 0
        if hierarchy is not None:
            for i in range(len(contours)):
                # Parent != -1 表示它是内部轮廓（洞）
                if hierarchy[0][i][3] != -1:
                    area = cv2.contourArea(contours[i])
                    if area > 10:  # 忽略噪点
                        hole_count += 1

        print(f"DEBUG Fix: 原字符='{candidate}' | 洞数量={hole_count}")

        # ================== 判决逻辑 ==================

        # 情况 A: 肯定是 8 (或者 B)
        if hole_count >= 2:
            if candidate == 'B': return 'B'  # 如果本来就是B，尊重原判
            return '8'  # 否则 6/G/8 统统变成 8

        # 情况 B: 肯定是 6 (G通常是开口的，没有洞)
        # 如果识别成 G 或 8，但发现只有1个洞，那一定是 6
        if hole_count == 1:
            return '6'

            # 情况 C: 没有洞 (0个)
        # 可能是 G，也可能是“断头 6”或者“实心 8”
        if hole_count == 0:
            if candidate == 'G':
                return 'G'  # 确实是G
            if candidate == '6':
                return '6'  # 可能是断裂的6，保持原判

            # 如果是 8 但没洞，检查“脖子” (之前的逻辑)
            # 8 的脖子(右上侧)是连通的(有像素)，6/G 是空的
            roi_neck = img_bin[12:22, 18:30]
            if cv2.countNonZero(roi_neck) / roi_neck.size > 0.45:
                return '8'
            else:
                return '6'

        return candidate

    # ---------------------------------------------------------
    # 🆕 新增模块：车牌矫正 (透视变换)
    # ---------------------------------------------------------
    def order_points(self, pts):
        """
        对四个角点进行排序：左上 -> 右上 -> 右下 -> 左下
        防止找错角导致车牌镜像或翻转
        """
        rect = np.zeros((4, 2), dtype="float32")

        # 左上角(sum最小) 和 右下角(sum最大)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # 右上角(diff最小) 和 左下角(diff最大)
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def unwarp_plate(self, img, box):
        """
        核心矫正逻辑：把歪的 box 抠出来并拉直
        """
        rect = self.order_points(box)
        (tl, tr, br, bl) = rect

        # 计算新图片的宽度 (取上下两条边的最大值)
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # 计算新图片的高度 (取左右两条边的最大值)
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # 目标点：标准的正长方形
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        # 计算透视变换矩阵并应用
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

        # 🛡️ 额外保护：如果是竖起来的长条(比如旋转了90度)，强制转回来
        h, w = warped.shape[:2]
        if h > w:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

        return warped





# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    img_path = "images/slant/5.jpg"  # 换成你的测试图片

    if os.path.exists(img_path):
        print(f"处理图片: {img_path}")
        recognizer = LicensePlateRecognizer()

        # 1. 预处理
        resized, morph, mask = recognizer.preprocess_enhanced(cv2.imread(img_path))

        # 2. 定位
        plate = recognizer.locate_plate_dual_strategy(resized, morph, mask)

        if plate is not None:
            # 3. 分割 (⚠️注意这里接收两个变量)
            top_imgs, bot_imgs = recognizer.segment_chars(plate)

            if top_imgs or bot_imgs:
                # 4. 识别 (⚠️调用新的 smart 函数)
                text = recognizer.recognize_smart(top_imgs, bot_imgs)
                print(f"\n✅ 最终结果: {text}")
            else:
                print("❌ 分割失败")
        else:
            print("❌ 定位失败")

        cv2.waitKey(0)
        cv2.destroyAllWindows()