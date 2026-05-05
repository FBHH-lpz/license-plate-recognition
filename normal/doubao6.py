import cv2
import numpy as np
import os


# 第一部分：字符模板（保留）
def generate_templates():
    templates = {}
    characters = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ广洲"  # 新增“广、洲”适配广州车牌
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


# 第二部分：核心类（修复第六步后无后续的问题）
class LicensePlateRecognizer:
    def __init__(self):
        self.templates = generate_templates()
        self.debug = True

    def preprocess_enhanced(self, img):
        """保留预处理逻辑，确保返回有效数据"""
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

        return img_resized, morph_final, mask_white_clean

    def locate_plate_dual_strategy(self, img_original, img_morph, mask_white):
        """修复定位逻辑，确保返回有效车牌区域"""
        contours, _ = cv2.findContours(img_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_contours = img_original.copy()
        cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
        if self.debug:
            cv2.imshow("4. All Contours", img_contours)

        raw_candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300:
                continue
            rect = cv2.minAreaRect(cnt)
            w, h = rect[1]
            if w < h:
                w, h = h, w
            ratio = w / h
            if 1.5 < ratio < 6.0:
                x, y, w_rect, h_rect = cv2.boundingRect(cnt)
                raw_candidates.append((x, y, w_rect, h_rect, area, rect[0]))

        # 兜底：如果没找到候选框，直接用白色掩码的最大连通域
        if not raw_candidates:
            contours_white, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours_white:
                cnt_max = max(contours_white, key=cv2.contourArea)
                x, y, w_rect, h_rect = cv2.boundingRect(cnt_max)
                raw_candidates.append(
                    (x, y, w_rect, h_rect, cv2.contourArea(cnt_max), (x + w_rect / 2, y + h_rect / 2)))

        if not raw_candidates:
            print("❌ 未找到任何车牌候选框")
            return None

        # 选最优候选框
        raw_candidates.sort(key=lambda x: x[4], reverse=True)
        best = raw_candidates[0]
        x, y, w, h, _, _ = best

        # 确保框不越界
        x_start = max(0, x - int(h * 0.2))
        y_start = max(0, y - int(h * 0.2))
        x_end = min(img_original.shape[1], x + w + int(h * 0.2))
        y_end = min(img_original.shape[0], y + h + int(h * 0.2))

        plate_img = img_original[y_start:y_end, x_start:x_end]
        # 检查车牌区域是否有效（尺寸≥50x20）
        if plate_img.shape[0] < 20 or plate_img.shape[1] < 50:
            print("❌ 提取的车牌区域尺寸过小")
            return None

        if self.debug:
            img_plate_box = img_original.copy()
            cv2.rectangle(img_plate_box, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
            cv2.imshow("5. Final Plate Box (Fixed)", img_plate_box)
            cv2.imshow("6. Extracted Plate", plate_img)  # 第六步窗口

        return plate_img

    def segment_chars(self, plate_img):
        """Step 28: 最终融合版 (Step 9 上层逻辑 + Step 21 下层逻辑 + 42% 分割线)"""
        if plate_img is None or plate_img.size == 0: return []

        # 1. 预处理
        h, w = plate_img.shape[:2]
        plate_img = plate_img[3:h - 3, 3:w - 3]
        h, w = plate_img.shape[:2]
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        # 【修正】分割线调整为 42%，给上层汉字更多空间
        split_y = int(h * 0.42)
        final_chars = []

        # ==========================================================
        # 🟢 PART 1: 上层处理 (Step 9 逻辑：C位优先 + 强力清洗)
        # ==========================================================
        top_gray = gray[0:split_y, :]
        top_h, top_w = top_gray.shape[:2]

        # 1. Step 9 的 Plan B 参数 (Block=13, C=10)
        top_bin = cv2.adaptiveThreshold(top_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 13, 10)

        # Step 9 的形态学操作：开运算去噪 -> 闭运算连接
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        top_bin = cv2.morphologyEx(top_bin, cv2.MORPH_OPEN, kernel)
        top_bin = cv2.morphologyEx(top_bin, cv2.MORPH_CLOSE, kernel)

        cnts_top, _ = cv2.findContours(top_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        top_candidates = []
        for cnt in cnts_top:
            x, y, w_c, h_c = cv2.boundingRect(cnt)
            cx = x + w_c / 2

            # --- Step 9 的严格过滤 ---
            # 1. 禁区过滤：杀掉最右边(二维码)和最左边(边框)
            if cx > top_w * 0.80: continue
            if cx < top_w * 0.05: continue

            # 2. 比例过滤
            ratio = w_c / h_c
            if ratio > 1.3: continue  # 太扁的
            if ratio < 0.1: continue  # 太细的

            # 3. 尺寸过滤 (相对于新的 top_h)
            if h_c < top_h * 0.2: continue

            # 4. 像素密度过滤
            roi = top_bin[y:y + h_c, x:x + w_c]
            pixel_density = cv2.countNonZero(roi) / (w_c * h_c)
            if pixel_density > 0.95: continue

            top_candidates.append((x, y, w_c, h_c, top_bin[y:y + h_c, x:x + w_c]))

        # --- Step 9 核心：C位优先 ---
        if len(top_candidates) > 2:
            center_x = top_w / 2
            # 离中心越近越优先 (杀掉边缘螺丝)
            top_candidates.sort(key=lambda c: abs((c[0] + c[2] / 2) - center_x))
            top_candidates = top_candidates[:2]

        # 选完后按顺序排好
        top_candidates.sort(key=lambda c: c[0])
        for item in top_candidates:
            final_chars.append(item[4])

        # ==========================================================
        # 🔵 PART 2: 下层处理 (Step 21 逻辑：保持不变)
        # ==========================================================
        bot_gray = gray[split_y:, :]
        bot_h = bot_gray.shape[0]

        # Step 21 参数 (Block=51, C=15)
        bot_bin = cv2.adaptiveThreshold(bot_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 51, 15)

        cnts_bot, _ = cv2.findContours(bot_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bot_items = []
        for cnt in cnts_bot:
            x, y, w_c, h_c = cv2.boundingRect(cnt)

            # 尺寸过滤 (适配新的 bot_h)
            if h_c < bot_h * 0.35: continue
            if w_c / h_c > 2.0: continue

            bot_items.append((x, y, w_c, h_c, bot_bin[y:y + h_c, x:x + w_c]))

        bot_items.sort(key=lambda item: item[0])
        for item in bot_items:
            final_chars.append(item[4])

        # ==========================================================
        # 🏁 统一输出
        # ==========================================================
        resized_chars = []
        for img in final_chars:
            if img.shape[0] == 0 or img.shape[1] == 0: continue
            resized_chars.append(cv2.resize(img, (30, 60)))

        # 调试显示
        if self.debug:
            print(f"✅ Step 28 最终版(42%分割) | Top: {len(top_candidates)}, Bot: {len(bot_items)}")
            vis_top = cv2.cvtColor(top_bin, cv2.COLOR_GRAY2BGR)
            vis_bot = cv2.cvtColor(bot_bin, cv2.COLOR_GRAY2BGR)

            # 画框框
            for (x, y, w_c, h_c, _) in top_candidates:
                cv2.rectangle(vis_top, (x, y), (x + w_c, y + h_c), (0, 0, 255), 2)
            for (x, y, w_c, h_c, _) in bot_items:
                cv2.rectangle(vis_bot, (x, y), (x + w_c, y + h_c), (0, 255, 0), 2)

            cv2.imshow("7. Top (Step9 + 42%)", vis_top)
            cv2.imshow("8. Bot (Step21)", vis_bot)
            cv2.waitKey(0)

        return resized_chars


# 主程序（关键：加cv2.waitKey()让窗口停留）
if __name__ == "__main__":
    # 替换成你的图片路径
    img_path = "images/normal/5.jpg"
    if not os.path.exists(img_path):
        print(f"❌ 图片路径不存在：{img_path}")
    else:
        original_img = cv2.imread(img_path)
        recognizer = LicensePlateRecognizer()

        # 第一步：预处理
        resized, morph, mask_white = recognizer.preprocess_enhanced(original_img)
        # 第二步：定位车牌（第六步窗口在这里触发）
        plate_img = recognizer.locate_plate_dual_strategy(resized, morph, mask_white)
        # 第三步：分割字符（第七、八步窗口在这里触发）
        chars = recognizer.segment_chars(plate_img)

        # 关键：加waitKey，让所有窗口停留（按任意键关闭）
        print("✅ 执行完成，所有窗口已显示（按任意键关闭）")
        cv2.waitKey(0)
        cv2.destroyAllWindows()