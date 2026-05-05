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
        """Step 9: 增加 C位优先逻辑，强力去除 Plan B 的边缘噪点"""
        if plate_img is None or plate_img.size == 0: return []
        h_plate, w_plate = plate_img.shape[:2]

        # --- 内部函数 1: 找轮廓 (通用) ---
        def find_candidates(binary_img):
            h_bin, w_bin = binary_img.shape
            contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            raw_candidates = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = cv2.contourArea(cnt)

                # 1. 基础尺寸过滤
                if area < 15: continue
                if h < h_bin * 0.08: continue
                if h > h_bin * 0.95: continue  # 稍微放宽一点上限

                # 2. 比例过滤
                ratio = w / h
                if ratio > 1.3: continue  # 太扁的肯定是噪点
                if ratio < 0.1: continue  # 太细的线

                # 3. 像素密度过滤 (新!)
                # 字通常是“空心”的，如果一个块全是白的(>90%)，通常是反光块
                roi = binary_img[y:y + h, x:x + w]
                pixel_density = cv2.countNonZero(roi) / (w * h)
                if pixel_density > 0.95: continue

                raw_candidates.append((x, y, w, h, binary_img[y:y + h, x:x + w]))
            return raw_candidates

        # --- 内部函数 2: 分行与清洗 (核心改进!) ---
        def sort_and_clean(candidates, img_h, img_w):
            if not candidates: return []
            y_split = img_h * 0.35
            top = []
            bottom = []

            for item in candidates:
                x, y, w, h, img = item
                cy = y + h / 2
                cx = x + w / 2

                if cy < y_split:
                    # === 上层清洗逻辑 ===
                    # 1. 禁区过滤：电动车牌右上角(>75%)全是二维码，直接杀
                    if cx > img_w * 0.80: continue
                    # 2. 左侧边缘过滤：左边太靠边(<5%)的通常是边框
                    if cx < img_w * 0.05: continue

                    top.append(item)
                else:
                    # === 下层清洗逻辑 ===
                    # 数字层比较简单，主要是去掉太胖的螺丝
                    ratio = w / h
                    if ratio < 0.85:  # 严格一点，数字通常很瘦
                        bottom.append(item)

            # === 核心算法：Top层“C位优先” ===
            # 上层通常只有2个汉字("广州"、"佛山")。
            # 如果 Plan B 抓到了 5 个东西（含噪点），我们只取离中心最近的 2 个！
            if len(top) > 2:
                center_x = img_w / 2
                # 按“距离图片中心的距离”排序
                top.sort(key=lambda c: abs((c[0] + c[2] / 2) - center_x))
                # 只保留最近的2个（也就是正中间的汉字）
                top = top[:2]

            # 重新按从左到右排序
            top.sort(key=lambda c: c[0])
            bottom.sort(key=lambda c: c[0])

            return [c[4] for c in top + bottom]

        # === 主流程 ===
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        # 1. Plan A: OTSU (稳健模式)
        _, bin_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
        # 稍微连接一下
        bin_otsu = cv2.morphologyEx(bin_otsu, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
        bin_otsu = bin_otsu[2:h_plate - 2, 2:w_plate - 2]  # 切边

        candidates_1 = find_candidates(bin_otsu)
        final_1 = sort_and_clean(candidates_1, h_plate - 4, w_plate - 4)

        # 决策：如果 OTSU 找到了 >=6 个字符，且上层正好有 2 个，直接通过
        # 增加一个条件：上层字符不能为0 (防止只识别出下层)
        chars_top_1 = len([x for x in candidates_1 if (x[1] + x[3] / 2) < (h_plate * 0.35)])

        if len(final_1) >= 6 and chars_top_1 >= 1:
            final_chars = final_1
            debug_bin = bin_otsu
            strategy = "Plan A (OTSU)"
        else:
            # 2. Plan B: Adaptive (强力模式)
            # 使用较小的 block size (13) 来切断噪点连接
            bin_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY_INV, 13, 10)
            # 增加一步“开运算”：先腐蚀再膨胀，去掉细小的噪点点
            kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            bin_adapt = cv2.morphologyEx(bin_adapt, cv2.MORPH_OPEN, kernel_clean)
            # 再做闭运算连接汉字
            bin_adapt = cv2.morphologyEx(bin_adapt, cv2.MORPH_CLOSE, kernel_clean)

            bin_adapt = bin_adapt[2:h_plate - 2, 2:w_plate - 2]

            candidates_2 = find_candidates(bin_adapt)
            final_2 = sort_and_clean(candidates_2, h_plate - 4, w_plate - 4)

            final_chars = final_2
            debug_bin = bin_adapt
            strategy = "Plan B (Adaptive)"

        # === 统一显示 ===
        if self.debug:
            print(f"✅ 策略: {strategy} | 字符: {len(final_chars)}")
            if debug_bin is not None:
                cv2.imshow("7. Binary Used", debug_bin)

            if final_chars:
                # 统一高度显示，方便观察
                display_imgs = [cv2.resize(img, (30, 60)) for img in final_chars]
                display_block = np.hstack(display_imgs)
                cv2.imshow("8. Segmented Result", display_block)
                cv2.waitKey(0)  # 加上这个，防止窗口一闪而过

        return [cv2.resize(img, (30, 60)) for img in final_chars]


# 主程序（关键：加cv2.waitKey()让窗口停留）
if __name__ == "__main__":
    # 替换成你的图片路径
    img_path = "images/normal/3.jpg"
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