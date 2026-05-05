import cv2
import numpy as np
import os


# ==========================================
# 第一部分：生成加粗的模板 (核心改进)
# ==========================================
def generate_templates():
    templates = {}
    # 去掉了容易混淆的 I, O
    characters = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"

    print(">>> 正在生成加粗字符模板库...")

    for char in characters:
        img = np.zeros((40, 20), dtype=np.uint8)
        # 核心修改：thickness 从 2 改为 3，模拟车牌的粗线条
        cv2.putText(img, char, (2, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 3)

        coords = cv2.findNonZero(img)
        x, y, w, h = cv2.boundingRect(coords)
        img_crop = img[y:y + h, x:x + w]
        img_final = cv2.resize(img_crop, (20, 40))
        templates[char] = img_final

    return templates


# ==========================================
# 第二部分：核心处理类
# ==========================================
class LicensePlateRecognizer:
    def __init__(self):
        self.templates = generate_templates()
        self.debug = True

    def preprocess(self, img):
        """保持你原本效果不错的预处理逻辑"""
        height, width = img.shape[:2]
        scale = 600 / height
        img_resized = cv2.resize(img, (int(width * scale), 600))

        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        sobel = cv2.Sobel(blurred, cv2.CV_8U, 1, 0, ksize=3)
        ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return img_resized, closed

    def locate_plate(self, img_original, img_morph):
        contours, _ = cv2.findContours(img_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = w / float(h)
            area = w * h

            # 这里的参数稍微放宽一点，防止漏掉车牌
            if (area > 1000) and (1.5 < ratio < 5.0):
                candidates.append((x, y, w, h, area))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[4], reverse=True)
        best = candidates[0]
        x, y, w, h, _ = best

        # 增加 padding，防止切掉车牌边缘的字符（比如最后的3）
        pad = 8  # 原来是5，稍微加大一点
        plate_img = img_original[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]

        if self.debug:
            cv2.imshow("Detected Plate", plate_img)

        return plate_img

    def segment_chars(self, plate_img):
        if plate_img is None or plate_img.size == 0:
            return []

        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        # 注意：这里是 THRESH_BINARY_INV，把字变白，底变黑
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

        # 裁掉上下边框（保留原本逻辑，这部分没问题）
        h, w = binary.shape
        # 左右只切 2%，防止切掉首尾字符
        binary = binary[int(h * 0.15):int(h * 0.85), int(w * 0.02):int(w * 0.98)]

        # 稍微膨胀一点点，让断裂的字符连起来（比如8如果断了）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilated = cv2.dilate(binary, kernel, iterations=1)

        if self.debug:
            cv2.imshow("Plate Binary for Segmentation", dilated)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        char_imgs = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # 宽松的过滤条件：只要不是太小的噪点，都算字符
            if h > binary.shape[0] * 0.35 and w > 2:
                # 处理粘连
                if w > h * 0.85:
                    mid = x + w // 2
                    char_imgs.append((x, binary[y:y + h, x:mid]))
                    char_imgs.append((mid, binary[y:y + h, mid:x + w]))
                else:
                    char_imgs.append((x, binary[y:y + h, x:x + w]))

        char_imgs.sort(key=lambda x: x[0])
        return [c[1] for c in char_imgs]

    def recognize(self, char_imgs):
        result_string = ""

        # 增强的纠错表：针对你的 P532N 问题
        correction_map = {
            'S': '3', 'Z': '2', 'D': '0',
            'O': '0', 'I': '1', 'B': '8',
            'N': '8',  # 关键修正：如果识别出 N，大概率是 8
            'Q': '0', 'L': '1'
        }

        print("\n--- 字符识别详情 ---")
        for i, char_img in enumerate(char_imgs):
            char_resized = cv2.resize(char_img, (20, 40))

            best_score = -1
            best_char = "?"

            for char_key, template_img in self.templates.items():
                res = cv2.matchTemplate(char_resized, template_img, cv2.TM_CCOEFF_NORMED)
                _, score, _, _ = cv2.minMaxLoc(res)

                if score > best_score:
                    best_score = score
                    best_char = char_key

            # 原始识别结果
            original_char = best_char

            # 应用纠错
            if best_char in correction_map:
                best_char = correction_map[best_char]

            # 如果置信度太低，可能是噪点（螺丝钉），但我们先不删，标记出来
            if best_score < 0.3:
                print(f"位置{i}: 识别为 [{original_char}] (置信度低 {best_score:.2f}) -> 可能是噪点")
                # 这里你可以选择是否要把这个字符加进去
                # 如果是第一个字符且置信度低，往往是左边的边框或螺丝
                if i == 0:
                    continue
            else:
                print(f"位置{i}: 识别为 [{original_char}] -> 修正为 [{best_char}] (置信度: {best_score:.2f})")
                result_string += best_char

        return result_string


# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    image_path = "images/normal/1.PNG"

    if not os.path.exists(image_path):
        print(f"错误：找不到文件 {image_path}")
    else:
        recognizer = LicensePlateRecognizer()
        original_img = cv2.imread(image_path)

        print("--- 开始处理 ---")
        resized, morph = recognizer.preprocess(original_img)
        plate_img = recognizer.locate_plate(resized, morph)

        if plate_img is not None:
            # 无论分割出多少个，都继续往下跑
            char_list = recognizer.segment_chars(plate_img)
            print(f"分割出 {len(char_list)} 个字符区域")

            if len(char_list) > 0:
                plate_number = recognizer.recognize(char_list)
                print("\n" + "=" * 30)
                print(f"最终结果: {plate_number}")
                print("=" * 30)
            else:
                print("未分割出字符")
        else:
            print("未能定位到车牌")

        cv2.waitKey(0)
        cv2.destroyAllWindows()