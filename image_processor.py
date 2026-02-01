import json
import os
import time
import numpy as np
import cv2
import pyautogui
import pygetwindow as gw
from tkinter import simpledialog

# -------------------------- 全局配置 --------------------------
CONFIG_DIR = "color_configs"  # 配置文件目录
DEFAULT_CONFIG = "default_config.json"  # 默认配置文件名
DEBUG_DIR = "debug"
SAMPLE_REGION_SIZE = 11
OUTLIER_STD_MULT = 1.5
BOARD_RESIZE = 450  # 从600减小到450以提高处理速度
SAMPLE_COUNT = 3
BOARD_HSV_LOWER = np.array([10, 0, 0])
BOARD_HSV_UPPER = np.array([40, 255, 255])
EMPTY_REGION_STD_THRESHOLD = 15  # 空白区域标准差阈值


# -------------------------- 颜色样本类 --------------------------
class ColorSample:
    def __init__(self):
        self.board_color_bgr = None
        self.black_color_bgr = None
        self.white_samples_bgr = []
        self.black_range_bgr = None
        self.white_saturation_threshold = 40
        self.white_value_threshold = 200

    def set_board_color(self, color_bgr):
        self.board_color_bgr = np.array(color_bgr, dtype=np.uint8)

    def set_black_color(self, color_bgr):
        self.black_color_bgr = np.array(color_bgr, dtype=np.uint8)
        self.black_range_bgr = self._calculate_color_range(color_bgr, tolerance=40)

    def add_white_sample(self, color_bgr):
        self.white_samples_bgr.append(np.array(color_bgr, dtype=np.uint8))

    def finalize_white_model(self):
        if not self.white_samples_bgr:
            return

        white_hsv = [cv2.cvtColor(np.uint8([[s]]), cv2.COLOR_BGR2HSV)[0][0] for s in self.white_samples_bgr]
        white_hsv = np.array(white_hsv)
        avg_s = np.mean(white_hsv[:, 1])
        avg_v = np.mean(white_hsv[:, 2])
        self.white_saturation_threshold = min(avg_s + 30, 255)
        self.white_value_threshold = max(avg_v - 40, 0)

    def _calculate_color_range(self, color, tolerance=20):
        color = np.array(color)
        lower = np.maximum(color - tolerance, 0)
        upper = np.minimum(color + tolerance, 255)
        return (lower, upper)

    def _is_color_in_range(self, color, color_range):
        lower, upper = color_range
        return np.all(color >= lower) and np.all(color <= upper)

    def save_config(self, file_path):
        config = {
            "board_color_bgr": self.board_color_bgr.tolist(),
            "black_color_bgr": self.black_color_bgr.tolist(),
            "black_range_bgr_lower": self.black_range_bgr[0].tolist(),
            "black_range_bgr_upper": self.black_range_bgr[1].tolist(),
            "white_saturation_threshold": int(self.white_saturation_threshold),
            "white_value_threshold": int(self.white_value_threshold)
        }
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=4)

    @classmethod
    def load_config(cls, file_path):
        if not os.path.exists(file_path):
            return None
        with open(file_path, 'r') as f:
            config = json.load(f)

        cs = cls()
        cs.board_color_bgr = np.array(config["board_color_bgr"], dtype=np.uint8)
        cs.black_color_bgr = np.array(config["black_color_bgr"], dtype=np.uint8)
        cs.black_range_bgr = (
            np.array(config["black_range_bgr_lower"], dtype=np.uint8),
            np.array(config["black_range_bgr_upper"], dtype=np.uint8)
        )
        cs.white_saturation_threshold = config["white_saturation_threshold"]
        cs.white_value_threshold = config["white_value_threshold"]
        return cs


class ImageProcessor:
    def __init__(self):
        self.color_sample = ColorSample()
        self.output_dir = "screenshots"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(DEBUG_DIR, exist_ok=True)
        os.makedirs(CONFIG_DIR, exist_ok=True)  # 创建配置文件目录
        self.selected_window = None  # 保存用户选择的窗口

    def set_selected_window(self, window_title):
        """设置用户选择的窗口"""
        all_windows = gw.getAllWindows()
        for window in all_windows:
            if window.title == window_title:
                self.selected_window = window
                return True
        return False

    def get_all_window_titles(self):
        """获取所有窗口标题"""
        all_windows = gw.getAllWindows()
        titles = []
        for window in all_windows:
            if window.title.strip():  # 只添加非空标题
                titles.append(window.title)
        return titles

    def find_game_window(self):
        """查找游戏窗口"""
        # 如果用户已经选择了窗口，直接使用
        if self.selected_window:
            return self.selected_window

        # 否则使用自动查找逻辑
        all_windows = gw.getAllWindows()

        target_titles = ['天天象棋']
        target_window = None

        for window in all_windows:
            title = window.title
            for target in target_titles:
                if target in title:
                    print(f"找到游戏窗口: {title}")
                    target_window = window
                    break
            if target_window:
                break

        if not target_window:
            print("未找到特定游戏窗口，使用活动窗口")
            try:
                target_window = gw.getActiveWindow()
                print(f"使用活动窗口: {target_window.title}")
            except:
                print("无法获取活动窗口")
                return None

        return target_window

    def screenshot_game(self):
        """截取游戏画面"""
        target_window = self.find_game_window()

        if not target_window:
            print("无法找到游戏窗口")
            return None

        try:
            target_window.activate()
            time.sleep(0.5)

            screenshot = pyautogui.screenshot(region=(
                target_window.left,
                target_window.top,
                target_window.width,
                target_window.height
            ))

            image = np.array(screenshot)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            # filename = os.path.join(self.output_dir, f"game_screenshot_{timestamp}.png")
            filename = os.path.join(self.output_dir, f"game_screenshot.png")

            cv2.imwrite(filename, image)
            print(f"截图已保存: {filename}")

            return image

        except Exception as e:
            print(f"截图失败: {e}")
            return None

    def get_avg_color_without_outliers(self, roi):
        """计算区域的平均颜色（剔除异常值）"""
        if roi.size == 0:
            return np.array([0, 0, 0], dtype=np.uint8)

        valid_pixels = roi.reshape(-1, 3)
        for channel in range(3):
            if valid_pixels.size == 0:
                break
            channel_vals = valid_pixels[:, channel]
            mean_val = np.mean(channel_vals)
            std_val = np.std(channel_vals)
            mask = (channel_vals >= mean_val - OUTLIER_STD_MULT * std_val) & \
                   (channel_vals <= mean_val + OUTLIER_STD_MULT * std_val)
            valid_pixels = valid_pixels[mask]

        if valid_pixels.size == 0:
            avg_color = np.mean(roi.reshape(-1, 3), axis=0).astype(np.uint8)
        else:
            avg_color = np.mean(valid_pixels, axis=0).astype(np.uint8)

        return avg_color

    def locate_chessboard(self, img):
        """定位棋盘区域"""
        img_copy = img.copy()
        h, w = img.shape[:2]
        print(f"原始图像尺寸：{w}x{h}")

        # 第一次定位
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, BOARD_HSV_LOWER, BOARD_HSV_UPPER)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        color_mask = cv2.erode(color_mask, kernel_small, iterations=1)
        color_mask = cv2.dilate(color_mask, kernel_small, iterations=1)
        
        # 保存第一次处理的掩码图像用于调试
        cv2.imwrite(os.path.join(DEBUG_DIR, "debug_first_mask.jpg"), color_mask)

        masked_img = cv2.bitwise_and(img, img, mask=color_mask)
        gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)
        
        # 保存边缘检测图像用于调试
        cv2.imwrite(os.path.join(DEBUG_DIR, "debug_edges.jpg"), edges)
        cv2.imwrite(os.path.join(DEBUG_DIR, "debug_closed_edges.jpg"), closed_edges)

        contours, _ = cv2.findContours(closed_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("警告：未检测到任何轮廓，返回原图")
            return img, 0, 0  # 返回图像和位置(0, 0)
        first_contour = max(contours, key=cv2.contourArea)
        x1, y1, w1, h1 = cv2.boundingRect(first_contour)
        first_crop = img_copy[y1:y1 + h1, x1:x1 + w1]
        if first_crop.size == 0:
            print("警告：第一次裁剪无效，返回原图")
            return img, 0, 0  # 返回图像和位置(0, 0)
        print(f"第一次裁剪：尺寸 {w1}x{h1}，位置 ({x1}, {y1})")

        # 二次提纯
        first_crop_hsv = cv2.cvtColor(first_crop, cv2.COLOR_BGR2HSV)
        secondary_hsv_lower = np.array([max(BOARD_HSV_LOWER[0] - 5, 0), 10, 10])
        secondary_hsv_upper = np.array([min(BOARD_HSV_UPPER[0] + 5, 179), 245, 245])
        secondary_mask = cv2.inRange(first_crop_hsv, secondary_hsv_lower, secondary_hsv_upper)

        kernel_secondary = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        secondary_mask = cv2.dilate(secondary_mask, kernel_secondary, iterations=1)
        secondary_mask = cv2.erode(secondary_mask, kernel_secondary, iterations=1)

        cv2.imwrite(os.path.join(DEBUG_DIR, "debug_secondary_mask.jpg"), secondary_mask)

        secondary_contours, _ = cv2.findContours(secondary_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not secondary_contours:
            print("警告：二次筛选无轮廓，回退到第一次裁剪结果")
            final_crop = first_crop
            crop_x, crop_y = x1, y1  # 位置信息
        else:
            secondary_contour = max(secondary_contours, key=cv2.contourArea)
            x2, y2, w2, h2 = cv2.boundingRect(secondary_contour)
            final_crop = first_crop[y2:y2 + h2, x2:x2 + w2]
            crop_x, crop_y = x1 + x2, y1 + y2  # 相对于原始图像的位置
            if final_crop.size == 0:
                print("警告：二次裁剪无效，回退到第一次裁剪结果")
                final_crop = first_crop
                crop_x, crop_y = x1, y1
            else:
                print(f"二次裁剪：尺寸 {w2}x{h2}（已剔除背景），位置 ({crop_x}, {crop_y}）")

        if final_crop.shape[0] < 100 or final_crop.shape[1] < 100:
            print(f"警告：最终裁剪尺寸过小（{final_crop.shape[1]}x{final_crop.shape[0]}），回退到第一次裁剪")
            final_crop = first_crop
            crop_x, crop_y = x1, y1

        cv2.imwrite(os.path.join(DEBUG_DIR, "cropped_chessboard.jpg"), final_crop)
        print(f"棋盘定位完成：最终尺寸 {final_crop.shape[1]}x{final_crop.shape[0]}，位置 ({crop_x}, {crop_y})")
        return final_crop, crop_x, crop_y

    def detect_pieces(self, chessboard):
        """识别棋子 - 优化版本，只分析可能有棋子的区域"""
        # 检查输入参数是否有效
        if chessboard is None or chessboard.size == 0:
            print("错误：输入的棋盘图像无效")
            return None
            
        # 确保输入是有效的numpy数组
        if not isinstance(chessboard, np.ndarray):
            print("错误：棋盘图像不是有效的numpy数组")
            return None

        try:
            img_resized = cv2.resize(chessboard, (BOARD_RESIZE, BOARD_RESIZE), interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(f"图像缩放失败：{str(e)}")
            return None
            
        grid_size = BOARD_RESIZE // 15
        piece_box_size = int(grid_size * 0.8)  # 根据网格大小动态调整
        box_offset = (grid_size - piece_box_size) // 2

        board_state = [[0 for _ in range(15)] for _ in range(15)]
        
        # 创建用于可视化棋子识别的图像
        debug_img = img_resized.copy()

        for col in range(15):
            for row in range(15):
                # 定义ROI区域
                box_x_start = col * grid_size + box_offset
                box_x_end = box_x_start + piece_box_size
                box_y_start = row * grid_size + box_offset
                box_y_end = box_y_start + piece_box_size
                piece_roi = img_resized[box_y_start:box_y_end, box_x_start:box_x_end]

                # 快速判断是否为空白区域
                # 转换为灰度图并计算标准差
                roi_gray = cv2.cvtColor(piece_roi, cv2.COLOR_BGR2GRAY)
                roi_std = np.std(roi_gray)

                # 如果标准差很小，说明是相对均匀的区域，很可能是空白交叉点
                if roi_std < EMPTY_REGION_STD_THRESHOLD:  # 使用全局配置的阈值
                    board_state[row][col] = 0
                    continue

                # 对于标准差较大的区域，进行详细分析
                avg_color = self.get_avg_color_without_outliers(piece_roi)

                piece_type = 0
                # 黑棋检测
                if self.color_sample.black_range_bgr and self.color_sample._is_color_in_range(avg_color,
                                                                                              self.color_sample.black_range_bgr):
                    piece_type = 2

                # 白棋检测
                elif piece_type == 0:
                    hsv = cv2.cvtColor(np.uint8([[avg_color]]), cv2.COLOR_BGR2HSV)[0][0]
                    if hsv[1] < self.color_sample.white_saturation_threshold and hsv[
                        2] > self.color_sample.white_value_threshold:
                        piece_type = 1

                board_state[row][col] = piece_type
                
                # 在调试图像上绘制识别结果
                center_x = (box_x_start + box_x_end) // 2
                center_y = (box_y_start + box_y_end) // 2
                if piece_type == 1:  # 白棋
                    cv2.circle(debug_img, (center_x, center_y), 8, (0, 0, 255), 2)  # 红色圆圈
                elif piece_type == 2:  # 黑棋
                    cv2.circle(debug_img, (center_x, center_y), 8, (255, 0, 0), 2)  # 蓝色圆圈

        # 统计棋子数量
        black_count = sum(row.count(2) for row in board_state)
        white_count = sum(row.count(1) for row in board_state)
        print(f"棋子识别完成：黑棋{black_count}个 | 白棋{white_count}个")
        
        # 保存棋子识别调试图像
        cv2.imwrite(os.path.join(DEBUG_DIR, "debug_pieces_identified.jpg"), debug_img)
        
        return board_state

    def capture_and_process(self):
        """截图并处理棋盘，返回15*15的棋盘数组"""
        import time
        start_time = time.time()
        
        # 加载颜色配置
        loaded_sample = ColorSample.load_config(os.path.join(CONFIG_DIR, DEFAULT_CONFIG))
        if loaded_sample:
            self.color_sample = loaded_sample
            print("颜色配置加载成功")
        else:
            print("未找到颜色配置文件")
            return None

        # 截图
        screenshot = self.screenshot_game()
        if screenshot is None:
            print("截图失败")
            return None

        # 定位棋盘
        chessboard_result = self.locate_chessboard(screenshot)
        if chessboard_result is None:
            print("棋盘定位失败")
            return None
            
        # 解析棋盘定位结果
        if isinstance(chessboard_result, tuple) and len(chessboard_result) == 3:
            chessboard, board_x, board_y = chessboard_result
        else:
            # 兼容旧版本返回值
            chessboard = chessboard_result
            
        # 检查棋盘图像是否有效
        if chessboard is None or chessboard.size == 0:
            print("棋盘图像无效")
            return None

        # 识别棋子
        board_state = self.detect_pieces(chessboard)
        if board_state is None:
            print("棋子识别失败")
            return None

        elapsed_time = time.time() - start_time
        print(f"图像处理耗时: {elapsed_time:.2f}秒")
        return (board_state, elapsed_time), (board_x, board_y)

    def collect_color_samples(self, chessboard):
        """修复后的颜色采样函数"""
        # 检查输入参数是否有效
        if chessboard is None or chessboard.size == 0:
            print("错误：输入的棋盘图像无效")
            return False
            
        # 确保输入是有效的numpy数组
        if not isinstance(chessboard, np.ndarray):
            print("错误：棋盘图像不是有效的numpy数组")
            return False

        try:
            img_resized = cv2.resize(chessboard, (BOARD_RESIZE, BOARD_RESIZE), interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(f"图像缩放失败：{str(e)}")
            return False

        # 使用类属性来存储采样状态
        self.bg_samples = []
        self.black_samples = []
        self.white_samples = []
        self.sample_state = 0  # 0=背景, professional_26.json=黑棋, 2=白棋
        self.sample_marked_img = img_resized.copy()

        # 创建窗口
        cv2.namedWindow("Color Sampling", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Color Sampling", 800, 800)

        # 显示初始说明
        instructions = img_resized.copy()
        cv2.putText(instructions, "Click to sample background color (5 times)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(instructions, "Press any key to start", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Color Sampling", instructions)
        cv2.waitKey(0)

        # 设置鼠标回调
        cv2.setMouseCallback("Color Sampling", self._sampling_mouse_callback)

        # 更新显示
        self._update_sampling_display()

        print("\n【颜色采样步骤】")
        print(f"1. 点击 {SAMPLE_COUNT} 个不同的棋盘空白区域 → 采集背景色")
        print(f"2. 点击 {SAMPLE_COUNT} 个不同的黑棋中心 → 采集黑棋色")
        print(f"3. 点击 {SAMPLE_COUNT} 个不同的白棋（含轻微阴影/高光）→ 采集白棋色")
        print("完成后关闭采样窗口")

        # 等待窗口关闭
        while True:
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty("Color Sampling", cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == 27:  # ESC键退出
                break

        cv2.destroyAllWindows()

        # 检查是否完成所有采样
        if (len(self.bg_samples) == SAMPLE_COUNT and
                len(self.black_samples) == SAMPLE_COUNT and
                len(self.white_samples) == SAMPLE_COUNT):

            # 计算平均颜色并设置
            if self.bg_samples:
                avg_bg = np.mean(np.array(self.bg_samples), axis=0).astype(np.uint8)
                self.color_sample.set_board_color(avg_bg)
                print(f"棋盘背景色: BGR{tuple(avg_bg)}")

            if self.black_samples:
                avg_black = np.mean(np.array(self.black_samples), axis=0).astype(np.uint8)
                self.color_sample.set_black_color(avg_black)
                print(f"黑棋色: BGR{tuple(avg_black)}")

            if self.white_samples:
                self.color_sample.white_samples_bgr = self.white_samples
                self.color_sample.finalize_white_model()
                print(f"白棋样本数: {len(self.white_samples)}")

            # 保存配置
            self.color_sample.save_config(os.path.join(CONFIG_DIR, DEFAULT_CONFIG))
            return True
        else:
            print("采样未完成")
            return False

    def _sampling_mouse_callback(self, event, x, y, flags, param):
        """采样鼠标回调函数"""
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        half_size = SAMPLE_REGION_SIZE // 2
        x_start = max(0, x - half_size)
        x_end = min(BOARD_RESIZE, x + half_size + 1)
        y_start = max(0, y - half_size)
        y_end = min(BOARD_RESIZE, y + half_size + 1)

        # 获取采样区域
        sample_roi = self.sample_marked_img[y_start:y_end, x_start:x_end]

        if sample_roi.size == 0:
            return

        avg_color = self.get_avg_color_without_outliers(sample_roi)
        print(f"采样位置: ({x}, {y}), 平均颜色: BGR{tuple(avg_color)}")

        if self.sample_state == 0:  # 背景采样
            self.bg_samples.append(avg_color)
            cv2.circle(self.sample_marked_img, (x, y), 8, (255, 0, 0), 2)
            cv2.putText(self.sample_marked_img, f"BG-{len(self.bg_samples)}", (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            print(f"已采集背景色 ({len(self.bg_samples)}/{SAMPLE_COUNT})")

            if len(self.bg_samples) >= SAMPLE_COUNT:
                self.sample_state = 1
                print("\n开始采集黑棋色...")

        elif self.sample_state == 1:  # 黑棋采样
            self.black_samples.append(avg_color)
            cv2.circle(self.sample_marked_img, (x, y), 8, (0, 0, 255), 2)
            cv2.putText(self.sample_marked_img, f"BLACK-{len(self.black_samples)}", (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            print(f"已采集黑棋色 ({len(self.black_samples)}/{SAMPLE_COUNT})")

            if len(self.black_samples) >= SAMPLE_COUNT:
                self.sample_state = 2
                print("\n开始采集白棋色...")

        elif self.sample_state == 2:  # 白棋采样
            self.white_samples.append(avg_color)
            cv2.circle(self.sample_marked_img, (x, y), 8, (0, 255, 0), 2)
            cv2.putText(self.sample_marked_img, f"WHITE-{len(self.white_samples)}", (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print(f"已采集白棋色 ({len(self.white_samples)}/{SAMPLE_COUNT})")

            if len(self.white_samples) >= SAMPLE_COUNT:
                print("\n所有采样完成！可以关闭窗口。")

        # 更新显示
        self._update_sampling_display()

    def _update_sampling_display(self):
        """更新采样显示"""
        display_img = self.sample_marked_img.copy()

        # 添加当前状态说明
        status_text = ""
        if self.sample_state == 0:
            status_text = f"Background Sampling: {len(self.bg_samples)}/{SAMPLE_COUNT}"
        elif self.sample_state == 1:
            status_text = f"Black Piece Sampling: {len(self.black_samples)}/{SAMPLE_COUNT}"
        elif self.sample_state == 2:
            status_text = f"White Piece Sampling: {len(self.white_samples)}/{SAMPLE_COUNT}"

        cv2.putText(display_img, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 添加图例
        cv2.putText(display_img, "Blue: Background", (10, BOARD_RESIZE - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(display_img, "Red: Black Pieces", (10, BOARD_RESIZE - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(display_img, "Green: White Pieces", (10, BOARD_RESIZE - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Color Sampling", display_img)

    def manual_sampling(self, root_window=None):
        """手动颜色采样"""
        try:
            # 截图并定位棋盘
            screenshot = self.screenshot_game()
            if screenshot is None:
                print("截图失败")
                return False

            chessboard_result = self.locate_chessboard(screenshot)
            if chessboard_result is None:
                print("棋盘定位失败")
                return False
                
            # 解析棋盘定位结果
            if isinstance(chessboard_result, tuple) and len(chessboard_result) == 3:
                chessboard, board_x, board_y = chessboard_result
            else:
                # 兼容旧版本返回值
                chessboard = chessboard_result
            
            # 检查棋盘图像是否有效
            if chessboard is None or chessboard.size == 0:
                print("棋盘图像无效")
                return False

            # 重置颜色样本
            self.color_sample = ColorSample()

            # 调用修复后的采样函数
            success = self.collect_color_samples(chessboard)
            
            # 如果采样成功，询问用户是否保存为特定名称
            if success and root_window:
                config_name = simpledialog.askstring("保存配置", "请输入配置名称:", parent=root_window)
                if config_name:
                    # 保存为指定名称的配置文件
                    config_filename = f"{config_name}.json"
                    config_path = os.path.join(CONFIG_DIR, config_filename)
                    self.color_sample.save_config(config_path)
                    print(f"配置已保存为: {config_filename}")

            return success

        except Exception as e:
            print(f"手动采样过程中出现错误：{str(e)}")
            return False

    def get_available_configs(self):
        """获取所有可用的配置文件"""
        configs = []
        if os.path.exists(CONFIG_DIR):
            for file in os.listdir(CONFIG_DIR):
                if file.endswith(".json"):
                    configs.append(file)
        return configs

    def load_config_by_name(self, config_name):
        """根据名称加载配置文件"""
        config_path = os.path.join(CONFIG_DIR, config_name)
        loaded_sample = ColorSample.load_config(config_path)
        if loaded_sample:
            self.color_sample = loaded_sample
            # 同时更新默认配置文件
            default_path = os.path.join(CONFIG_DIR, DEFAULT_CONFIG)
            loaded_sample.save_config(default_path)
            return True
        return False