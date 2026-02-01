import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox, font

from ai_calculator import AICalculator

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from image_processor import ImageProcessor


# -------------------------- 主应用类 --------------------------
class GomokuAssistant:
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.ai_calculator = AICalculator(use_mixed_precision=False)
        
        self.player_color = 2  # professional_26.json:白棋, 2:黑棋，默认为黑棋
        self.output_dir = "screenshots"
        os.makedirs(self.output_dir, exist_ok=True)
        self.suggestion_animation_id = None  # 用于动画的ID
        self.is_processing = False  # 控制并发分析
        self.capture_btn = None
        self.fonts = {
            'title': ('Segoe UI', 20, 'bold'),
            'subtitle': ('Segoe UI Semibold', 13),
            'body': ('Segoe UI', 11),
            'body_bold': ('Segoe UI Semibold', 11),
            'mono': ('Cascadia Mono', 10),
            'status': ('Cascadia Mono', 10)
        }

        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("五子棋AI助手")
        self.root.geometry("550x1070")
        self.root.configure(bg="#f0f0f0")
        self.root.resizable(False, False)  # 禁止调整窗口大小

        self.status_var = tk.StringVar(value="准备就绪")

        # 设置现代化样式
        self.setup_styles()
        self.setup_ui()
        
        # 程序启动后自动执行截图分析
        self.root.after(1000, self.capture_and_analyze)
        
        # 存储单选按钮引用以便更新样式
        self.black_radio = None
        self.white_radio = None

    def setup_styles(self):
        """设置现代化样式"""
        # 定义颜色主题
        self.style = ttk.Style()
        try:
            self.style.theme_use('clam')
        except tk.TclError:
            pass

        self.colors = {
            'primary': '#1f2a44',
            'secondary': '#253450',
            'accent': '#3498db',
            'success': '#27ae60',
            'warning': '#f1c40f',
            'danger': '#e74c3c',
            'light': '#f6f8fb',
            'dark': '#1f2a44',
            'board_bg': '#DEB887',
            'card_bg': '#ffffff',
            'card_border': '#dfe3eb',
            'text_primary': '#1f2a44',
            'text_secondary': '#6b7285'
        }
        
        # 配置样式
        self.style.configure('Title.TLabel', 
                           font=self.fonts['title'],
                           foreground=self.colors['primary'],
                           background="#f0f0f0")
        
        self.style.configure('Header.TLabel',
                           font=self.fonts['subtitle'],
                           foreground=self.colors['text_primary'],
                           background=self.colors['card_bg'])
        
        self.style.configure('Card.TFrame',
                           background=self.colors['card_bg'],
                           relief='solid',
                           borderwidth=1)
        
        self.style.configure('Primary.TButton',
                           font=self.fonts['body_bold'],
                           background=self.colors['accent'],
                           foreground='white',
                           padding=6)
        
        self.style.map('Primary.TButton',
                      background=[('active', '#e67822')])
        
        self.style.configure('Secondary.TButton',
                           font=self.fonts['body'],
                           background=self.colors['light'],
                           foreground=self.colors['text_primary'],
                           padding=6)
        
        self.style.map('Secondary.TButton',
                      background=[('active', '#e4e9f2')])
        
        self.style.configure('Settings.TLabelframe',
                           background=self.colors['card_bg'],
                           foreground=self.colors['primary'],
                           font=self.fonts['body_bold'])
        
        self.style.configure('Settings.TLabelframe.Label',
                           background=self.colors['card_bg'],
                           foreground=self.colors['primary'],
                           font=self.fonts['body_bold'])

    def setup_ui(self):
        """设置用户界面"""
        # 主框架
        main_frame = tk.Frame(self.root, bg="#f0f0f0", padx=20, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 标题（已删除界面中的标题文字，只保留窗口标题）
        title_frame = tk.Frame(main_frame, bg="#f0f0f0")
        title_frame.pack(fill=tk.X, pady=(0, 5))  # 减小间距从20到10
        
        # 移除了界面中的标题标签，只保留窗口标题

        # 设置区域卡片（移除了"游戏设置"标题）
        settings_card = tk.Frame(main_frame, bg=self.colors['card_bg'], relief="solid", bd=1, 
                                highlightbackground="#ddd", highlightthickness=1)
        settings_card.pack(fill=tk.X, pady=(0, 10))
        
        # 玩家颜色选择（移除了"游戏设置"标题）
        color_frame = tk.Frame(settings_card, bg=self.colors['card_bg'])
        color_frame.pack(fill=tk.X, padx=15, pady=10)
        
        color_label = tk.Label(color_frame, text="选择你的棋子:", 
                              bg=self.colors['card_bg'], fg=self.colors['text_secondary'], 
                              font=self.fonts['body_bold'])  # 加大加粗字体
        color_label.pack(side=tk.LEFT)
        
        self.color_var = tk.StringVar(value="black")
        black_radio = tk.Radiobutton(color_frame, text="黑棋", variable=self.color_var,
                                    value="black", command=self.on_color_change,
                                    bg="#e0e0e0", fg="black",  # 默认背景和文字颜色
                                    activebackground="#e0e0e0",
                                    font=self.fonts['body_bold'], selectcolor=self.colors['accent'],  # 选中时背景色
                                    indicatoron=0,  # 使用按钮样式而不是默认的圆形
                                    width=6, height=1,  # 减小控件尺寸
                                    relief='raised', bd=2,  # 添加边框
                                    activeforeground='white')  # 选中时文字为白色
        black_radio.pack(side=tk.LEFT, padx=(15, 0))
        
        white_radio = tk.Radiobutton(color_frame, text="白棋", variable=self.color_var,
                                    value="white", command=self.on_color_change,
                                    bg="#e0e0e0", fg="black",  # 默认背景和文字颜色
                                    activebackground="#e0e0e0",
                                    font=self.fonts['body_bold'], selectcolor=self.colors['accent'],  # 选中时背景色
                                    indicatoron=0,  # 使用按钮样式而不是默认的圆形
                                    width=6, height=1,  # 减小控件尺寸
                                    relief='raised', bd=2,  # 添加边框
                                    activeforeground='white')  # 选中时文字为白色
        white_radio.pack(side=tk.LEFT, padx=(15, 0))

        # 窗口选择区域
        window_frame = tk.Frame(settings_card, bg=self.colors['card_bg'])
        window_frame.pack(fill=tk.X, padx=15, pady=(0, 10))
        
        window_label = tk.Label(window_frame, text="游戏窗口:", 
                               bg=self.colors['card_bg'], fg=self.colors['text_secondary'], 
                               font=self.fonts['body_bold'])
        window_label.pack(side=tk.LEFT)
        
        self.window_var = tk.StringVar()
        self.window_combobox = ttk.Combobox(window_frame, textvariable=self.window_var, width=20)
        self.window_combobox.pack(side=tk.LEFT, padx=(15, 0))
        self.update_window_list()
        
        refresh_window_btn = tk.Button(window_frame, text="刷新", command=self.update_window_list,
                                      bg=self.colors['light'], fg=self.colors['text_primary'],
                                      font=self.fonts['body'], relief='flat', padx=10, pady=5)
        refresh_window_btn.pack(side=tk.LEFT, padx=(5, 0))
        
        select_window_btn = tk.Button(window_frame, text="选择", command=self.select_window,
                                     bg=self.colors['accent'], fg='white',
                                     font=self.fonts['body_bold'], relief='flat', padx=10, pady=5)
        select_window_btn.pack(side=tk.LEFT, padx=(5, 0))

        # 配置选择区域
        config_frame = tk.Frame(settings_card, bg=self.colors['card_bg'])
        config_frame.pack(fill=tk.X, padx=15, pady=(0, 10))
        
        config_label = tk.Label(config_frame, text="采样配置:", 
                               bg=self.colors['card_bg'], fg=self.colors['text_secondary'], 
                               font=self.fonts['body_bold'])
        config_label.pack(side=tk.LEFT)
        
        self.config_var = tk.StringVar()
        self.config_combobox = ttk.Combobox(config_frame, textvariable=self.config_var, width=20)
        self.config_combobox.pack(side=tk.LEFT, padx=(15, 0))
        self.update_config_list()
        
        refresh_btn = tk.Button(config_frame, text="刷新", command=self.update_config_list,
                               bg=self.colors['light'], fg=self.colors['text_primary'],
                               font=self.fonts['body'], relief='flat', padx=10, pady=5)
        refresh_btn.pack(side=tk.LEFT, padx=(5, 0))
        
        load_btn = tk.Button(config_frame, text="加载", command=self.load_selected_config,
                            bg=self.colors['accent'], fg='white',
                            font=self.fonts['body_bold'], relief='flat', padx=10, pady=5)
        load_btn.pack(side=tk.LEFT, padx=(5, 0))

        # 操作按钮区域卡片（移除了"操作"标题）
        buttons_card = tk.Frame(main_frame, bg=self.colors['card_bg'], relief="solid", bd=1,
                               highlightbackground="#ddd", highlightthickness=1)
        buttons_card.pack(fill=tk.X, pady=(0, 12))
        
        buttons_frame = tk.Frame(buttons_card, bg=self.colors['card_bg'])
        buttons_frame.pack(fill=tk.X, padx=15, pady=12)
        
        self.capture_btn = tk.Button(buttons_frame, text="AI一下",
                                     command=self.capture_and_analyze,
                                     bg=self.colors['accent'], fg='white',
                                     font=self.fonts['body_bold'],
                                     relief='flat', padx=25, pady=10,
                                     cursor="hand2")
        self.capture_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        sample_btn = tk.Button(buttons_frame, text="棋子采样",
                              command=self.manual_sampling,
                              bg=self.colors['light'], fg=self.colors['text_primary'],
                              font=self.fonts['body'],
                               relief='flat', padx=25, pady=10,
                               cursor="hand2")
        sample_btn.pack(side=tk.LEFT)
        
        status_label = tk.Label(buttons_card, textvariable=self.status_var,
                                bg=self.colors['card_bg'], fg=self.colors['text_secondary'],
                                font=self.fonts['status'], anchor='w')
        status_label.pack(fill=tk.X, padx=15, pady=(0, 12))

        # 棋盘显示区域卡片
        board_card = tk.Frame(main_frame, bg=self.colors['card_bg'], relief="solid", bd=1,
                             highlightbackground="#ddd", highlightthickness=1)
        board_card.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        board_header = ttk.Label(board_card, text="棋盘状态", style='Header.TLabel')
        board_header.pack(anchor=tk.W, padx=15, pady=(15, 10))
        
        board_frame = tk.Frame(board_card, bg=self.colors['card_bg'])
        board_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        
        # 创建画布用于显示图形化棋盘
        self.board_canvas = tk.Canvas(board_frame, bg=self.colors['board_bg'], 
                                     width=480, height=480, bd=0, highlightthickness=0,
                                     relief='ridge')
        self.board_canvas.pack(fill=tk.BOTH, expand=True)

        # AI建议显示区域卡片
        advice_card = tk.Frame(main_frame, bg=self.colors['card_bg'], relief="solid", bd=1,
                              highlightbackground="#ddd", highlightthickness=1)
        advice_card.pack(fill=tk.X)
        
        advice_header = ttk.Label(advice_card, text="AI建议", style='Header.TLabel')
        advice_header.pack(anchor=tk.W, padx=15, pady=(15, 10))
        
        advice_frame = tk.Frame(advice_card, bg=self.colors['card_bg'])
        advice_frame.pack(fill=tk.X, padx=15, pady=(0, 12))
        advice_frame.grid_columnconfigure(0, weight=1)
        
        self.advice_text = tk.Text(advice_frame, width=50, height=8, 
                                  font=self.fonts['body'], 
                                  bg="#f8f9fa", fg=self.colors['text_primary'],
                                  relief="solid", bd=1, padx=15, pady=12,
                                  wrap=tk.WORD)
        self.advice_text.grid(row=0, column=0, sticky="nsew")

        # 添加滚动条到AI建议文本框
        advice_scrollbar = tk.Scrollbar(advice_frame, orient="vertical", command=self.advice_text.yview)
        advice_scrollbar.grid(row=0, column=1, sticky="ns")
        self.advice_text.config(yscrollcommand=advice_scrollbar.set)

    def set_processing_state(self, processing, message=None):
        """更新AI处理状态并同步按钮、提示文字。"""
        self.is_processing = processing
        if self.capture_btn:
            state = tk.DISABLED if processing else tk.NORMAL
            self.capture_btn.config(state=state)
        if message is not None:
            self.status_var.set(message)

    def update_window_list(self):
        """更新窗口列表"""
        windows = self.image_processor.get_all_window_titles()
        self.window_combobox['values'] = windows
        if windows:
            # 设置默认选中项为第一个
            self.window_var.set(windows[0])

    def select_window(self):
        """选择窗口"""
        selected_window = self.window_var.get()
        if selected_window:
            success = self.image_processor.set_selected_window(selected_window)
            if success:
                messagebox.showinfo("成功", f"已选择窗口: {selected_window}")
            else:
                messagebox.showerror("错误", f"选择窗口失败: {selected_window}")
        else:
            messagebox.showwarning("警告", "请先选择一个窗口")

    def update_config_list(self):
        """更新配置文件列表"""
        configs = self.image_processor.get_available_configs()
        self.config_combobox['values'] = configs
        if configs:
            # 设置默认选中项
            if "default_config.json" in configs:
                self.config_var.set("default_config.json")
            else:
                self.config_var.set(configs[0])

    def load_selected_config(self):
        """加载选中的配置文件"""
        selected_config = self.config_var.get()
        if selected_config:
            success = self.image_processor.load_config_by_name(selected_config)
            if success:
                messagebox.showinfo("成功", f"已加载配置: {selected_config}")
            else:
                messagebox.showerror("错误", f"加载配置失败: {selected_config}")
        else:
            messagebox.showwarning("警告", "请先选择一个配置文件")

    def draw_board(self, board_state):
        """绘制图形化棋盘"""
        self.board_canvas.delete("all")
        
        # 棋盘尺寸
        canvas_width = self.board_canvas.winfo_width()
        canvas_height = self.board_canvas.winfo_height()
        
        # 如果画布尺寸为1，使用默认值
        if canvas_width <= 1:
            canvas_width = 480
            canvas_height = 480
            
        margin = 30  # 增加边距以避免坐标被遮挡
        board_size = min(canvas_width, canvas_height) - 2 * margin
        cell_size = board_size // 14
        
        # 绘制列坐标 (A-O) - 仅在下方显示
        for i in range(15):
            x = margin + i * cell_size
            self.board_canvas.create_text(x, margin + 14 * cell_size + 20, 
                                         text=chr(ord('A') + i), 
                                         font=self.fonts['mono'], 
                                         fill=self.colors['text_primary'])
        
        # 绘制行坐标 (15-professional_26.json) - 仅在左侧显示，从上到下为15-professional_26.json
        for i in range(15):
            y = margin + i * cell_size
            self.board_canvas.create_text(margin - 20, y, 
                                         text=str(15 - i), 
                                         font=self.fonts['mono'], 
                                         fill=self.colors['text_primary'])
        
        # 绘制棋盘线
        for i in range(15):
            # 垂直线
            x = margin + i * cell_size
            self.board_canvas.create_line(x, margin, x, margin + 14 * cell_size, 
                                         fill=self.colors['dark'], width=1)
            
            # 水平线
            y = margin + i * cell_size
            self.board_canvas.create_line(margin, y, margin + 14 * cell_size, y, 
                                         fill=self.colors['dark'], width=1)
        
        # 绘制天元和星位
        star_points = [(3, 3), (3, 11), (11, 3), (11, 11), (7, 7)]
        for row, col in star_points:
            x = margin + col * cell_size
            y = margin + row * cell_size
            self.board_canvas.create_oval(x-4, y-4, x+4, y+4, fill=self.colors['dark'], outline="")

        # 绘制棋子（移除了阴影和高光效果）
        for row in range(15):
            for col in range(15):
                if board_state[row][col] != 0:
                    x = margin + col * cell_size
                    y = margin + row * cell_size
                    color = "black" if board_state[row][col] == 2 else "white"
                    outline = "black" if board_state[row][col] == 1 else "black"
                    
                    # 绘制棋子主体（无阴影和高光）
                    self.board_canvas.create_oval(
                        x - cell_size//2 + 2, 
                        y - cell_size//2 + 2,
                        x + cell_size//2 - 2,
                        y + cell_size//2 - 2,
                        fill=color, 
                        outline=outline,
                        width=1
                    )

    def highlight_suggestion(self, row, col):
        """高亮显示AI建议的落子位置，使用闪闪发光的棋子"""
        # 棋盘尺寸
        canvas_width = self.board_canvas.winfo_width()
        canvas_height = self.board_canvas.winfo_height()
        
        # 如果画布尺寸为1，则使用默认值
        if canvas_width <= 1:
            canvas_width = 480
            canvas_height = 480
            
        margin = 30  # 与draw_board中使用的边距保持一致
        board_size = min(canvas_width, canvas_height) - 2 * margin
        cell_size = board_size // 14
        
        # 计算落子位置坐标
        x = margin + col * cell_size
        y = margin + row * cell_size
        
        # 确定推荐棋子颜色（与玩家颜色相同）
        # 玩家选择黑棋(self.player_color=2)时，AI推荐黑棋
        # 玩家选择白棋(self.player_color=professional_26.json)时，AI推荐白棋
        color = "black" if self.player_color == 2 else "white"
        
        # 创建闪闪发光的效果
        self.create_shining_stone(x, y, color, cell_size)

    def create_shining_stone(self, x, y, color, cell_size):
        """创建闪闪发光的棋子"""
        # 停止之前的动画
        if self.suggestion_animation_id:
            self.root.after_cancel(self.suggestion_animation_id)
        
        # 绘制棋子
        stone = self.board_canvas.create_oval(
            x - cell_size//2 + 2, 
            y - cell_size//2 + 2,
            x + cell_size//2 - 2,
            y + cell_size//2 - 2,
            fill=color, 
            outline="gold",
            width=2
        )
        
        # 绘制发光效果（多个同心圆，透明度不同）
        glow_circles = []
        for i in range(3):
            circle = self.board_canvas.create_oval(
                x - cell_size//2 - i*3, 
                y - cell_size//2 - i*3,
                x + cell_size//2 + i*3,
                y + cell_size//2 + i*3,
                outline="gold",
                width=2
            )
            glow_circles.append(circle)
        
        # 创建闪烁动画
        self.animate_glow(glow_circles, 0)

    def animate_glow(self, glow_circles, step):
        """动画效果：让光晕闪烁"""
        if step % 2 == 0:
            # 显示光晕
            for circle in glow_circles:
                self.board_canvas.itemconfig(circle, state='normal')
        else:
            # 隐藏光晕
            for circle in glow_circles:
                self.board_canvas.itemconfig(circle, state='hidden')
        
        # 继续动画
        step = (step + 1) % 2
        self.suggestion_animation_id = self.root.after(500, self.animate_glow, glow_circles, step)

    def on_color_change(self):
        """玩家颜色改变回调"""
        self.player_color = 1 if self.color_var.get() == "white" else 2

    def start_radio_animation(self):
        """开始单选按钮的闪烁动画"""
        self.animate_radio(0)

    def animate_radio(self, step):
        """动画效果：让选中的单选按钮闪烁"""
        if self.selected_radio is None:
            return
            
        if step % 2 == 0:
            # 显示亮色
            self.selected_radio.config(bg="#5dade2", fg="white")
        else:
            # 显示原始颜色
            self.selected_radio.config(bg="#3498db", fg="white")
        
        # 继续动画
        step = (step + 1) % 2
        self.radio_animation_id = self.root.after(300, self.animate_radio, step)

    def capture_and_analyze(self):
        """截图并分析棋盘"""
        if self.is_processing:
            self.status_var.set("上一轮分析仍在进行，请稍候...")
            return

        self.set_processing_state(True, "正在准备截图...")
        # 确保使用最新的玩家颜色设置
        self.on_color_change()
        
        # 在后台线程中执行耗时操作
        threading.Thread(target=self._capture_and_analyze_thread, daemon=True).start()

    def _capture_and_analyze_thread(self):
        """在后台线程中执行截图和分析"""
        def fail(message, dialog_fn=None):
            if dialog_fn:
                self.root.after(0, dialog_fn)
            self.root.after(0, lambda m=message: self.set_processing_state(False, m))

        try:
            # 加载颜色配置
            if not os.path.exists(os.path.join("color_configs", "default_config.json")):
                fail("缺少颜色配置，请先采样", lambda: messagebox.showwarning("警告", "未找到颜色配置文件，请先进行手动采样"))
                return

            # 截图
            self.root.after(0, lambda: self._update_advice_text("正在截图..."))
            self.root.after(0, lambda: self.status_var.set("正在截图..."))

            # 使用图像处理器截图并识别棋盘
            result = self.image_processor.capture_and_process()
            
            if result is None:
                fail("截图或识别失败", lambda: messagebox.showerror("错误", "截图或识别失败"))
                return

            # 解析图像处理结果
            if isinstance(result, tuple) and len(result) == 2:
                # 新格式：((board_state, elapsed_time), (board_x, board_y))
                board_info, board_position = result
                if isinstance(board_info, tuple) and len(board_info) == 2:
                    board_state, image_processing_time = board_info
                    board_x, board_y = board_position
                else:
                    # 兼容旧格式
                    board_state = board_info
                    image_processing_time = 0
                    board_x, board_y = 0, 0
            else:
                # 旧格式：board_state
                board_state = result
                image_processing_time = 0
                board_x, board_y = 0, 0

            if board_state is None:
                fail("棋盘识别失败", lambda: messagebox.showerror("错误", "截图或识别失败"))
                return

            # 显示图形化棋盘
            self.root.after(0, lambda: self.draw_board(board_state))

            # 自动调用AI建议，帮助用户决策
            self.root.after(0, lambda t=image_processing_time: self._update_advice_text(f"识别完成，耗时{t:.2f}秒，正在获取AI建议..."))
            self.root.after(0, lambda: self.status_var.set("识别完成，AI正在思考..."))

            # 直接调用AI算法获取最佳落子位置（AI帮助用户下棋）
            self.auto_get_ai_suggestion(board_state, image_processing_time)

        except Exception as e:
            fail("分析过程中出现错误", lambda: messagebox.showerror("错误", f"分析过程中出现错误：{str(e)}"))

    def auto_get_ai_suggestion(self, board_state, image_processing_time=0):
        """自动获取AI建议（内部调用）"""
        # 在后台线程中执行AI计算
        threading.Thread(target=self._auto_get_ai_suggestion_thread, args=(board_state, image_processing_time), daemon=True).start()

    def _auto_get_ai_suggestion_thread(self, board_state, image_processing_time=0):
        """在后台线程中执行AI建议计算"""
        try:
            # 获取AI建议
            self.root.after(0, lambda: self._update_advice_text("AI正在思考..."))
            self.root.after(0, lambda: self.status_var.set("AI正在思考..."))

            # AI计算用户方（玩家选择的颜色）的最佳落子位置
            # player_color: professional_26.json=白棋, 2=黑棋
            result = self.ai_calculator.get_best_move(board_state, self.player_color)
            
            # 解析AI返回的结果
            best_move = None
            elapsed_time = 0
            
            # 检查返回结果的格式
            if isinstance(result, tuple) and len(result) == 2:
                # 格式为 ((row, col), elapsed_time)
                best_move, elapsed_time = result
            else:
                # 其他格式，直接使用
                best_move = result

            if best_move is not None and best_move != (None, 0):
                # 确保best_move是有效的坐标对
                if isinstance(best_move, tuple) and len(best_move) == 2 and \
                   best_move[0] is not None and best_move[1] is not None:
                    row, col = best_move
                else:
                    self.root.after(0, lambda: self._update_advice_text("AI返回了无效的落子位置"))
                    self.root.after(0, lambda: self.set_processing_state(False, "AI返回了无效结果"))
                    return
                    
                # 转换为棋盘坐标（A-O, 15-professional_26.json）
                col_char = chr(ord('A') + col)
                row_num = 15 - row  # 调整行号以匹配棋盘显示


                advice = f"图像识别耗时：{image_processing_time:.2f}秒\n"
                advice += f"AI建议落子位置：{col_char}{row_num}\n"
                advice += f"坐标：第{row_num}行，第{col + 1}列\n"
                if elapsed_time > 0:
                    advice += f"AI思考耗时：{elapsed_time:.2f}秒\n"
                else:
                    advice += "AI思考耗时：时间信息不可用\n"
                
                advice += f"建议在横线{row_num}与竖线{col_char}的交点处落子"

                self.root.after(0, lambda a=advice: self._update_advice_text(a))
                
                # 在棋盘上高亮显示建议位置
                self.root.after(0, lambda: self.highlight_suggestion(row, col))
                self.root.after(0, lambda: self.set_processing_state(False, "AI建议已更新"))
            else:
                self.root.after(0, lambda: self._update_advice_text("AI未能找到合适的落子位置"))
                self.root.after(0, lambda: self.set_processing_state(False, "AI未能找到合适的落子"))

        except Exception as e:
            def show_error():
                messagebox.showerror("错误", f"获取AI建议时出现错误：{str(e)}")

            self.root.after(0, show_error)
            self.root.after(0, lambda: self.set_processing_state(False, "AI建议获取失败"))

    def _update_advice_text(self, text):
        """更新AI建议文本框"""
        self.advice_text.delete(1.0, tk.END)
        self.advice_text.insert(tk.END, text)

    def manual_sampling(self):
        """手动颜色采样"""
        try:
            # 使用图像处理器进行采样
            success = self.image_processor.manual_sampling(self.root)
            
            if success:
                # 更新配置列表
                self.root.after(0, self.update_config_list)
                messagebox.showinfo("成功", "颜色采样完成")
            else:
                messagebox.showwarning("警告", "颜色采样未完成")

        except Exception as e:
            messagebox.showerror("错误", f"手动采样过程中出现错误：{str(e)}")

    def run(self):
        """运行应用程序"""
        self.root.mainloop()


# -------------------------- 主程序 --------------------------
if __name__ == "__main__":
    app = GomokuAssistant()
    app.run()