import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pydicom
import os
import threading
import time
from matplotlib import rcParams

# 设置matplotlib支持中文
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


class WindowLevelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("焊缝射线底片 - 窗宽窗位调节工具")
        self.root.geometry("1200x800")

        # 初始化变量
        self.original_image = None  # 原始16位图像
        self.display_image = None  # 显示用8位图像
        self.processed_image = None  # 处理后的16位图像（可选）
        self.window_width = tk.IntVar(value=32768)
        self.window_level = tk.IntVar(value=32768)
        self.output_bit_depth = tk.StringVar(value="8bit")  # 输出位深选择

        # ROI选择相关变量
        self.roi_selecting = False
        self.roi_start_x = 0
        self.roi_start_y = 0
        self.roi_rect = None
        self.image_scale = 1.0  # 图像缩放比例
        self.image_offset_x = 0  # 图像在画布上的偏移
        self.image_offset_y = 0

        # 更新控制
        self.update_pending = False
        self.last_update_time = 0
        self.update_delay = 0.05  # 50ms延迟

        # 直方图数据缓存
        self.hist_data = None
        self.hist_bins = None

        self.setup_ui()

    def setup_ui(self):
        # 创建菜单栏
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="打开DICOM...", command=lambda: self.load_image('dcm'))
        file_menu.add_command(label="打开BMP...", command=lambda: self.load_image('bmp'))
        file_menu.add_command(label="打开TIF/TIFF...", command=lambda: self.load_image('tif'))
        file_menu.add_command(label="打开所有支持格式...", command=lambda: self.load_image('all'))
        file_menu.add_separator()
        file_menu.add_command(label="保存处理后图像(8位)...", command=lambda: self.save_image(8))
        file_menu.add_command(label="保存处理后图像(16位)...", command=lambda: self.save_image(16))
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)

        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧：图像显示区域
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 图像画布
        self.canvas = tk.Canvas(left_frame, bg='black', width=600, height=600)
        self.canvas.pack(side=tk.TOP, padx=5, pady=5)

        # 绑定鼠标事件 - ROI选择
        self.canvas.bind("<Button-1>", self.on_roi_start)
        self.canvas.bind("<B1-Motion>", self.on_roi_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_roi_end)

        # 参数显示框架
        param_frame = ttk.LabelFrame(left_frame, text="窗宽窗位参数")
        param_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # 窗宽控制（带输入框）
        ww_frame = ttk.Frame(param_frame)
        ww_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(ww_frame, text="窗宽:").pack(side=tk.LEFT)

        # 窗宽输入框
        self.ww_entry = ttk.Entry(ww_frame, width=8, textvariable=self.window_width)
        self.ww_entry.pack(side=tk.LEFT, padx=5)
        self.ww_entry.bind("<Return>", self.on_entry_change)
        self.ww_entry.bind("<FocusOut>", self.on_entry_change)

        # 窗宽滑块
        self.ww_scale = ttk.Scale(ww_frame, from_=1, to=65535,
                                  variable=self.window_width,
                                  command=self.on_scale_change)
        self.ww_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        # 窗位控制（带输入框）
        wl_frame = ttk.Frame(param_frame)
        wl_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(wl_frame, text="窗位:").pack(side=tk.LEFT)

        # 窗位输入框
        self.wl_entry = ttk.Entry(wl_frame, width=8, textvariable=self.window_level)
        self.wl_entry.pack(side=tk.LEFT, padx=5)
        self.wl_entry.bind("<Return>", self.on_entry_change)
        self.wl_entry.bind("<FocusOut>", self.on_entry_change)

        # 窗位滑块
        self.wl_scale = ttk.Scale(wl_frame, from_=0, to=65535,
                                  variable=self.window_level,
                                  command=self.on_scale_change)
        self.wl_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        # 预设按钮
        preset_frame = ttk.Frame(param_frame)
        preset_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(preset_frame, text="自动调节",
                   command=self.auto_window).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="重置",
                   command=self.reset_window).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="清除ROI",
                   command=self.clear_roi).pack(side=tk.LEFT, padx=5)

        # 输出模式选择
        output_frame = ttk.Frame(param_frame)
        output_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(output_frame, text="映射模式:").pack(side=tk.LEFT)
        ttk.Radiobutton(output_frame, text="8位显示 (0-255)",
                        variable=self.output_bit_depth, value="8bit",
                        command=self.on_entry_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(output_frame, text="16位处理 (0-65535)",
                        variable=self.output_bit_depth, value="16bit",
                        command=self.on_entry_change).pack(side=tk.LEFT, padx=5)

        # 右侧：直方图显示区域
        right_frame = ttk.LabelFrame(main_frame, text="像素直方图")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5)

        # 创建matplotlib图形
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 状态栏
        self.status_bar = ttk.Label(self.root, text="请打开一个DICOM、BMP或TIF文件",
                                    relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # 操作说明
        info_text = """操作说明：
1. 支持格式：DICOM (.dcm)、BMP (.bmp)、TIF/TIFF (.tif, .tiff)
2. 直接输入数值或拖动滑块调节窗宽窗位
3. 在图像上框选感兴趣区域(ROI)：
   - 自动计算该区域的最优窗宽窗位
   - 使ROI内像素充分利用显示动态范围
4. 映射模式：
   - 8位显示：映射到0-255，用于屏幕显示
   - 16位处理：映射到0-65535，保留更多细节"""
        info_label = ttk.Label(left_frame, text=info_text, justify=tk.LEFT)
        info_label.pack(side=tk.TOP, padx=5, pady=5)

    def load_image(self, file_type):
        """加载图像文件"""
        if file_type == 'dcm':
            filetypes = [("DICOM files", "*.dcm *.DCM")]
        elif file_type == 'bmp':
            filetypes = [("BMP files", "*.bmp *.BMP")]
        elif file_type == 'tif':
            filetypes = [("TIF/TIFF files", "*.tif *.tiff *.TIF *.TIFF")]
        else:  # 'all'
            filetypes = [
                ("All supported", "*.dcm *.DCM *.bmp *.BMP *.tif *.tiff *.TIF *.TIFF"),
                ("DICOM files", "*.dcm *.DCM"),
                ("BMP files", "*.bmp *.BMP"),
                ("TIF/TIFF files", "*.tif *.tiff *.TIF *.TIFF"),
                ("All files", "*.*")
            ]

        filename = filedialog.askopenfilename(filetypes=filetypes)
        if not filename:
            return

        try:
            self.current_filename = os.path.basename(filename)
            file_ext = os.path.splitext(filename)[1].lower()

            if file_ext in ['.dcm']:
                # 加载DICOM文件
                ds = pydicom.dcmread(filename)
                self.original_image = ds.pixel_array.astype(np.float32)

                # 应用DICOM的rescale slope和intercept
                if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                    self.original_image = self.original_image * ds.RescaleSlope + ds.RescaleIntercept

            elif file_ext in ['.tif', '.tiff']:
                # 加载TIF/TIFF文件
                img = Image.open(filename)

                # 获取图像信息
                mode = img.mode
                bit_depth = img.getextrema()

                # 处理不同模式的TIF图像
                if mode == 'I;16' or mode == 'I;16B' or mode == 'I;16L':
                    # 16位灰度图像
                    self.original_image = np.array(img, dtype=np.float32)
                elif mode == 'I':
                    # 32位灰度图像
                    self.original_image = np.array(img, dtype=np.float32)
                elif mode == 'F':
                    # 32位浮点图像
                    self.original_image = np.array(img, dtype=np.float32)
                elif mode == 'L':
                    # 8位灰度图像
                    self.original_image = np.array(img, dtype=np.float32)
                elif mode == 'RGB' or mode == 'RGBA':
                    # 彩色图像，转换为灰度
                    gray_img = img.convert('L')
                    self.original_image = np.array(gray_img, dtype=np.float32)
                    messagebox.showinfo("信息", "彩色图像已转换为灰度图像")
                else:
                    # 其他模式，尝试转换
                    try:
                        # 尝试转换为16位整数模式
                        img = img.convert('I')
                        self.original_image = np.array(img, dtype=np.float32)
                    except:
                        # 如果失败，转换为8位灰度
                        img = img.convert('L')
                        self.original_image = np.array(img, dtype=np.float32)

                # 输出图像模式信息
                print(f"TIF图像模式: {mode}, 形状: {self.original_image.shape}")

            else:  # BMP或其他格式
                # 加载BMP文件或其他格式
                img = Image.open(filename)
                if img.mode == 'I;16':
                    self.original_image = np.array(img, dtype=np.float32)
                elif img.mode == 'RGB' or img.mode == 'RGBA':
                    # 彩色图像，转换为灰度
                    gray_img = img.convert('L')
                    self.original_image = np.array(gray_img, dtype=np.float32)
                    if file_ext in ['.bmp']:
                        messagebox.showinfo("信息", "彩色BMP图像已转换为灰度图像")
                else:
                    self.original_image = np.array(img.convert('I'), dtype=np.float32)

            # 确保图像是2D的（灰度图）
            if len(self.original_image.shape) == 3:
                # 如果是多通道，取第一个通道或平均值
                if self.original_image.shape[2] == 3 or self.original_image.shape[2] == 4:
                    self.original_image = np.mean(self.original_image[:, :, :3], axis=2)
                else:
                    self.original_image = self.original_image[:, :, 0]

            # 获取图像统计信息
            self.img_min = np.min(self.original_image)
            self.img_max = np.max(self.original_image)
            self.img_mean = np.mean(self.original_image)
            self.img_std = np.std(self.original_image)

            # 计算直方图（只计算一次）
            self.compute_histogram()

            # 更新滑块范围
            self.ww_scale.configure(from_=1, to=int(self.img_max - self.img_min))
            self.wl_scale.configure(from_=int(self.img_min), to=int(self.img_max))

            # 自动设置初始窗宽窗位
            self.auto_window()

            # 更新显示
            self.update_display()
            self.update_histogram()

            # 更新状态栏
            self.status_bar.config(text=f"已加载: {self.current_filename} ({file_ext}) | "
                                        f"尺寸: {self.original_image.shape} | "
                                        f"范围: [{int(self.img_min)}, {int(self.img_max)}]")

        except Exception as e:
            messagebox.showerror("错误", f"无法加载文件：\n{str(e)}")

    def compute_histogram(self):
        """计算并缓存直方图数据"""
        if self.original_image is None:
            return

        # 计算直方图，使用256个bins
        self.hist_data, self.hist_bins = np.histogram(
            self.original_image.flatten(),
            bins=256,
            range=(self.img_min, self.img_max)
        )

    def apply_window_level(self, image, window_width, window_level, output_bits=8):
        """应用窗宽窗位变换"""
        window_min = window_level - window_width / 2
        window_max = window_level + window_width / 2

        output = np.zeros_like(image)

        if output_bits == 8:
            max_val = 255
            dtype = np.uint8
        else:
            max_val = 65535
            dtype = np.uint16

        mask = (image >= window_min) & (image <= window_max)
        output[mask] = ((image[mask] - window_min) / window_width * max_val)

        output[image < window_min] = 0
        output[image > window_max] = max_val

        return output.astype(dtype)

    def on_scale_change(self, event=None):
        """滑块变化时的延迟更新"""
        self.schedule_update()

    def on_entry_change(self, event=None):
        """输入框变化时立即更新"""
        try:
            # 验证输入值的有效性
            ww = self.window_width.get()
            wl = self.window_level.get()

            if self.original_image is not None:
                # 限制范围
                ww = max(1, min(ww, int(self.img_max - self.img_min)))
                wl = max(int(self.img_min), min(wl, int(self.img_max)))

                self.window_width.set(ww)
                self.window_level.set(wl)

            self.update_display()
            self.update_histogram()
        except:
            pass  # 忽略无效输入

    def schedule_update(self):
        """调度延迟更新"""
        current_time = time.time()

        if not self.update_pending:
            self.update_pending = True
            delay = max(0, self.update_delay - (current_time - self.last_update_time))
            self.root.after(int(delay * 1000), self.perform_update)

    def perform_update(self):
        """执行实际更新"""
        self.update_pending = False
        self.last_update_time = time.time()
        self.update_display()
        self.update_histogram()

    def update_display(self):
        """更新图像显示"""
        if self.original_image is None:
            return

        ww = self.window_width.get()
        wl = self.window_level.get()

        # 根据选择的模式处理图像
        if self.output_bit_depth.get() == "16bit":
            self.processed_image = self.apply_window_level(self.original_image, ww, wl, 16)
            display_img = (self.processed_image / 256).astype(np.uint8)
        else:
            display_img = self.apply_window_level(self.original_image, ww, wl, 8)
            self.processed_image = display_img

        # 调整图像大小以适应画布
        h, w = display_img.shape
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width > 1 and canvas_height > 1:
            scale = min(canvas_width / w, canvas_height / h, 1.0)
            self.image_scale = scale
            new_w = int(w * scale)
            new_h = int(h * scale)

            # 计算图像在画布上的偏移（居中显示）
            self.image_offset_x = (canvas_width - new_w) // 2
            self.image_offset_y = (canvas_height - new_h) // 2

            # 转换为PIL图像并调整大小
            pil_img = Image.fromarray(display_img)
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # 转换为PhotoImage
            self.photo = ImageTk.PhotoImage(pil_img)

            # 更新画布
            self.canvas.delete("image")
            self.canvas.create_image(canvas_width // 2, canvas_height // 2,
                                     anchor=tk.CENTER, image=self.photo, tags="image")

            # 如果有ROI框，保持显示
            if self.roi_rect:
                self.canvas.tag_raise(self.roi_rect)

        # 更新状态栏
        if self.original_image is not None:
            mode_text = "16位处理模式" if self.output_bit_depth.get() == "16bit" else "8位显示模式"
            self.status_bar.config(text=f"文件: {self.current_filename} | 模式: {mode_text} | "
                                        f"窗宽: {ww} | 窗位: {wl}")

    def update_histogram(self):
        """更新直方图显示"""
        if self.original_image is None or self.hist_data is None:
            return

        self.ax.clear()

        # 绘制完整直方图（背景，浅色）
        self.ax.bar(self.hist_bins[:-1], self.hist_data,
                    width=self.hist_bins[1] - self.hist_bins[0],
                    color='lightgray', alpha=0.5, edgecolor='none', label='全部像素')

        # 获取窗宽窗位
        ww = self.window_width.get()
        wl = self.window_level.get()
        window_min = wl - ww / 2
        window_max = wl + ww / 2

        # 找出窗口范围内的bins并高亮显示
        window_mask = (self.hist_bins[:-1] >= window_min) & (self.hist_bins[:-1] <= window_max)
        if np.any(window_mask):
            self.ax.bar(self.hist_bins[:-1][window_mask],
                        self.hist_data[window_mask],
                        width=self.hist_bins[1] - self.hist_bins[0],
                        color='steelblue', alpha=0.8, edgecolor='none',
                        label=f'窗口内像素 [{int(window_min)}, {int(window_max)}]')

        # 添加窗口边界线
        self.ax.axvline(x=window_min, color='blue', linestyle='--', linewidth=2, alpha=0.7)
        self.ax.axvline(x=window_max, color='red', linestyle='--', linewidth=2, alpha=0.7)
        self.ax.axvline(x=wl, color='green', linestyle='-', linewidth=2, alpha=0.7)

        # 添加映射说明
        mode = "8位(0-255)" if self.output_bit_depth.get() == "8bit" else "16位(0-65535)"

        self.ax.set_xlabel('像素值')
        self.ax.set_ylabel('频数')
        self.ax.set_title(f'像素直方图与窗宽窗位映射 (输出: {mode})')
        self.ax.legend(loc='upper right', fontsize=8)
        self.ax.grid(True, alpha=0.3)

        # 设置y轴为对数刻度（可选，有助于显示动态范围大的直方图）
        # self.ax.set_yscale('log')

        self.canvas_plot.draw()

    def on_roi_start(self, event):
        """开始ROI选择"""
        self.roi_selecting = True
        self.roi_start_x = event.x
        self.roi_start_y = event.y

        # 删除旧的ROI框
        if self.roi_rect:
            self.canvas.delete(self.roi_rect)
            self.roi_rect = None

    def on_roi_drag(self, event):
        """ROI选择过程中"""
        if not self.roi_selecting:
            return

        # 删除旧的矩形
        if self.roi_rect:
            self.canvas.delete(self.roi_rect)

        # 绘制新的矩形
        self.roi_rect = self.canvas.create_rectangle(
            self.roi_start_x, self.roi_start_y, event.x, event.y,
            outline='yellow', width=2, dash=(5, 5), tags="roi"
        )

    def on_roi_end(self, event):
        """结束ROI选择并计算窗宽窗位"""
        if not self.roi_selecting or self.original_image is None:
            self.roi_selecting = False
            return

        self.roi_selecting = False

        # 计算ROI在图像中的实际位置
        x1 = min(self.roi_start_x, event.x)
        x2 = max(self.roi_start_x, event.x)
        y1 = min(self.roi_start_y, event.y)
        y2 = max(self.roi_start_y, event.y)

        # 检查ROI是否有效（面积大于最小值）
        if abs(x2 - x1) < 10 or abs(y2 - y1) < 10:
            if self.roi_rect:
                self.canvas.delete(self.roi_rect)
                self.roi_rect = None
            return

        # 转换画布坐标到图像坐标
        img_x1 = int((x1 - self.image_offset_x) / self.image_scale)
        img_x2 = int((x2 - self.image_offset_x) / self.image_scale)
        img_y1 = int((y1 - self.image_offset_y) / self.image_scale)
        img_y2 = int((y2 - self.image_offset_y) / self.image_scale)

        # 确保坐标在图像范围内
        h, w = self.original_image.shape
        img_x1 = max(0, min(img_x1, w - 1))
        img_x2 = max(0, min(img_x2, w - 1))
        img_y1 = max(0, min(img_y1, h - 1))
        img_y2 = max(0, min(img_y2, h - 1))

        # 提取ROI区域的像素
        roi_pixels = self.original_image[img_y1:img_y2, img_x1:img_x2]

        if roi_pixels.size > 0:
            # 计算ROI的统计信息
            roi_min = np.min(roi_pixels)
            roi_max = np.max(roi_pixels)
            roi_mean = np.mean(roi_pixels)
            roi_std = np.std(roi_pixels)

            # 设置窗宽窗位
            # 方法1：使用ROI的完整范围
            # new_wl = int((roi_min + roi_max) / 2)
            # new_ww = int(roi_max - roi_min)

            # 方法2：使用均值和标准差（更稳健，排除异常值）
            new_wl = int(roi_mean)
            new_ww = int(min(4 * roi_std, roi_max - roi_min))  # 4倍标准差或全范围

            # 确保窗宽至少为1
            new_ww = max(1, new_ww)

            # 更新窗宽窗位
            self.window_width.set(new_ww)
            self.window_level.set(new_wl)

            # 更新显示
            self.update_display()
            self.update_histogram()

            # 更新状态栏显示ROI信息
            self.status_bar.config(
                text=f"ROI统计 - 范围: [{int(roi_min)}, {int(roi_max)}], "
                     f"均值: {int(roi_mean)}, 标准差: {int(roi_std)} | "
                     f"窗宽: {new_ww}, 窗位: {new_wl}"
            )

    def clear_roi(self):
        """清除ROI选择框"""
        if self.roi_rect:
            self.canvas.delete(self.roi_rect)
            self.roi_rect = None

    def auto_window(self):
        """自动调节窗宽窗位"""
        if self.original_image is None:
            return

        # 使用均值和标准差自动设置
        wl = int(self.img_mean)
        ww = int(4 * self.img_std)

        ww = max(1, min(ww, self.img_max - self.img_min))

        self.window_width.set(ww)
        self.window_level.set(wl)

        self.update_display()
        self.update_histogram()

    def reset_window(self):
        """重置窗宽窗位到全范围"""
        if self.original_image is None:
            return

        self.window_width.set(int(self.img_max - self.img_min))
        self.window_level.set(int((self.img_max + self.img_min) / 2))

        self.clear_roi()
        self.update_display()
        self.update_histogram()

    def save_image(self, bit_depth):
        """保存处理后的图像"""
        if self.original_image is None:
            messagebox.showwarning("警告", "请先加载一个图像")
            return

        ww = self.window_width.get()
        wl = self.window_level.get()

        if bit_depth == 8:
            processed = self.apply_window_level(self.original_image, ww, wl, 8)
            filetypes = [("PNG files", "*.png"), ("BMP files", "*.bmp"), ("TIF files", "*.tif")]
            default_ext = '.png'
        else:
            processed = self.apply_window_level(self.original_image, ww, wl, 16)
            filetypes = [("TIFF files", "*.tif *.tiff"), ("PNG files", "*.png")]
            default_ext = '.tif'

        filename = filedialog.asksaveasfilename(
            defaultextension=default_ext,
            filetypes=filetypes
        )

        if filename:
            try:
                img = Image.fromarray(processed)
                img.save(filename)

                info = f"图像已保存:\n文件: {os.path.basename(filename)}\n"
                info += f"位深: {bit_depth}位\n"
                info += f"窗宽: {ww}\n窗位: {wl}\n"
                info += "像素范围: 0-255" if bit_depth == 8 else "像素范围: 0-65535"

                messagebox.showinfo("保存成功", info)

            except Exception as e:
                messagebox.showerror("保存失败", f"无法保存图像：\n{str(e)}")


def main():
    root = tk.Tk()
    app = WindowLevelApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()