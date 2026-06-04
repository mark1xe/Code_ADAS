import cv2
import os
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import sys

# ── CẤU HÌNH CAMERA ────────
CAM_W = 320
CAM_H = 240
CAM_FIXED_EXPOSURE = 25000  
CAM_ANALOGUE_GAIN  = 4.0    
CAM_FIXED_AWB      = True   
CAM_AWB_GAINS      = (1.6, 1.4)  

# ── TFT SPI display ────────
TFT_W  = 320
TFT_H  = 240
TFT_FB = "/dev/fb1"

try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except ImportError:
    HAS_PICAMERA2 = False

class TFTDisplay:
    def __init__(self):
        self._luma = None
        self._fb   = None
        try:
            from luma.lcd.device import ili9341 as ILI9341Dev
            from luma.core.interface.serial import spi as LumaSPI
            _dev = 0 if os.path.exists("/dev/spidev0.0") else 1
            _serial = LumaSPI(port=0, device=_dev,
                              gpio_DC=25, gpio_RST=24,
                              bus_speed_hz=32_000_000,
                              gpio_LIGHT=None)
            self._luma = ILI9341Dev(_serial, rotate=1, width=TFT_W, height=TFT_H)
            print(f"TFT: luma.lcd OK (SPI0.{_dev} DC=GPIO25 RST=GPIO24)")
            return
        except ImportError:
            pass
        try:
            self._fb = open(TFT_FB, "wb")
            print(f"TFT: framebuffer {TFT_FB} OK")
        except OSError:
            pass

    @property
    def ok(self):
        return self._luma is not None or self._fb is not None

    def show(self, img_rgb: np.ndarray) -> None:
        disp = cv2.resize(img_rgb, (TFT_W, TFT_H), interpolation=cv2.INTER_LINEAR)
        if self._luma is not None:
            try:
                from PIL import Image
                portrait = cv2.rotate(disp, cv2.ROTATE_90_CLOCKWISE)
                self._luma.display(Image.fromarray(portrait))
            except Exception:
                self._luma = None
        elif self._fb is not None:
            try:
                r = disp[:,:,0].astype(np.uint16)
                g = disp[:,:,1].astype(np.uint16)
                b = disp[:,:,2].astype(np.uint16)
                rgb565 = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
                self._fb.seek(0)
                self._fb.write(rgb565.astype(np.uint16).tobytes())
                self._fb.flush()
            except OSError:
                self._fb = None


class DataCollectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Thu thập Dataset (Màu chuẩn 100%)")
        self.root.geometry("450x320")

        self.is_capturing = False
        self.save_thread = None
        
        self.cam = None
        self.camera_running = False
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        self.tft = TFTDisplay()

        # --- UI ---
        tk.Label(root, text="Thư mục lưu:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.folder_path = tk.StringVar(value="/home/pi/dataset")
        tk.Entry(root, textvariable=self.folder_path, width=25).grid(row=0, column=1, pady=5)
        tk.Button(root, text="Duyệt...", command=self.browse_folder).grid(row=0, column=2, padx=5)

        tk.Label(root, text="Bắt đầu đánh số từ:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.start_index = tk.IntVar(value=0)
        tk.Entry(root, textvariable=self.start_index, width=15).grid(row=1, column=1, sticky="w", pady=5)

        tk.Label(root, text="Số lượng ảnh cần chụp:").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        self.total_images = tk.IntVar(value=100)
        tk.Entry(root, textvariable=self.total_images, width=15).grid(row=2, column=1, sticky="w", pady=5)

        tk.Label(root, text="Giãn cách (giây/tấm):").grid(row=3, column=0, sticky="w", padx=10, pady=5)
        self.interval = tk.DoubleVar(value=1.0)
        tk.Entry(root, textvariable=self.interval, width=15).grid(row=3, column=1, sticky="w", pady=5)

        self.status_label = tk.Label(root, text="Đang khởi động Camera...", fg="blue", font=("Arial", 10, "bold"))
        self.status_label.grid(row=4, column=0, columnspan=3, pady=15)

        self.btn_start = tk.Button(root, text="▶ BẮT ĐẦU CHỤP", bg="green", fg="white", font=("Arial", 10, "bold"), command=self.start_capture, state=tk.DISABLED)
        self.btn_start.grid(row=5, column=0, columnspan=2, pady=5)

        self.btn_stop = tk.Button(root, text="⏹ DỪNG LẠI", bg="red", fg="white", font=("Arial", 10, "bold"), command=self.stop_capture, state=tk.DISABLED)
        self.btn_stop.grid(row=5, column=1, columnspan=2, pady=5)

        self.root.after(100, self.init_camera)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def init_camera(self):
        if not HAS_PICAMERA2:
            self.status_label.config(text="LỖI: Chưa cài đặt Picamera2!", fg="red")
            return

        try:
            self.cam = Picamera2()
            self.cam.configure(self.cam.create_preview_configuration(
                main={"format": "RGB888", "size": (CAM_W, CAM_H)}))
            
            self.cam.start()
            time.sleep(1.0) 

            _controls = {}
            _controls["AeEnable"] = False
            _controls["ExposureTime"] = CAM_FIXED_EXPOSURE
            _controls["AnalogueGain"] = CAM_ANALOGUE_GAIN
            
            if CAM_FIXED_AWB:
                _controls["AwbEnable"] = False
                _controls["ColourGains"] = CAM_AWB_GAINS
                
            self.cam.set_controls(_controls)
            time.sleep(0.3)

            self.camera_running = True
            threading.Thread(target=self._capture_worker, daemon=True).start()
            threading.Thread(target=self._display_worker, daemon=True).start()

            self.status_label.config(text=f"Camera OK! ({CAM_W}x{CAM_H}) - Sẵn sàng", fg="green")
            self.btn_start.config(state=tk.NORMAL)

        except Exception as e:
            self.status_label.config(text=f"Lỗi Camera: {e}", fg="red")

    def _capture_worker(self):
        while self.camera_running:
            try:
                with self.frame_lock:
                    self.latest_frame = self.cam.capture_array()
            except Exception:
                pass
            time.sleep(0.02) 

    def _display_worker(self):
        """Luồng hiển thị ảnh TRỰC TIẾP LÊN MÀN HÌNH TFT"""
        while self.camera_running:
            frame_raw = None
            with self.frame_lock:
                if self.latest_frame is not None:
                    # Lấy khung hình gốc (đã xác nhận là BGR)
                    frame_raw = self.latest_frame.copy()
            
            if frame_raw is not None:
                if self.tft is not None and self.tft.ok:
                    # Màn TFT cần RGB, nên ta lật BGR -> RGB
                    frame_for_tft = frame_raw[:, :, ::-1]
                    self.tft.show(frame_for_tft)

            time.sleep(0.03)

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.folder_path.set(folder)

    def start_capture(self):
        save_dir = self.folder_path.get()
        if not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir)
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể tạo thư mục:\n{e}")
                return

        self.is_capturing = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.status_label.config(text="Trạng thái: ĐANG CHỤP...", fg="blue")

        threading.Thread(target=self._save_worker, daemon=True).start()

    def stop_capture(self):
        self.is_capturing = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.status_label.config(text="Trạng thái: ĐÃ DỪNG", fg="red")

    def _save_worker(self):
        current_idx = self.start_index.get()
        target_count = self.total_images.get()
        delay = self.interval.get()
        save_dir = self.folder_path.get()
        count = 0

        while self.is_capturing and count < target_count:
            frame_raw = None
            with self.frame_lock:
                if self.latest_frame is not None:
                    # Lấy khung hình gốc (đã xác nhận là BGR)
                    frame_raw = self.latest_frame.copy()

            if frame_raw is not None:
                # ── LƯU TRỰC TIẾP KHÔNG LẬT ──
                # Ảnh gốc là BGR, cv2.imwrite cũng cần BGR -> khớp hoàn hảo!
                filename = os.path.join(save_dir, f"img_{current_idx:04d}.jpg")
                cv2.imwrite(filename, frame_raw)
                
                count += 1
                current_idx += 1
                self.status_label.config(text=f"Đang chụp: {count}/{target_count} (Lưu: img_{current_idx-1:04d}.jpg)")
            
            time.sleep(delay)

        if count >= target_count:
            self.status_label.config(text=f"Hoàn thành! Đã chụp đủ {count} tấm.", fg="green")
            self.is_capturing = False
            self.btn_start.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)

    def on_close(self):
        self.is_capturing = False
        self.camera_running = False
        if self.cam is not None:
            self.cam.stop()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = DataCollectorApp(root)
    root.mainloop()