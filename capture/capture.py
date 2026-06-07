import cv2
import os
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import sys

# ── CẤU HÌNH CAMERA ────────
CAM_W = 320
CAM_H = 240

# ── PRESET ÁNH SÁNG ────────
# run_yolo.py dùng ExposureTime=25000, Gain=4.0 để detect lane/đèn
# nhưng quá tối cho biển báo và người → dùng preset bên dưới khi chụp dataset
PRESETS = {
    "🏠 Trong nhà (đèn trần)": {
        "ExposureTime": 33000,   # ~1/30s — đủ sáng cho đèn huỳnh quang/LED
        "AnalogueGain": 2.0,
        "AwbEnable": True,       # AWB tự điều chỉnh theo bóng đèn trong nhà
        "AeEnable": False,
        "desc": "Phòng học, hành lang, đèn LED"
    },
    "🌤 Ngoài trời (bóng râm)": {
        "ExposureTime": 12000,   # ~1/83s
        "AnalogueGain": 1.5,
        "AwbEnable": True,
        "AeEnable": False,
        "desc": "Dưới mái hiên, trong bóng cây"
    },
    "☀ Ngoài trời (có nắng)": {
        "ExposureTime": 5000,    # ~1/200s
        "AnalogueGain": 1.0,
        "AeEnable": False,
        "AwbEnable": True,
        "desc": "Ánh sáng trực tiếp ban ngày"
    },
    "🌙 Tương tự run_yolo (tối)": {
        "ExposureTime": 25000,
        "AnalogueGain": 4.0,
        "AeEnable": False,
        "AwbEnable": False,
        "ColourGains": (1.6, 1.4),
        "desc": "Giống hệt run_yolo — chỉ dùng khi train lane/đèn"
    },
    "🔧 Tự động hoàn toàn (AE+AWB)": {
        "AeEnable": True,
        "AwbEnable": True,
        "desc": "Camera tự chỉnh — phù hợp mọi môi trường"
    },
}

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
        self.root.title("Thu thập Dataset - Chọn mức sáng phù hợp")
        self.root.geometry("520x440")

        self.is_capturing = False
        self.save_thread  = None
        self.cam          = None
        self.camera_running = False
        self.latest_frame   = None
        self.frame_lock     = threading.Lock()
        self.tft            = TFTDisplay()

        # --- Preset ánh sáng ---
        tk.Label(root, text="Môi trường chụp ảnh:", font=("Arial", 9, "bold")).grid(
            row=0, column=0, sticky="w", padx=10, pady=(10, 2))

        self.preset_var = tk.StringVar()
        preset_names = list(PRESETS.keys())
        # Mặc định: trong nhà — phù hợp nhất để capture biển báo + người
        self.preset_var.set(preset_names[0])

        self.preset_combo = ttk.Combobox(
            root, textvariable=self.preset_var,
            values=preset_names, width=38, state="readonly")
        self.preset_combo.grid(row=1, column=0, columnspan=3, padx=10, pady=2, sticky="w")
        self.preset_combo.bind("<<ComboboxSelected>>", self._on_preset_change)

        self.preset_desc = tk.Label(root, text="", fg="gray", font=("Arial", 8, "italic"))
        self.preset_desc.grid(row=2, column=0, columnspan=3, sticky="w", padx=12)

        # Hiển thị thông số hiện tại
        self.params_label = tk.Label(root, text="", fg="#555", font=("Courier", 8))
        self.params_label.grid(row=3, column=0, columnspan=3, sticky="w", padx=12, pady=(0, 6))

        # Separator
        ttk.Separator(root, orient="horizontal").grid(
            row=4, column=0, columnspan=3, sticky="ew", padx=10, pady=4)

        # --- Cấu hình lưu file ---
        tk.Label(root, text="Thư mục lưu:").grid(row=5, column=0, sticky="w", padx=10, pady=4)
        self.folder_path = tk.StringVar(value="/home/pi/dataset")
        tk.Entry(root, textvariable=self.folder_path, width=25).grid(row=5, column=1, pady=4)
        tk.Button(root, text="Duyệt...", command=self.browse_folder).grid(row=5, column=2, padx=5)

        tk.Label(root, text="Bắt đầu đánh số từ:").grid(row=6, column=0, sticky="w", padx=10, pady=4)
        self.start_index = tk.IntVar(value=0)
        tk.Entry(root, textvariable=self.start_index, width=15).grid(row=6, column=1, sticky="w", pady=4)

        tk.Label(root, text="Số lượng ảnh cần chụp:").grid(row=7, column=0, sticky="w", padx=10, pady=4)
        self.total_images = tk.IntVar(value=100)
        tk.Entry(root, textvariable=self.total_images, width=15).grid(row=7, column=1, sticky="w", pady=4)

        tk.Label(root, text="Giãn cách (giây/tấm):").grid(row=8, column=0, sticky="w", padx=10, pady=4)
        self.interval = tk.DoubleVar(value=1.0)
        tk.Entry(root, textvariable=self.interval, width=15).grid(row=8, column=1, sticky="w", pady=4)

        # --- Trạng thái & nút ---
        self.status_label = tk.Label(
            root, text="Đang khởi động Camera...", fg="blue",
            font=("Arial", 10, "bold"))
        self.status_label.grid(row=9, column=0, columnspan=3, pady=10)

        self.btn_apply = tk.Button(
            root, text="⚙ ÁP DỤNG PRESET", bg="#1565C0", fg="white",
            font=("Arial", 9, "bold"), command=self._apply_preset_now,
            state=tk.DISABLED)
        self.btn_apply.grid(row=10, column=0, pady=4, padx=8, sticky="ew")

        self.btn_start = tk.Button(
            root, text="▶ BẮT ĐẦU CHỤP", bg="green", fg="white",
            font=("Arial", 10, "bold"), command=self.start_capture,
            state=tk.DISABLED)
        self.btn_start.grid(row=10, column=1, pady=4, padx=4, sticky="ew")

        self.btn_stop = tk.Button(
            root, text="⏹ DỪNG LẠI", bg="red", fg="white",
            font=("Arial", 10, "bold"), command=self.stop_capture,
            state=tk.DISABLED)
        self.btn_stop.grid(row=10, column=2, pady=4, padx=4, sticky="ew")

        self._update_preset_display()
        self.root.after(100, self.init_camera)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ── Preset helpers ──────────────────────────────────────────────

    def _get_current_preset(self) -> dict:
        return PRESETS[self.preset_var.get()]

    def _on_preset_change(self, _event=None):
        self._update_preset_display()

    def _update_preset_display(self):
        p = self._get_current_preset()
        self.preset_desc.config(text=p.get("desc", ""))
        # Tạo chuỗi tóm tắt thông số
        parts = []
        if not p.get("AeEnable", False):
            parts.append(f"Exposure={p.get('ExposureTime', '—')}μs")
            parts.append(f"Gain={p.get('AnalogueGain', '—')}")
        else:
            parts.append("AE=Auto")
        parts.append("AWB=Auto" if p.get("AwbEnable", True) else
                     f"AWB=Fixed{p.get('ColourGains', '')}")
        self.params_label.config(text="  " + "  |  ".join(parts))

    def _apply_preset_now(self):
        """Áp dụng preset lên camera đang chạy mà không cần restart."""
        if self.cam is None:
            return
        p = self._get_current_preset()
        controls = {}
        if p.get("AeEnable", False):
            controls["AeEnable"] = True
        else:
            controls["AeEnable"] = False
            controls["ExposureTime"] = p["ExposureTime"]
            controls["AnalogueGain"] = p["AnalogueGain"]

        controls["AwbEnable"] = p.get("AwbEnable", True)
        if not p.get("AwbEnable", True) and "ColourGains" in p:
            controls["ColourGains"] = p["ColourGains"]

        try:
            self.cam.set_controls(controls)
            # Chờ camera ổn định sau khi đổi thông số
            time.sleep(0.5)
            preset_name = self.preset_var.get().split(" ", 1)[-1]
            self.status_label.config(
                text=f"✓ Đã áp dụng: {preset_name}", fg="#0D6E0D")
        except Exception as e:
            self.status_label.config(text=f"Lỗi set_controls: {e}", fg="red")

    # ── Camera init ─────────────────────────────────────────────────

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

            # Áp dụng preset mặc định (trong nhà)
            self._apply_preset_now()
            time.sleep(0.3)

            self.camera_running = True
            threading.Thread(target=self._capture_worker, daemon=True).start()
            threading.Thread(target=self._display_worker, daemon=True).start()

            self.status_label.config(
                text=f"Camera OK! ({CAM_W}x{CAM_H}) — Chọn preset rồi nhấn ÁP DỤNG",
                fg="green")
            self.btn_start.config(state=tk.NORMAL)
            self.btn_apply.config(state=tk.NORMAL)

        except Exception as e:
            self.status_label.config(text=f"Lỗi Camera: {e}", fg="red")

    # ── Worker threads ───────────────────────────────────────────────

    def _capture_worker(self):
        while self.camera_running:
            try:
                # picamera2 format="RGB888" thực ra trả về dữ liệu theo thứ tự BGR
                # (quirk của picamera2) → lưu thẳng vào latest_frame dưới dạng BGR
                with self.frame_lock:
                    self.latest_frame = self.cam.capture_array()
            except Exception:
                pass
            time.sleep(0.02)

    def _display_worker(self):
        while self.camera_running:
            frame_bgr = None
            with self.frame_lock:
                if self.latest_frame is not None:
                    frame_bgr = self.latest_frame.copy()
            if frame_bgr is not None and self.tft is not None and self.tft.ok:
                # TFT.show() nhận RGB → phải convert BGR→RGB trước khi gửi lên màn hình
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                self.tft.show(frame_rgb)
            time.sleep(0.03)

    # ── UI callbacks ─────────────────────────────────────────────────

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
        self.btn_apply.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.status_label.config(text="Trạng thái: ĐANG CHỤP...", fg="blue")
        threading.Thread(target=self._save_worker, daemon=True).start()

    def stop_capture(self):
        self.is_capturing = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_apply.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.status_label.config(text="Trạng thái: ĐÃ DỪNG", fg="red")

    def _save_worker(self):
        current_idx  = self.start_index.get()
        target_count = self.total_images.get()
        delay        = self.interval.get()
        save_dir     = self.folder_path.get()
        count        = 0

        while self.is_capturing and count < target_count:
            frame_raw = None
            with self.frame_lock:
                if self.latest_frame is not None:
                    frame_raw = self.latest_frame.copy()

            if frame_raw is not None:
                # latest_frame đã là BGR (picamera2 quirk) → cv2.imwrite cần BGR → lưu thẳng
                filename = os.path.join(save_dir, f"img_{current_idx:04d}.jpg")
                cv2.imwrite(filename, frame_raw)

                count        += 1
                current_idx  += 1
                self.status_label.config(
                    text=f"Đang chụp: {count}/{target_count} (img_{current_idx-1:04d}.jpg)")

            time.sleep(delay)

        if count >= target_count:
            self.status_label.config(
                text=f"✓ Hoàn thành! Đã chụp đủ {count} tấm.", fg="green")
            self.is_capturing = False
            self.btn_start.config(state=tk.NORMAL)
            self.btn_apply.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)

    def on_close(self):
        self.is_capturing   = False
        self.camera_running = False
        if self.cam is not None:
            self.cam.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app  = DataCollectorApp(root)
    root.mainloop()
