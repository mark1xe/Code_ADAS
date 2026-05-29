#!/usr/bin/env python3
"""
cv_lane_driver.py — Tự lái bằng OpenCV truyền thống (Canny + Hough)
=====================================================================
Không cần model ONNX. Dùng edge detection + Hough lines để tìm làn.
Hoạt động ngay trên bất kỳ track nào có vạch kẻ hoặc rìa đường.

Chạy trên RPi:
    python3 cv_lane_driver.py                   # chạy bình thường
    python3 cv_lane_driver.py --auto-start       # không cần bấm A
    python3 cv_lane_driver.py --speed 10 --debug # lưu frame debug

Protocol UART → ESP32 (giống rpi_ai_driver.py):
    "F,<steer -100..+100>,<speed>\n"
    "S,0,0\n"
"""

import sys, os, time, threading, argparse
import numpy as np
import cv2
import serial

# ── UART ──────────────────────────────────────────────────────────────────
UART_PORT  = "/dev/serial0"
UART_BAUD  = 115200
SEND_HZ    = 10

# ── Camera ────────────────────────────────────────────────────────────────
CAM_W = 320
CAM_H = 240   # landscape 4:3
# ── Camera exposure (khoá để tránh auto-exposure rối loạn khi gặp ánh sáng chói) ───
# Đặt CAM_FIXED_EXPOSURE = 0 để dùng auto-exposure (mặc định)
CAM_FIXED_EXPOSURE = 20000  # micro-giây (20 ms). Tăng nếu ảnh quá tối, giảm nếu bị lòe.
CAM_ANALOGUE_GAIN  = 4.0    # gain cố định (1.0–8.0). Tăng nếu vạch quá tối.
CAM_FIXED_AWB      = True   # khoá white-balance để màu đường không bị đổi
CAM_AWB_GAINS      = (1.6, 1.4)  # (red_gain, blue_gain) — ánh sáng trong nhà
# ── Điều khiển ────────────────────────────────────────────────────────────
DRIVE_SPEED  = 12
SLOW_SPEED   = 6
SMOOTH_ALPHA = 0.35   # EMA steer — giảm để phản hồi nhanh hơn (ít lag hơn)
DEADBAND     = 0.04   # |offset| < 4% → đi thẳng
CAM_BIAS     = 0.0    # Bù camera lệch tâm: dương=camera lệch phải, âm=lệch trái
                       # Nếu xe luôn lệch trái → tăng CAM_BIAS, lệch phải → giảm

# ── TFT SPI display ──────────────────────────────────────────────────────
TFT_W  = 320
TFT_H  = 240
TFT_FB = "/dev/fb1"

# ── CV params ─────────────────────────────────────────────────────────────
# Vùng quan sát: chỉ xử lý phần dưới của ảnh (nơi có làn đường)
ROI_TOP    = 0.45   # bắt đầu từ 45% chiều cao (bỏ phần trời/nền)
CANNY_LOW  = 30
CANNY_HIGH = 90
HOUGH_THRESHOLD  = 30   # tăng để bỏ qua đường ngắn/noise (cũ: 22)
HOUGH_MIN_LENGTH = 50   # chỉ nhận đường dài → vạch làn thật (cũ: 33)
HOUGH_MAX_GAP    = 25   # khe hở tối đa trong 1 đường

# ── Chống ánh sáng chói (Anti-Glare) ────────────────────────────────────
# Chỉ dùng CLAHE + temporal filter — không lọc màu để tránh mất lane
CLAHE_CLIP      = 3.5   # giới hạn clip CLAHE (cân bằng sáng cục bộ — tăng khi ảnh tối)
MAX_OFFSET_JUMP = 0.70  # chỉ loại spike cực nhanh (glare 1 frame); đủ rộng để xe có thể vào cua

# ── Shared state ──────────────────────────────────────────────────────────
_uart_steer        = 0
_uart_speed        = 0
_uart_running      = False
_uart_lock         = threading.Lock()
_stop_ev           = threading.Event()
_latest_frame      = None
_frame_lock        = threading.Lock()
_prev_valid_offset = 0.0   # temporal filter: lưu offset hợp lệ gần nhất


# ══════════════════════════════════════════════════════════════════════════
#  TFT SPI DISPLAY
# ══════════════════════════════════════════════════════════════════════════
class TFTDisplay:
    """ILI9341 320×240. Backend: luma.lcd → /dev/fb1."""
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
            print("[TFT] luma.lcd chưa cài → pip install luma.lcd pillow")
        except Exception as e:
            print(f"[TFT] luma.lcd lỗi ({e}) → thử /dev/fb1")
        try:
            self._fb = open(TFT_FB, "wb")
            print(f"TFT: framebuffer {TFT_FB} OK")
        except OSError as e:
            print(f"[TFT] Không dùng được ({e})")

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
            except Exception as e:
                print(f"\n[TFT] {e}"); self._luma = None
        elif self._fb is not None:
            try:
                r = disp[:,:,0].astype(np.uint16)
                g = disp[:,:,1].astype(np.uint16)
                b = disp[:,:,2].astype(np.uint16)
                rgb565 = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
                self._fb.seek(0)
                self._fb.write(rgb565.astype(np.uint16).tobytes())
                self._fb.flush()
            except OSError as e:
                print(f"\n[TFT] fb lỗi: {e}"); self._fb = None


# ══════════════════════════════════════════════════════════════════════════
#  LANE DETECTION bằng OpenCV
# ══════════════════════════════════════════════════════════════════════════
def detect_lane_offset(frame_rgb: np.ndarray, debug: bool = False):
    """
    Phát hiện làn đường bằng Canny + HoughLinesP.
    Trả về (offset, debug_frame):
      offset: float [-1, 1] — dương=lệch phải, âm=lệch trái, None=không thấy làn
      debug_frame: ảnh BGR với overlay vẽ các đường detect được
    """
    H, W = frame_rgb.shape[:2]
    roi_y = int(H * ROI_TOP)

    # ── Tiền xử lý ────────────────────────────────────────────────────
    gray  = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    # CLAHE: cân bằng sáng cục bộ → giảm vùng quá chói mà không mất vạch kẻ
    _clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(8, 8))
    gray   = _clahe.apply(gray)
    blur  = cv2.GaussianBlur(gray, (9, 9), 0)   # kernel lớn hơn → xoá noise bề mặt
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

    # ── Mask ROI: hình thang (bỏ phần trên) ──────────────────────────
    mask = np.zeros_like(edges)
    roi_pts = np.array([[
        (0,       H),
        (0,       roi_y),
        (W,       roi_y),
        (W,       H),
    ]], dtype=np.int32)
    cv2.fillPoly(mask, roi_pts, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # ── Hough Lines ───────────────────────────────────────────────────
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1, theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LENGTH,
        maxLineGap=HOUGH_MAX_GAP
    )

    dbg = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR).copy() if debug else None

    if lines is None:
        return None, dbg

    # ── Phân loại trái / phải ─────────────────────────────────────────
    left_xs  = []
    right_xs = []
    cx = W / 2.0

    neg_xs = []   # slope < 0
    pos_xs = []   # slope > 0
    neg_lines = []
    pos_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        dy = y2 - y1
        dx = x2 - x1
        if abs(dy) < 1:
            continue
        slope = dy / dx
        if abs(slope) < 0.25:
            continue

        x_bot = x1 + (H - y1) * dx / dy
        x_bot = float(np.clip(x_bot, 0.0, float(W)))

        if slope < 0:
            neg_xs.append(x_bot)
            neg_lines.append((x1, y1, x2, y2))
        else:
            pos_xs.append(x_bot)
            pos_lines.append((x1, y1, x2, y2))

    if not neg_xs and not pos_xs:
        # Tất cả lines bị filter bỏ (slope quá ngang)
        return None, dbg

    if neg_xs and pos_xs:
        # Đường thẳng: 2 slope khác nhau → phân loại bình thường
        left_xs  = neg_xs
        right_xs = pos_xs
        line_pairs = [(l, (0, 255, 0)) for l in neg_lines] + \
                     [(l, (0, 100, 255)) for l in pos_lines]
    else:
        # Góc cua: chỉ có 1 loại slope
        # Dùng slope sign để ÉP hướng steer đúng:
        #   slope âm (tất cả /)  → road cua TRÁI  → ép lane_center < cx
        #   slope dương (tất cả \) → road cua PHẢI → ép lane_center > cx
        all_xbots = neg_xs if neg_xs else pos_xs
        all_lines = neg_lines if neg_lines else pos_lines
        mean_x = float(np.mean(all_xbots))
        if neg_xs:
            # Cua trái → ép lane_center ≤ 70% cx (luôn steer trái)
            lane_center = float(min(mean_x, cx * 0.70))
        else:
            # Cua phải → ép lane_center ≥ 130% cx (luôn steer phải)
            lane_center = float(max(mean_x, cx * 1.30))
        # Gán trực tiếp, bỏ qua has_left/has_right logic bên dưới
        offset_curve = float(np.clip((lane_center - cx) / cx, -1.0, 1.0))
        offset_curve = float(np.clip(offset_curve * 1.6, -1.0, 1.0))   # boost cua
        if debug and dbg is not None:
            for (x1, y1, x2, y2) in all_lines:
                cv2.line(dbg, (x1, y1), (x2, y2), (0, 200, 200), 2)
            cv2.line(dbg, (int(lane_center), H-10), (int(lane_center), roi_y), (0, 255, 255), 2)
            cv2.line(dbg, (int(cx), H-10), (int(cx), roi_y), (128, 128, 128), 1)
            cv2.line(dbg, (0, roi_y), (W, roi_y), (50, 50, 50), 1)
            tag = f"off={offset_curve:+.3f} CURVE {'L' if neg_xs else 'R'}"
            cv2.putText(dbg, tag, (5, H-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
        return float(np.clip(offset_curve - CAM_BIAS, -1.0, 1.0)), dbg

    if debug and dbg is not None:
        for (x1, y1, x2, y2), color in line_pairs:
            cv2.line(dbg, (x1, y1), (x2, y2), color, 2)

    # ── Tính offset ───────────────────────────────────────────────────
    has_left  = len(left_xs)  > 0
    has_right = len(right_xs) > 0

    single_lane = False
    if has_left and has_right:
        xl = float(np.mean(left_xs))
        xr = float(np.mean(right_xs))
        lane_center = (xl + xr) / 2.0
    elif has_left:
        xl = float(np.mean(left_xs))
        lane_center = xl + W * 0.35   # ước tính: làn rộng ~70% ảnh
        single_lane = True
    elif has_right:
        xr = float(np.mean(right_xs))
        lane_center = xr - W * 0.35
        single_lane = True
    else:
        return None, dbg

    offset = float(np.clip((lane_center - cx) / cx, -1.0, 1.0))

    # Khi chỉ thấy 1 vạch (góc cua hoặc mất vạch) → tăng offset để xe rẽ mạnh hơn
    if single_lane:
        offset = float(np.clip(offset * 1.6, -1.0, 1.0))

    # Bù lệch camera (CAM_BIAS): điều chỉnh nếu xe luôn lệch 1 phía trên đường thẳng
    offset = float(np.clip(offset - CAM_BIAS, -1.0, 1.0))

    if debug and dbg is not None:
        # Vẽ đường tâm làn
        cv2.line(dbg, (int(lane_center), H - 10), (int(lane_center), roi_y), (0, 255, 255), 2)
        cv2.line(dbg, (int(cx), H - 10), (int(cx), roi_y), (128, 128, 128), 1)
        cv2.line(dbg, (0, roi_y), (W, roi_y), (50, 50, 50), 1)
        tag = f"off={offset:+.3f} L={has_left} R={has_right}"
        cv2.putText(dbg, tag, (5, H - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    return offset, dbg


def offset_to_steer(offset: float | None) -> int:
    """offset [-1,1] → steer_val [-100,100], phi tuyến với deadband."""
    if offset is None or abs(offset) < DEADBAND:
        return 0
    sign = 1 if offset > 0 else -1
    x = min((abs(offset) - DEADBAND) / (1.0 - DEADBAND), 1.0)
    mag = 5.0 + 95.0 * x   # tuyến tính — mạnh hơn ở offset vừa (cua nhẹ)
    # offset > 0 → lane center ở phải ảnh → xe trôi trái → cần lái phải (+)
    # offset < 0 → lane center ở trái ảnh → xe trôi phải → cần lái trái (-)
    return int(round(sign * min(mag, 100.0)))


# ══════════════════════════════════════════════════════════════════════════
#  THREADS
# ══════════════════════════════════════════════════════════════════════════
def _uart_thread(uart, stop_ev: threading.Event) -> None:
    interval = 1.0 / SEND_HZ
    while not stop_ev.is_set():
        t0 = time.time()
        with _uart_lock:
            steer, speed, running = _uart_steer, _uart_speed, _uart_running
        try:
            if running:
                uart.write(f"F,{steer},{speed}\n".encode())
            else:
                uart.write(b"S,0,0\n")
        except Exception:
            pass
        rem = interval - (time.time() - t0)
        if rem > 0:
            time.sleep(rem)


def _capture_thread(cam, stop_ev: threading.Event) -> None:
    global _latest_frame
    while not stop_ev.is_set():
        try:
            with _frame_lock:
                _latest_frame = cam.capture_array()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    global ROI_TOP, CANNY_LOW, CANNY_HIGH, SMOOTH_ALPHA   # khai báo trước mọi usage
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed",       type=int,   default=DRIVE_SPEED)
    ap.add_argument("--port",        default=UART_PORT)
    ap.add_argument("--smooth",      type=float, default=SMOOTH_ALPHA)
    ap.add_argument("--canny-low",   type=int,   default=CANNY_LOW)
    ap.add_argument("--canny-high",  type=int,   default=CANNY_HIGH)
    ap.add_argument("--roi-top",     type=float, default=ROI_TOP,
                    help="ROI bắt đầu từ N*H (0=đỉnh, 1=đáy, mặc định 0.45)")
    ap.add_argument("--debug",       action="store_true",
                    help="Lưu frame debug vào /tmp/cv_debug_NNN.jpg")
    ap.add_argument("--auto-start",  action="store_true",
                    help="Tự lái ngay (không cần bấm A)")
    ap.add_argument("--no-camera",   action="store_true",
                    help="Test không camera (dùng ảnh trắng)")
    ap.add_argument("--no-display",  action="store_true",
                    help="Tắt TFT SPI display")
    ap.add_argument("--exposure",    type=int,   default=CAM_FIXED_EXPOSURE,
                    help="Thời gian phơi sáng cố định (μs). 0 = dùng auto-exposure.")
    ap.add_argument("--gain",        type=float, default=CAM_ANALOGUE_GAIN,
                    help="Analogue gain cố định (1.0–8.0). Chỉ dùng khi --exposure > 0.")
    ap.add_argument("--auto-awb",    action="store_true",
                    help="Dùng auto white-balance (mặc định: khoá AWB)")
    args = ap.parse_args()

    # Ghi đè globals từ args
    ROI_TOP     = args.roi_top
    CANNY_LOW   = args.canny_low
    CANNY_HIGH  = args.canny_high
    SMOOTH_ALPHA = args.smooth

    # ── UART ──────────────────────────────────────────────────────────
    uart = None
    for p in [args.port, "/dev/ttyAMA0", "/dev/ttyS0"]:
        try:
            uart = serial.Serial(p, UART_BAUD, timeout=0.05)
            print(f"UART: {p} @ {UART_BAUD}")
            break
        except serial.SerialException as e:
            print(f"  [{p}] {e}")
    if uart is None:
        print("[LỖI] Không mở được UART"); sys.exit(1)

    # ── Camera ────────────────────────────────────────────────────────
    cam = None
    if not args.no_camera:
        from picamera2 import Picamera2
        cam = Picamera2()
        cam.configure(cam.create_preview_configuration(
            main={"format": "RGB888", "size": (CAM_W, CAM_H)}))
        cam.start()
        time.sleep(1.0)   # chờ AE ổn định trước khi khoá

        # ── Khoá exposure để chống ánh sáng chói ───────────────────────
        # Auto-exposure mặc định sẽ tự tối/sáng theo đèn trong frame
        # → làm vạch kẻ mất khi ánh đèn chiếu vào
        _controls = {}
        if args.exposure > 0:
            _controls["AeEnable"]       = False
            _controls["ExposureTime"]   = args.exposure
            _controls["AnalogueGain"]   = args.gain
            print(f"Camera: exposure khoá {args.exposure}μs, gain={args.gain}")
        else:
            print("Camera: auto-exposure (có thể bị lòe ánh đèn)")
        if not args.auto_awb:
            _controls["AwbEnable"]      = False
            _controls["ColourGains"]    = CAM_AWB_GAINS
            print(f"Camera: AWB khoá gains={CAM_AWB_GAINS}")
        if _controls:
            cam.set_controls(_controls)
            time.sleep(0.3)   # chờ controls áp dụng

        print(f"Camera OK ({CAM_W}x{CAM_H})")
        threading.Thread(target=_capture_thread, args=(cam, _stop_ev),
                         daemon=True, name="capture").start()
        t_wait = time.time()
        while _latest_frame is None and time.time() - t_wait < 3.0:
            time.sleep(0.05)

    # ── TFT display ───────────────────────────────────────────────────
    tft = None
    if not args.no_display:
        tft = TFTDisplay()
        if not tft.ok:
            tft = None

    # ── UART thread ───────────────────────────────────────────────────
    threading.Thread(target=_uart_thread, args=(uart, _stop_ev),
                     daemon=True, name="uart").start()

    # ── Shared state global ───────────────────────────────────────────
    global _uart_steer, _uart_speed, _uart_running

    running      = bool(args.auto_start)
    smooth_steer = 0.0
    no_lane_cnt  = 0
    fps_t0       = time.time()
    fps_cnt      = 0
    fps          = 0.0
    debug_cnt    = 0

    with _uart_lock:
        _uart_running = running

    print("═" * 50)
    print("  CV LANE DRIVER  (OpenCV Hough)")
    print("═" * 50)
    print("  A + Enter = Bắt đầu  |  S + Enter = Dừng")
    print(f"  ROI top={ROI_TOP:.0%}  Canny {CANNY_LOW}/{CANNY_HIGH}")
    print("═" * 50)
    if running:
        print(">>> AUTO-START (--auto-start)")

    try:
        import select, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        has_tty = True
    except Exception:
        has_tty = False

    def read_key() -> str | None:
        if not has_tty:
            return None
        try:
            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.read(1).lower()
        except Exception:
            pass
        return None

    if has_tty:
        try:
            tty.setcbreak(sys.stdin.fileno())
        except Exception:
            pass

    try:
        while True:
            # ── Bàn phím ──────────────────────────────────────────────
            ch = read_key()
            if ch == 'a':
                running = True
                smooth_steer = 0.0
                with _uart_lock:
                    _uart_running = True
                print("\n>>> TỰ LÁI BẮT ĐẦU")
            elif ch == 's':
                running = False
                smooth_steer = 0.0
                with _uart_lock:
                    _uart_running = False
                    _uart_steer = 0; _uart_speed = 0
                print("\n>>> DỪNG")
            elif ch in ('q', '\x03'):
                break

            # ── Lấy frame ─────────────────────────────────────────────
            if cam is not None:
                with _frame_lock:
                    frame = _latest_frame
                if frame is None:
                    time.sleep(0.01); continue
            else:
                frame = np.ones((CAM_H, CAM_W, 3), dtype=np.uint8) * 200

            # ── CV detect ─────────────────────────────────────────────
            need_dbg = (tft is not None) or (args.debug and debug_cnt % 10 == 0)
            offset, dbg_img = detect_lane_offset(frame, debug=need_dbg)
            if args.debug and (debug_cnt % 10 == 0) and dbg_img is not None:
                cv2.imwrite(f"/tmp/cv_debug_{debug_cnt:04d}.jpg", dbg_img)

            # ── Temporal filter: loại offset nhảy đột ngột (anti-glare) ──
            global _prev_valid_offset
            if offset is not None:
                if abs(offset - _prev_valid_offset) > MAX_OFFSET_JUMP:
                    offset = None   # đột biến quá lớn → bỏ (glare thoáng qua)
                else:
                    _prev_valid_offset = offset
            # ── TFT: hiện ảnh debug mỗi 3 frame ─────────────────────
            if tft is not None and debug_cnt % 3 == 0:
                show_img = dbg_img if dbg_img is not None else cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                tft.show(cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB))
            debug_cnt += 1

            raw_steer = offset_to_steer(offset)
            smooth_steer = SMOOTH_ALPHA * raw_steer + (1 - SMOOTH_ALPHA) * smooth_steer
            steer_val = int(round(smooth_steer))

            lane_found = offset is not None
            no_lane_cnt = 0 if lane_found else no_lane_cnt + 1

            # ── Cập nhật UART state ───────────────────────────────────
            spd = args.speed if lane_found else SLOW_SPEED
            if no_lane_cnt > 20:
                spd = 0; steer_val = 0
            with _uart_lock:
                _uart_steer = steer_val
                _uart_speed = spd

            # ── FPS ───────────────────────────────────────────────────
            fps_cnt += 1
            if time.time() - fps_t0 >= 2.0:
                fps = fps_cnt / (time.time() - fps_t0)
                fps_cnt = 0; fps_t0 = time.time()

            off_str   = f"{offset:+.3f}" if lane_found else " N/A "
            state_tag = "RUN " if running else "STOP"
            lane_tag  = "LANE" if lane_found else "NOLN"
            print(f"\r[{state_tag}] {lane_tag}  fps={fps:4.1f}  "
                  f"off={off_str}  steer={steer_val:+4d}  "
                  f"noln={no_lane_cnt:3d}",
                  end="", flush=True)

    except KeyboardInterrupt:
        print("\nCtrl+C")
    finally:
        if has_tty:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except Exception:
                pass
        _stop_ev.set()
        with _uart_lock:
            _uart_running = False
        time.sleep(0.15)
        try:
            uart.write(b"S,0,0\n")
        except Exception:
            pass
        uart.close()
        if cam is not None:
            cam.stop()
        print("Đã dừng.")


if __name__ == "__main__":
    main()
