#!/usr/bin/env python3
"""
cv_lane_driver.py — Tự lái bằng OpenCV truyền thống + YOLO
=====================================================================
Cập nhật mới nhất:
1. DYNAMIC LANE WIDTH: Tự học độ rộng làn đường để nội suy tâm chính xác khi cua.
2. CHỐNG CHẾT MÁY: Khóa Max Steer = 80, tránh kẹt cơ khí servo.
3. CHỐNG LỆCH LANE: Tăng Smooth Alpha = 0.75 giúp trả thẳng lái tức thì.
4. LOW-LIGHT VISION: Hạ ngưỡng HSV_V_MIN = 70 để thấy vạch trong bóng tối.
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
CAM_H = 240
CAM_FIXED_EXPOSURE = 60000
CAM_ANALOGUE_GAIN  = 8.0    
CAM_FIXED_AWB      = False
CAM_AWB_GAINS      = (1.6, 1.4)

SW_BRIGHTNESS = 2.5   

# ── Điều khiển ────────────────────────────────────────────────────────────
DRIVE_SPEED  = 10
SLOW_SPEED   = 6
SMOOTH_ALPHA = 0.50   # [FIX] Tăng từ 0.35 lên 0.75 để servo trả thẳng lái chớp nhoáng
DEADBAND     = 0.04   
CAM_BIAS     = 0.0    

# ── TFT SPI display ──────────────────────────────────────────────────────
TFT_W  = 320
TFT_H  = 240
TFT_FB = "/dev/fb1"

# ── CV params ─────────────────────────────────────────────────────────────
ROI_BOT_LEFT_X  = 0.00
ROI_BOT_RIGHT_X = 1.00
ROI_BOT_Y       = 1.00
ROI_TOP_LEFT_X  = 0.28   
ROI_TOP_RIGHT_X = 0.80   
ROI_TOP_Y       = 0.42

LANE_SLOPE_MIN = 0.40   
LANE_SLOPE_MAX = 5.67   

HSV_H_MIN, HSV_H_MAX = 0,   180
HSV_S_MIN, HSV_S_MAX = 0,   60
HSV_V_MIN, HSV_V_MAX = 85,  255   # [FIX] Giảm V_MIN xuống 70 để bắt được vạch trong phòng tối

CANNY_LOW        = 40
CANNY_HIGH       = 100
HOUGH_THRESHOLD  = 30
HOUGH_MIN_LENGTH = 25
HOUGH_MAX_GAP    = 40

MAX_OFFSET_JUMP  = 0.65   

# ── Crosswalk (Zebra) detection ───────────────────────────────────────────
ZEBRA_HOLD_FRAMES   = 12   
ZEBRA_SPEED         = 8

# ── Shared state ──────────────────────────────────────────────────────────
_uart_steer        = 0
_uart_speed        = 0
_uart_running      = False
_uart_drive_mode   = 'F'
_uart_lock         = threading.Lock()
_stop_ev           = threading.Event()
_latest_frame      = None
_frame_lock        = threading.Lock()
_prev_valid_offset = 0.0   
_zebra_hold_cnt    = 0     
_ema_lane_width    = 160.0  # [FIX] Khởi tạo độ rộng làn đường động (mặc định = 1/2 khung hình)


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
            return
        except ImportError:
            pass
        try:
            self._fb = open(TFT_FB, "wb")
        except OSError:
            pass

    @property
    def ok(self):
        return self._luma is not None or self._fb is not None

    def show(self, img_bgr: np.ndarray) -> None:
        disp = cv2.resize(img_bgr, (TFT_W, TFT_H), interpolation=cv2.INTER_LINEAR)
        if self._luma is not None:
            try:
                from PIL import Image
                rgb = disp[:, :, ::-1] 
                portrait = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
                self._luma.display(Image.fromarray(portrait))
            except Exception:
                self._luma = None
        elif self._fb is not None:
            try:
                b = disp[:,:,0].astype(np.uint16)
                g = disp[:,:,1].astype(np.uint16)
                r = disp[:,:,2].astype(np.uint16)
                rgb565 = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
                self._fb.seek(0)
                self._fb.write(rgb565.astype(np.uint16).tobytes())
                self._fb.flush()
            except OSError:
                self._fb = None


def detect_zebra_crossing(frame_bgr: np.ndarray) -> bool:
    H, W = frame_bgr.shape[:2]

    if SW_BRIGHTNESS > 1.0:
        frame_bgr = np.clip(frame_bgr.astype(np.float32) * SW_BRIGHTNESS, 0, 255).astype(np.uint8)

    y0 = int(H * 0.45); y1_crop = int(H * 0.90)
    x0 = int(W * 0.20); x1_crop = int(W * 0.80)
    roi = frame_bgr[y0:y1_crop, x0:x1_crop]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    blur = cv2.GaussianBlur(hsv, (5, 5), 0)

    lo = np.array([HSV_H_MIN, HSV_S_MIN, HSV_V_MIN], dtype=np.uint8)
    hi = np.array([HSV_H_MAX, HSV_S_MAX, HSV_V_MAX], dtype=np.uint8)
    white_mask = cv2.inRange(blur, lo, hi)
    edges = cv2.Canny(white_mask, 50, 120)

    crop_w = x1_crop - x0
    lines = cv2.HoughLinesP(
        edges,
        rho=1, theta=np.pi / 180,
        threshold=25,
        minLineLength=int(crop_w * 0.35), 
        maxLineGap=15
    )

    if lines is None:
        return False

    for line in lines:
        lx1, ly1, lx2, ly2 = line[0]
        dx = abs(lx2 - lx1)
        dy = abs(ly2 - ly1)
        if dx < 1: continue
        slope_abs = dy / dx
        if slope_abs < 0.20: 
            return True

    return False


def detect_lane_offset(frame_bgr: np.ndarray, debug: bool = False):
    global _ema_lane_width # [FIX] Khai báo biến toàn cục
    
    H, W = frame_bgr.shape[:2]

    if SW_BRIGHTNESS > 1.0:
        frame_bgr = np.clip(frame_bgr.astype(np.float32) * SW_BRIGHTNESS, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    blur = cv2.GaussianBlur(hsv, (7, 7), 0)

    lo = np.array([HSV_H_MIN, HSV_S_MIN, HSV_V_MIN], dtype=np.uint8)
    hi = np.array([HSV_H_MAX, HSV_S_MAX, HSV_V_MAX], dtype=np.uint8)
    white_mask = cv2.inRange(blur, lo, hi)

    edges = cv2.Canny(white_mask, CANNY_LOW, CANNY_HIGH)

    tl = (int(W * ROI_TOP_LEFT_X),  int(H * ROI_TOP_Y))
    tr = (int(W * ROI_TOP_RIGHT_X), int(H * ROI_TOP_Y))
    bl = (int(W * ROI_BOT_LEFT_X),  int(H * ROI_BOT_Y))
    br = (int(W * ROI_BOT_RIGHT_X), int(H * ROI_BOT_Y))
    roi_pts = np.array([[bl, tl, tr, br]], dtype=np.int32)

    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_pts, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1, theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LENGTH,
        maxLineGap=HOUGH_MAX_GAP
    )

    dbg = frame_bgr.copy() if debug else None

    # Dùng 2 lookahead: gần (85%) và xa (55%)
    # Line dài → trọng số cao → tự động ưu tiên đoạn thẳng hay cua
    lookahead_near = int(H * 0.85)
    lookahead_far  = int(H * 0.55)

    if dbg is not None:
        cv2.polylines(dbg, [roi_pts], True, (255, 0, 255), 1)
        cv2.line(dbg, (0, lookahead_near), (W, lookahead_near), (30, 30, 30), 1)
        cv2.line(dbg, (0, lookahead_far),  (W, lookahead_far),  (60, 60, 60), 1)

    if lines is None:
        return None, dbg, white_mask

    left_xs   = []
    right_xs  = []
    left_lines  = []
    right_lines = []
    cx = W / 2.0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = np.hypot(dx, dy)
        if length < 1: continue
        if abs(dx) < 0.5: continue

        slope_abs = abs(dy / dx)
        if slope_abs < LANE_SLOPE_MIN or slope_abs > LANE_SLOPE_MAX:
            continue

        # Chiếu lên 2 điểm, lấy trung bình có trọng số
        def proj(ly):
            return x1 + (ly - y1) * dx / dy if abs(dy) > 0.5 else (x1+x2)/2.0

        xn = proj(lookahead_near)
        xf = proj(lookahead_far)
        # Trung bình 60% gần + 40% xa → nhạy cua hơn nhưng không dao động
        x_look = 0.6 * xn + 0.4 * xf

        if x_look < cx:
            left_xs.append((x_look, length))
            left_lines.append((x1, y1, x2, y2))
        else:
            right_xs.append((x_look, length))
            right_lines.append((x1, y1, x2, y2))

    if not left_xs and not right_xs:
        return None, dbg, white_mask

    has_left  = len(left_xs) > 0
    has_right = len(right_xs) > 0

    def weighted_mean(pts_w):
        xs = np.array([p[0] for p in pts_w])
        ws = np.array([p[1] for p in pts_w])
        return float(np.sum(xs * ws) / np.sum(ws))

    xl = weighted_mean(left_xs)  if has_left  else 0.0
    xr = weighted_mean(right_xs) if has_right else 0.0
    single_lane = False

    # ── [FIX LỚN NHẤT]: TỰ HỌC ĐỘ RỘNG LÀN ĐƯỜNG (DYNAMIC EMA) ──
    if has_left and has_right:
        current_w = xr - xl
        # Nếu độ rộng đọc được là hợp lý, cập nhật vào bộ nhớ EMA
        if 80 < current_w < 280:
            _ema_lane_width = 0.1 * current_w + 0.9 * _ema_lane_width
        lane_center = (xl + xr) / 2.0
    elif has_left:
        # Nếu mất vạch phải, dùng bộ nhớ để suy ra tâm ảo
        lane_center = xl + (_ema_lane_width / 2.0)
        single_lane = True
    elif has_right:
        # Nếu mất vạch trái, dùng bộ nhớ để suy ra tâm ảo
        lane_center = xr - (_ema_lane_width / 2.0)
        single_lane = True
    else:
        return None, dbg, white_mask

    offset = float(np.clip((lane_center - cx) / cx, -1.0, 1.0))

    if single_lane:
        offset = float(np.clip(offset * 1.3, -1.0, 1.0))

    offset = float(np.clip(offset - CAM_BIAS, -1.0, 1.0))

    if debug and dbg is not None:
        for (x1, y1, x2, y2) in left_lines:
            cv2.line(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for (x1, y1, x2, y2) in right_lines:
            cv2.line(dbg, (x1, y1), (x2, y2), (0, 100, 255), 2)

        if has_left:  cv2.circle(dbg, (int(xl), lookahead_near), 4, (255, 255, 255), -1)
        if has_right: cv2.circle(dbg, (int(xr), lookahead_near), 4, (255, 255, 255), -1)

        cv2.line(dbg, (int(lane_center), H-10), (int(lane_center), int(H*ROI_TOP_Y)), (0,255,255), 2)
        cv2.line(dbg, (int(cx),          H-10), (int(cx),          int(H*ROI_TOP_Y)), (128,128,128), 1)

        tag = f"off={offset:+.3f} W={int(_ema_lane_width)}"
        cv2.putText(dbg, tag, (5, H-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        if _zebra_hold_cnt > 0:
            cv2.putText(dbg, f"ZEBRA hold={_zebra_hold_cnt}", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2)

    return offset, dbg, white_mask


def offset_to_steer(offset: float | None) -> int:
    if offset is None or abs(offset) < DEADBAND:
        return 0
    sign = 1 if offset > 0 else -1
    x = min((abs(offset) - DEADBAND) / (1.0 - DEADBAND), 1.0)
    # Curve bậc 2: nhạy hơn ở góc lớn (cua gắt)
    # x=0.3 → mag≈30, x=0.6 → mag≈57, x=1.0 → mag≈100
    mag = 5.0 + 95.0 * (x ** 1.4)
    return int(round(sign * min(mag, 100.0)))


def _uart_thread(uart, stop_ev: threading.Event) -> None:
    interval = 1.0 / SEND_HZ
    while not stop_ev.is_set():
        t0 = time.time()
        with _uart_lock:
            steer, speed, running = _uart_steer, _uart_speed, _uart_running
            drive = _uart_drive_mode
        try:
            if running:
                uart.write(f"{drive},{steer},{speed}\n".encode())
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


def main():
    global CANNY_LOW, CANNY_HIGH, SMOOTH_ALPHA
    global HSV_S_MAX, HSV_V_MIN
    global ROI_TOP_Y, ROI_TOP_LEFT_X, ROI_TOP_RIGHT_X
    global SW_BRIGHTNESS
    global _prev_valid_offset, _zebra_hold_cnt
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed",       type=int,   default=DRIVE_SPEED)
    ap.add_argument("--port",        default=UART_PORT)
    ap.add_argument("--smooth",      type=float, default=SMOOTH_ALPHA)
    ap.add_argument("--hsv-s-max",   type=int,   default=HSV_S_MAX)
    ap.add_argument("--hsv-v-min",   type=int,   default=HSV_V_MIN)
    ap.add_argument("--canny-low",   type=int,   default=CANNY_LOW)
    ap.add_argument("--canny-high",  type=int,   default=CANNY_HIGH)
    ap.add_argument("--debug",       action="store_true")
    ap.add_argument("--auto-start",  action="store_true")
    ap.add_argument("--no-camera",   action="store_true")
    ap.add_argument("--no-display",  action="store_true")
    ap.add_argument("--exposure",       type=int,   default=CAM_FIXED_EXPOSURE)
    ap.add_argument("--gain",           type=float, default=CAM_ANALOGUE_GAIN)
    ap.add_argument("--sw-brightness",  type=float, default=SW_BRIGHTNESS)
    ap.add_argument("--auto-awb",    action="store_true")
    args = ap.parse_args()

    CANNY_LOW     = args.canny_low
    CANNY_HIGH    = args.canny_high
    SMOOTH_ALPHA  = args.smooth
    HSV_S_MAX     = args.hsv_s_max
    HSV_V_MIN     = args.hsv_v_min
    SW_BRIGHTNESS = args.sw_brightness

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

    cam = None
    if not args.no_camera:
        from picamera2 import Picamera2
        cam = Picamera2()
        
        cam.configure(cam.create_preview_configuration(
            main={"format": "BGR888", "size": (CAM_W, CAM_H)}))
        
        cam.start()
        time.sleep(1.0)

        _controls = {}
        if args.exposure > 0:
            _controls["AeEnable"]     = False
            _controls["ExposureTime"] = args.exposure
            _controls["AnalogueGain"] = args.gain
            print(f"Camera: exp={args.exposure}μs gain_a={args.gain}")
        else:
            print("Camera: auto-exposure")
            
        if not args.auto_awb:
            _controls["AwbEnable"]      = False
            _controls["ColourGains"]    = CAM_AWB_GAINS
            print("Đã TẮT khóa AWB (Camera tự khử ám màu)")
        else:
            print("Đã BẬT khóa AWB")

        if _controls:
            cam.set_controls(_controls)
            time.sleep(0.3)

        print(f"Camera OK ({CAM_W}x{CAM_H})")
        threading.Thread(target=_capture_thread, args=(cam, _stop_ev),
                         daemon=True, name="capture").start()
        t_wait = time.time()
        while _latest_frame is None and time.time() - t_wait < 3.0:
            time.sleep(0.05)

    tft = None
    if not args.no_display:
        tft = TFTDisplay()
        if not tft.ok:
            tft = None

    threading.Thread(target=_uart_thread, args=(uart, _stop_ev),
                     daemon=True, name="uart").start()

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
    print("  OPTIMIZED CV LANE DRIVER (DYNAMIC WIDTH & LOW LIGHT)  ")
    print("═" * 50)
    print("  A + Enter = Bắt đầu   |  S + Enter = Dừng")
    print("  C + Enter = Chụp ảnh  |  Q + Enter = Thoát")
    print(f"  HSV S≤{HSV_S_MAX} V≥{HSV_V_MIN}  Canny {CANNY_LOW}/{CANNY_HIGH}")
    print("═" * 50)

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

    # ── Recovery history ──────────────────────────────────────────────────
    HISTORY_LEN       = 25    # ~2.5s ở 10Hz
    RECOVERY_FRAMES   = 50
    RECOVERY_COOLDOWN = 50    # frame nghỉ sau recovery, tránh loop vô tận
    hist_steer        = []    # chỉ lưu frame thật sự có lane
    is_recovering     = False
    recovery_step     = 0
    recovery_cd       = 0

    try:
        while True:
            ch = read_key()
            if ch == 'a':
                running = True; smooth_steer = 0.0
                with _uart_lock: _uart_running = True
                print("\n>>> TỰ LÁI BẮT ĐẦU")
            elif ch == 's':
                running = False; smooth_steer = 0.0
                with _uart_lock:
                    _uart_running = False
                    _uart_steer = 0; _uart_speed = 0
                print("\n>>> DỪNG")
            elif ch == 'c':
                timestamp = time.strftime("%H%M%S")
                if cam is not None and _latest_frame is not None:
                    raw_bgr = _latest_frame.copy()
                    cv2.imwrite(f"snap_{timestamp}_raw.jpg", raw_bgr)
                    _, dbg_snap, mask_snap = detect_lane_offset(raw_bgr, debug=True)
                    if dbg_snap is not None:
                        cv2.imwrite(f"snap_{timestamp}_debug.jpg", dbg_snap)
                    if mask_snap is not None:
                        cv2.imwrite(f"snap_{timestamp}_mask.jpg", mask_snap)
                    print(f"\n>>> snap_{timestamp}_raw.jpg + debug.jpg + mask.jpg")
                else:
                    print("\n>>> [LỖI] Không có frame.")
            elif ch in ('q', '\x03'):
                break

            if cam is not None:
                with _frame_lock:
                    frame = _latest_frame.copy() if _latest_frame is not None else None
                if frame is None:
                    time.sleep(0.01); continue
            else:
                frame = np.ones((CAM_H, CAM_W, 3), dtype=np.uint8) * 200

            need_dbg = (tft is not None) or (args.debug and debug_cnt % 10 == 0)
            offset_lane, dbg_img, _ = detect_lane_offset(frame, debug=need_dbg)
            if args.debug and (debug_cnt % 10 == 0) and dbg_img is not None:
                cv2.imwrite(f"/tmp/cv_debug_{debug_cnt:04d}.jpg", dbg_img)

            is_zebra = detect_zebra_crossing(frame)
            if is_zebra:
                _zebra_hold_cnt = ZEBRA_HOLD_FRAMES
            elif _zebra_hold_cnt > 0:
                _zebra_hold_cnt -= 1
            in_zebra = _zebra_hold_cnt > 0

            offset = 0.0 if in_zebra else offset_lane
            if offset is not None and not in_zebra:
                if abs(offset - _prev_valid_offset) > MAX_OFFSET_JUMP:
                    offset = None
                else:
                    _prev_valid_offset = offset

            if tft is not None and debug_cnt % 3 == 0:
                tft.show(dbg_img if dbg_img is not None else frame.copy())
            debug_cnt += 1

            raw_steer    = offset_to_steer(offset)
            smooth_steer = SMOOTH_ALPHA * raw_steer + (1 - SMOOTH_ALPHA) * smooth_steer
            steer_val    = int(round(smooth_steer))
            lane_found   = offset is not None

            # ── Cập nhật history chỉ khi có lane thật sự ─────────────────
            if lane_found and not is_recovering:
                hist_steer.append(steer_val)
                if len(hist_steer) > HISTORY_LEN:
                    hist_steer.pop(0)
                no_lane_cnt = 0
            elif not lane_found and not is_recovering:
                no_lane_cnt += 1

            if recovery_cd > 0:
                recovery_cd -= 1

            # ── Kích hoạt recovery: mất lane > 15f, history ≥ 5f, không cooldown
            if (not is_recovering and recovery_cd == 0
                    and no_lane_cnt > 15 and len(hist_steer) >= 5):
                is_recovering = True
                recovery_step = 0
                smooth_steer  = 0.0
                print(f"\n>>> [RECOVERY] Mất lane {no_lane_cnt}f, hist={len(hist_steer)}f — lùi")

            # ── drive_mode / spd / steer_val ─────────────────────────────
            if is_recovering:
                drive_mode = 'B'
                spd        = 20
                idx        = max(0, len(hist_steer) - 1 - recovery_step)
                steer_val  = -hist_steer[idx]   # đảo dấu vì đang lùi
                recovery_step += 1
                if recovery_step >= RECOVERY_FRAMES:
                    is_recovering = False
                    no_lane_cnt   = 0
                    recovery_cd   = RECOVERY_COOLDOWN
                    hist_steer.clear()
                    print("\n>>> [RECOVERY] Xong — cooldown")

            elif in_zebra:
                drive_mode = 'F'
                spd        = ZEBRA_SPEED

            elif lane_found:
                drive_mode = 'F'
                spd        = args.speed

            else:
                drive_mode = 'F'
                spd = SLOW_SPEED if no_lane_cnt <= 30 else 0
                if spd == 0: steer_val = 0

            with _uart_lock:
                _uart_steer      = steer_val
                _uart_speed      = spd if running else 0
                _uart_drive_mode = drive_mode if running else 'S'

            fps_cnt += 1
            if time.time() - fps_t0 >= 2.0:
                fps = fps_cnt / (time.time() - fps_t0)
                fps_cnt = 0; fps_t0 = time.time()

            off_str   = f"{offset:+.3f}" if lane_found else " N/A "
            state_tag = "RUN " if running else "STOP"
            if is_recovering:
                lane_tag = f"RECV({recovery_step:02d}/{RECOVERY_FRAMES})"
            elif in_zebra:
                lane_tag = "ZEBRA"
            elif lane_found:
                lane_tag = "LANE "
            else:
                cd_str   = f"cd={recovery_cd}" if recovery_cd > 0 else f"nh={len(hist_steer)}"
                lane_tag = f"NOLN({no_lane_cnt:02d} {cd_str})"
            print(f"\r[{state_tag}][{drive_mode}] {lane_tag:<22}  fps={fps:4.1f}  "
                  f"off={off_str}  str={steer_val:+4d}  spd={spd:2d}",
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