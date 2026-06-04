#!/usr/bin/env python3
"""
cv_lane_driver.py — Tự lái bằng OpenCV truyền thống
=====================================================================
Cập nhật mới:
1. Fix Vạch Ngựa Vằn & Loạng choạng: Chỉ bắt vạch ngoài cùng (Outermost filtering).
2. Fix Servo góc cua: Tăng điểm lookahead (65%), tăng Curve Boost (x2.0), tăng độ nhạy Smooth Alpha (0.6).
3. Chuẩn hóa Lane Width thực tế = 82% khung hình.

Chạy trên RPi:
    python3 cv_lane_driver.py                   # Chạy bình thường
    python3 cv_lane_driver.py --auto-start      # Tự chạy
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
CAM_FIXED_EXPOSURE = 25000  
CAM_ANALOGUE_GAIN  = 4.0    
CAM_FIXED_AWB      = True   
CAM_AWB_GAINS      = (1.6, 1.4)  

# ── Điều khiển ────────────────────────────────────────────────────────────
DRIVE_SPEED  = 12
SLOW_SPEED   = 6
SMOOTH_ALPHA = 0.60   # Tăng từ 0.35 lên 0.6 để servo bẻ lái nhanh và dứt khoát hơn
DEADBAND     = 0.04   
CAM_BIAS     = 0.0    

# ── TFT SPI display ──────────────────────────────────────────────────────
TFT_W  = 320
TFT_H  = 240
TFT_FB = "/dev/fb1"

# ── CV params ─────────────────────────────────────────────────────────────
ROI_TOP          = 0.50   
WHITE_THRESHOLD  = 0      
CANNY_LOW        = 80     
CANNY_HIGH       = 150
HOUGH_THRESHOLD  = 40     
HOUGH_MIN_LENGTH = 30     
HOUGH_MAX_GAP    = 40     

MAX_OFFSET_JUMP  = 0.70  

# ── Shared state ──────────────────────────────────────────────────────────
_uart_steer        = 0
_uart_speed        = 0
_uart_running      = False
_uart_lock         = threading.Lock()
_stop_ev           = threading.Event()
_latest_frame      = None
_frame_lock        = threading.Lock()
_prev_valid_offset = 0.0   


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


def detect_lane_offset(frame_rgb: np.ndarray, debug: bool = False):
    H, W = frame_rgb.shape[:2]
    roi_y = int(H * ROI_TOP)

    gray  = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    blur  = cv2.GaussianBlur(gray, (11, 11), 0)   
    
    if WHITE_THRESHOLD == 0:
        _, white_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, white_mask = cv2.threshold(blur, WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    edges = cv2.Canny(white_mask, CANNY_LOW, CANNY_HIGH)

    mask = np.zeros_like(edges)
    roi_pts = np.array([[
        (0, H),
        (0, roi_y),
        (W, roi_y),
        (W, H),
    ]], dtype=np.int32)
    cv2.fillPoly(mask, roi_pts, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

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

    left_xs  = []
    right_xs = []
    left_lines = []
    right_lines = []
    cx = W / 2.0
    
    # Đẩy mốc nhìn xa lên 65% (nhìn xa hơn cũ 75%) để xe cua sớm
    lookahead_y = int(H * 0.65) 

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1: continue
        dy = float(y2 - y1)
        dx = float(x2 - x1)
        if abs(dy) < 1: continue
        slope = dy / dx
        
        if abs(slope) < 0.15:
            continue

        x_look = x1 + (lookahead_y - y1) * dx / dy

        if x_look < cx:
            left_xs.append(x_look)
            left_lines.append((x1,y1,x2,y2))
        else:
            right_xs.append(x_look)
            right_lines.append((x1,y1,x2,y2))

    if not left_xs and not right_xs:
        return None, dbg

    # Tính độ rộng làn đường thực tế từ ảnh chụp (~82% khung hình)
    LANE_WIDTH = W * 0.82  
    
    # ── LOGIC LỌC VẠCH NGOÀI CÙNG (BỎ QUA NGỰA VẰN) ──
    has_left = len(left_xs) > 0
    has_right = len(right_xs) > 0
    xl = 0.0
    xr = 0.0
    single_lane = False

    if has_left:
        xl = float(np.min(left_xs))
        
    if has_right:
        xr = float(np.max(right_xs))

    # TÍNH TOÁN LANE CENTER
    if has_left and has_right:
        lane_center = (xl + xr) / 2.0
    elif has_left:
        # Chỉ có vạch trái (Cua Trái hoặc mất vạch phải)
        lane_center = xl + (LANE_WIDTH / 2.0)
        single_lane = True
    elif has_right:
        # Chỉ có vạch phải (Cua Phải hoặc mất vạch trái)
        lane_center = xr - (LANE_WIDTH / 2.0)
        single_lane = True
    else:
        # Mất cả 2 vạch
        return None, dbg

    offset = float(np.clip((lane_center - cx) / cx, -1.0, 1.0))

    if single_lane:
        # Tăng hệ số bẻ lái góc cua lên x2.0 để thoát cua gắt
        offset = float(np.clip(offset * 2.0, -1.0, 1.0))

    offset = float(np.clip(offset - CAM_BIAS, -1.0, 1.0))

    if debug and dbg is not None:
        for (x1, y1, x2, y2) in left_lines:
            cv2.line(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for (x1, y1, x2, y2) in right_lines:
            cv2.line(dbg, (x1, y1), (x2, y2), (0, 100, 255), 2)
            
        cv2.line(dbg, (0, lookahead_y), (W, lookahead_y), (50, 50, 50), 1)
        # Vẽ các vạch ngoài cùng được chọn bằng màu trắng nét đứt
        if has_left: cv2.circle(dbg, (int(xl), lookahead_y), 4, (255, 255, 255), -1)
        if has_right: cv2.circle(dbg, (int(xr), lookahead_y), 4, (255, 255, 255), -1)
        
        cv2.line(dbg, (int(lane_center), H - 10), (int(lane_center), roi_y), (0, 255, 255), 2)
        cv2.line(dbg, (int(cx), H - 10), (int(cx), roi_y), (128, 128, 128), 1)
        cv2.polylines(dbg, [roi_pts], True, (255, 0, 255), 1)
        tag = f"off={offset:+.3f} L={has_left} R={has_right}"
        cv2.putText(dbg, tag, (5, H - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    return offset, dbg


def offset_to_steer(offset: float | None) -> int:
    if offset is None or abs(offset) < DEADBAND:
        return 0
    sign = 1 if offset > 0 else -1
    x = min((abs(offset) - DEADBAND) / (1.0 - DEADBAND), 1.0)
    mag = 5.0 + 95.0 * x  
    return int(round(sign * min(mag, 100.0)))


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


def main():
    global ROI_TOP, CANNY_LOW, CANNY_HIGH, SMOOTH_ALPHA, WHITE_THRESHOLD
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed",       type=int,   default=DRIVE_SPEED)
    ap.add_argument("--port",        default=UART_PORT)
    ap.add_argument("--smooth",      type=float, default=SMOOTH_ALPHA)
    ap.add_argument("--thresh",      type=int,   default=WHITE_THRESHOLD)
    ap.add_argument("--canny-low",   type=int,   default=CANNY_LOW)
    ap.add_argument("--canny-high",  type=int,   default=CANNY_HIGH)
    ap.add_argument("--roi-top",     type=float, default=ROI_TOP)
    ap.add_argument("--debug",       action="store_true")
    ap.add_argument("--auto-start",  action="store_true")
    ap.add_argument("--no-camera",   action="store_true")
    ap.add_argument("--no-display",  action="store_true")
    ap.add_argument("--exposure",    type=int,   default=CAM_FIXED_EXPOSURE)
    ap.add_argument("--gain",        type=float, default=CAM_ANALOGUE_GAIN)
    ap.add_argument("--auto-awb",    action="store_true")
    args = ap.parse_args()

    ROI_TOP         = args.roi_top
    CANNY_LOW       = args.canny_low
    CANNY_HIGH      = args.canny_high
    SMOOTH_ALPHA    = args.smooth
    WHITE_THRESHOLD = args.thresh

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
            main={"format": "RGB888", "size": (CAM_W, CAM_H)}))
        cam.start()
        time.sleep(1.0)

        _controls = {}
        if args.exposure > 0:
            _controls["AeEnable"]       = False
            _controls["ExposureTime"]   = args.exposure
            _controls["AnalogueGain"]   = args.gain
            print(f"Camera: exposure khoá {args.exposure}μs, gain={args.gain}")
        else:
            print("Camera: auto-exposure")
        if not args.auto_awb:
            _controls["AwbEnable"]      = False
            _controls["ColourGains"]    = CAM_AWB_GAINS
            print(f"Camera: AWB khoá gains={CAM_AWB_GAINS}")
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
    print("  OPTIMIZED CV LANE DRIVER (CROSSWALK FIXED)  ")
    print("═" * 50)
    print("  A + Enter = Bắt đầu   |  S + Enter = Dừng")
    print("  C + Enter = Chụp ảnh  |  Q + Enter = Thoát")
    print(f"  Thresh={'Otsu (Auto)' if WHITE_THRESHOLD==0 else WHITE_THRESHOLD}  Canny {CANNY_LOW}/{CANNY_HIGH}")
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

    try:
        while True:
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
            elif ch == 'c':
                timestamp = time.strftime("%H%M%S")
                if cam is not None and _latest_frame is not None:
                    raw_bgr = cv2.cvtColor(_latest_frame, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f"snap_{timestamp}_raw.jpg", raw_bgr)
                    
                    _, dbg_snap = detect_lane_offset(_latest_frame, debug=True)
                    if dbg_snap is not None:
                        cv2.imwrite(f"snap_{timestamp}_debug.jpg", dbg_snap)
                        
                    print(f"\n>>> [SCREENSHOT] Đã lưu ảnh snap_{timestamp}_raw.jpg và snap_{timestamp}_debug.jpg")
                else:
                    print("\n>>> [SCREENSHOT LỖI] Không có khung hình từ camera.")
            elif ch in ('q', '\x03'):
                break

            if cam is not None:
                with _frame_lock:
                    frame = _latest_frame
                if frame is None:
                    time.sleep(0.01); continue
            else:
                frame = np.ones((CAM_H, CAM_W, 3), dtype=np.uint8) * 200

            need_dbg = (tft is not None) or (args.debug and debug_cnt % 10 == 0)
            offset, dbg_img = detect_lane_offset(frame, debug=need_dbg)
            if args.debug and (debug_cnt % 10 == 0) and dbg_img is not None:
                cv2.imwrite(f"/tmp/cv_debug_{debug_cnt:04d}.jpg", dbg_img)

            global _prev_valid_offset
            if offset is not None:
                if abs(offset - _prev_valid_offset) > MAX_OFFSET_JUMP:
                    offset = None   
                else:
                    _prev_valid_offset = offset

            if tft is not None and debug_cnt % 3 == 0:
                show_img = dbg_img if dbg_img is not None else cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                tft.show(cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB))
            debug_cnt += 1

            raw_steer = offset_to_steer(offset)
            smooth_steer = SMOOTH_ALPHA * raw_steer + (1 - SMOOTH_ALPHA) * smooth_steer
            steer_val = int(round(smooth_steer))

            lane_found = offset is not None
            no_lane_cnt = 0 if lane_found else no_lane_cnt + 1

            spd = args.speed if lane_found else SLOW_SPEED
            if no_lane_cnt > 20:   
                spd = 0; steer_val = 0
            with _uart_lock:
                _uart_steer = steer_val
                _uart_speed = spd if running else 0

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