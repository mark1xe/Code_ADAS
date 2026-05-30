#!/usr/bin/env python3
"""
cv_lane_driver.py — Tự lái bằng OpenCV truyền thống
=====================================================================
Cập nhật mới nhất:
1. Fix Vạch Ngựa Vằn: Nhận diện Vạch Dừng (Stop Line) nằm ngang để ép xe chạy thẳng qua giao lộ.
2. Fix Lỗi Scope: Đã sửa UnboundLocalError cho các biến toàn cục.
3. Fix Servo góc cua: Tăng điểm lookahead (65%), tăng Curve Boost (x2.0), tăng độ nhạy Smooth Alpha (0.6).
4. Chuẩn hóa Lane Width thực tế = 82% khung hình.

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
CAM_FIXED_EXPOSURE = 80000   # Tăng từ 25000 → 80000μs (80ms)
                               # Ảnh thực tế gray.max=49 → quá tối, cần ~3-4x
                               # Nếu vẫn tối: thử --exposure 120000
                               # Nếu quá sáng/lòe: giảm về 60000
CAM_ANALOGUE_GAIN  = 6.0     # Tăng từ 4.0 → 6.0 để hỗ trợ thêm
CAM_FIXED_AWB      = True
CAM_AWB_GAINS      = (1.6, 1.4)

# ── Điều khiển ────────────────────────────────────────────────────────────
DRIVE_SPEED  = 12
SLOW_SPEED   = 6
SMOOTH_ALPHA = 0.35   # Giảm từ 0.60 → phản hồi mượt hơn, không giật
                       # 0.60 quá cao: steer thay đổi 60% mỗi frame → servo nhảy
DEADBAND     = 0.04   
CAM_BIAS     = 0.0    

# ── TFT SPI display ──────────────────────────────────────────────────────
TFT_W  = 320
TFT_H  = 240
TFT_FB = "/dev/fb1"

# ── CV params ─────────────────────────────────────────────────────────────
ROI_TOP          = 0.50

# HSV range cho vạch TRẮNG — đo thực tế từ ảnh sa hình:
# Vạch trắng: S p95=35, V p5=159  → dùng S_MAX=45, V_MIN=130 (có buffer)
# Nền nâu tối: V p95=83           → V_MIN=130 cách xa nền, không lọt nhiễu
# Để tinh chỉnh khi chạy thực tế dùng: --hsv-s-max / --hsv-v-min
HSV_H_MIN, HSV_H_MAX = 0,   180
HSV_S_MIN, HSV_S_MAX = 0,   55    # vạch trắng S thực tế thấp
HSV_V_MIN, HSV_V_MAX = 80,  255   # thấp để bắt được kể cả khi hơi tối
                                   # nền nâu V thường <50 → vẫn an toàn

CANNY_LOW        = 50
CANNY_HIGH       = 120
HOUGH_THRESHOLD  = 40
HOUGH_MIN_LENGTH = 30
HOUGH_MAX_GAP    = 40

MAX_OFFSET_JUMP  = 0.70

# ── Crosswalk (Zebra) detection ───────────────────────────────────────────
ZEBRA_HOLD_FRAMES   = 22
ZEBRA_SPEED         = 8

# ── YOLO Traffic Detector ─────────────────────────────────────────────────
YOLO_MODEL_PATH  = "traffic_detector.onnx"
YOLO_IMG_SIZE    = 320
YOLO_CONF        = 0.45
YOLO_NMS         = 0.40
YOLO_SKIP_FRAMES = 3     # chạy YOLO mỗi N frame — giảm tải RPi (~3Hz ở 10fps)

CLS_STOP_SIGN    = 0
CLS_NO_ENTRY     = 1
CLS_RED_LIGHT    = 2
CLS_YELLOW_LIGHT = 3
CLS_GREEN_LIGHT  = 4
CLS_PERSON       = 5
CLASSES = ["stop_sign", "no_entry", "red_light", "yellow_light", "green_light", "person"]
# Màu debug RGB cho mỗi class (vẽ lên BGR frame thì swap trước khi dùng)
CLS_COLORS_RGB = [
    (220,  0,   0),   # 0 stop_sign    — đỏ đậm
    (180,  0,   0),   # 1 no_entry     — đỏ
    (255,  0,   0),   # 2 red_light    — đỏ tươi
    (255, 200,  0),   # 3 yellow_light — vàng
    (  0, 220,  0),   # 4 green_light  — xanh lá
    (  0, 180, 255),  # 5 person       — xanh dương
]
# BGR version để vẽ lên dbg_bgr frame
CLS_COLORS = [(b, g, r) for r, g, b in CLS_COLORS_RGB]

STOP_DURATION_S  = 3.0   # giây dừng khi gặp stop_sign / no_entry / red_light
YOLO_STOP_HOLD   = 30    # frame duy trì lệnh dừng sau khi mất detection

# ── Shared state ──────────────────────────────────────────────────────────
_uart_steer        = 0
_uart_speed        = 0
_uart_running      = False
_uart_lock         = threading.Lock()
_stop_ev           = threading.Event()
_latest_frame      = None
_frame_lock        = threading.Lock()
_prev_valid_offset = 0.0
_zebra_hold_cnt    = 0
_yolo_detections   = []      # list[(cls_id, conf, x1,y1,x2,y2)] — frame gần nhất
_yolo_stop_cnt     = 0       # đếm ngược frame còn trong trạng thái dừng do YOLO
_yolo_slow_cnt     = 0       # đếm ngược frame còn trong trạng thái chậm (yellow)


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




# ─────────────────────────────────────────────────────────────────────────
# YOLO Detector (OpenCV DNN — không cần ultralytics trên RPi)
# ─────────────────────────────────────────────────────────────────────────
class YOLODetector:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            print(f"[YOLO] ⚠️  Không tìm thấy {model_path} — YOLO bị tắt")
            self._net = None
            return
        self._net = cv2.dnn.readNetFromONNX(model_path)
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print(f"[YOLO] Model loaded: {model_path}")

    @property
    def ok(self):
        return self._net is not None

    def detect(self, frame_rgb: np.ndarray) -> list:
        """
        frame_rgb: numpy (H, W, 3) RGB
        Trả về list[(cls_id, conf, x1, y1, x2, y2)] tọa độ pixel gốc.

        YOLOv8 ONNX output shape: (1, 10, 8400)
          10 = 4 bbox (cx,cy,w,h normalized to IMG_SIZE) + 6 class scores
          8400 = số anchor predictions
        """
        if self._net is None:
            return []

        H, W = frame_rgb.shape[:2]
        # swapRB=False vì frame đã là RGB (DNN blob cần RGB)
        blob = cv2.dnn.blobFromImage(
            frame_rgb, scalefactor=1/255.0,
            size=(YOLO_IMG_SIZE, YOLO_IMG_SIZE),
            swapRB=False, crop=False
        )
        self._net.setInput(blob)
        raw = self._net.forward()[0]   # shape (10, 8400)

        # Transpose → (8400, 10)
        raw = raw.T
        sx = W / YOLO_IMG_SIZE
        sy = H / YOLO_IMG_SIZE

        boxes, confs, class_ids = [], [], []
        for row in raw:
            scores = row[4:]
            cid    = int(np.argmax(scores))
            conf   = float(scores[cid])
            if conf < YOLO_CONF:
                continue
            cx, cy, bw, bh = row[:4]
            x1 = int((cx - bw / 2) * sx)
            y1 = int((cy - bh / 2) * sy)
            x2 = int((cx + bw / 2) * sx)
            y2 = int((cy + bh / 2) * sy)
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confs.append(conf)
            class_ids.append(cid)

        if not boxes:
            return []

        idxs = cv2.dnn.NMSBoxes(boxes, confs, YOLO_CONF, YOLO_NMS)
        results = []
        flat = idxs.flatten() if len(idxs) else []
        for i in flat:
            x, y, w_, h_ = boxes[i]
            results.append((class_ids[i], confs[i], x, y, x + w_, y + h_))
        return results


def apply_yolo_detections(detections: list) -> tuple[str, int]:
    """
    Phân tích kết quả YOLO → trả về (action, speed_override).
    action: 'stop' | 'slow' | 'go' | 'none'
    speed_override: tốc độ áp đặt (0 = dừng, -1 = không override)
    """
    if not detections:
        return 'none', -1

    cls_ids = [d[0] for d in detections]

    # Ưu tiên: stop > no_entry > red_light > yellow > green
    if CLS_STOP_SIGN in cls_ids or CLS_NO_ENTRY in cls_ids or CLS_RED_LIGHT in cls_ids:
        return 'stop', 0
    if CLS_YELLOW_LIGHT in cls_ids:
        return 'slow', SLOW_SPEED
    if CLS_GREEN_LIGHT in cls_ids:
        return 'go', -1
    return 'none', -1


def draw_yolo_overlay(dbg_bgr: np.ndarray, detections: list) -> np.ndarray:
    """Vẽ bounding box + label lên debug frame BGR."""
    for cid, conf, x1, y1, x2, y2 in detections:
        color = CLS_COLORS[cid] if cid < len(CLS_COLORS) else (255, 255, 255)
        label = f"{CLASSES[cid]} {conf:.2f}"
        cv2.rectangle(dbg_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(dbg_bgr, label, (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return dbg_bgr


def detect_zebra_crossing(frame_rgb: np.ndarray) -> bool:
    """
    Phát hiện vạch đi bộ bằng cách tìm VẠCH DỪNG (Stop line) ngang.
    Dùng HSV mask thay threshold tĩnh → ổn định hơn khi ánh sáng thay đổi.
    """
    H, W = frame_rgb.shape[:2]
    roi_y = int(H * 0.4)

    hsv  = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    blur = cv2.GaussianBlur(hsv, (7, 7), 0)

    lo = np.array([HSV_H_MIN, HSV_S_MIN, HSV_V_MIN], dtype=np.uint8)
    hi = np.array([HSV_H_MAX, HSV_S_MAX, HSV_V_MAX], dtype=np.uint8)
    white_mask = cv2.inRange(blur, lo, hi)

    edges = cv2.Canny(white_mask, 50, 120)

    roi_edges = np.zeros_like(edges)
    roi_edges[roi_y:H, :] = edges[roi_y:H, :]

    lines = cv2.HoughLinesP(
        roi_edges,
        rho=1, theta=np.pi / 180,
        threshold=40,
        minLineLength=int(W * 0.25),
        maxLineGap=20
    )

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1: continue
            slope = abs((y2 - y1) / (x2 - x1))
            if slope < 0.15:
                return True

    return False


def detect_lane_offset(frame_rgb: np.ndarray, debug: bool = False):
    H, W = frame_rgb.shape[:2]
    roi_y = int(H * ROI_TOP)

    # ── HSV white mask với CLAHE trên V channel ──────────────────────────
    # CLAHE normalize độ sáng cục bộ → vạch vẫn detect được dù ảnh tối/không đều
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    # Tăng cường V channel bằng CLAHE trước khi blur
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    blur = cv2.GaussianBlur(hsv, (7, 7), 0)

    lo = np.array([HSV_H_MIN, HSV_S_MIN, HSV_V_MIN], dtype=np.uint8)
    hi = np.array([HSV_H_MAX, HSV_S_MAX, HSV_V_MAX], dtype=np.uint8)
    white_mask = cv2.inRange(blur, lo, hi)

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

    LANE_WIDTH = W * 0.82  
    
    has_left = len(left_xs) > 0
    has_right = len(right_xs) > 0
    xl = 0.0
    xr = 0.0
    single_lane = False

    if has_left:
        xl = float(np.min(left_xs))
        
    if has_right:
        xr = float(np.max(right_xs))

    if has_left and has_right:
        lane_center = (xl + xr) / 2.0
    elif has_left:
        lane_center = xl + (LANE_WIDTH / 2.0)
        single_lane = True
    elif has_right:
        lane_center = xr - (LANE_WIDTH / 2.0)
        single_lane = True
    else:
        return None, dbg

    offset = float(np.clip((lane_center - cx) / cx, -1.0, 1.0))

    if single_lane:
        offset = float(np.clip(offset * 1.3, -1.0, 1.0))
        # Giảm từ x2.0 → x1.3: boost nhẹ khi mất 1 vạch
        # x2.0 gây servo nhảy mạnh khi Hough mất vạch giữa cua

    offset = float(np.clip(offset - CAM_BIAS, -1.0, 1.0))

    if debug and dbg is not None:
        for (x1, y1, x2, y2) in left_lines:
            cv2.line(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for (x1, y1, x2, y2) in right_lines:
            cv2.line(dbg, (x1, y1), (x2, y2), (0, 100, 255), 2)
            
        cv2.line(dbg, (0, lookahead_y), (W, lookahead_y), (50, 50, 50), 1)
        if has_left: cv2.circle(dbg, (int(xl), lookahead_y), 4, (255, 255, 255), -1)
        if has_right: cv2.circle(dbg, (int(xr), lookahead_y), 4, (255, 255, 255), -1)
        
        cv2.line(dbg, (int(lane_center), H - 10), (int(lane_center), roi_y), (0, 255, 255), 2)
        cv2.line(dbg, (int(cx), H - 10), (int(cx), roi_y), (128, 128, 128), 1)
        cv2.polylines(dbg, [roi_pts], True, (255, 0, 255), 1)
        tag = f"off={offset:+.3f} L={has_left} R={has_right}"
        cv2.putText(dbg, tag, (5, H - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        global _zebra_hold_cnt
        if _zebra_hold_cnt > 0:
            cv2.putText(dbg, f"ZEBRA hold={_zebra_hold_cnt}", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

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
    global ROI_TOP, CANNY_LOW, CANNY_HIGH, SMOOTH_ALPHA
    global HSV_S_MAX, HSV_V_MIN
    global _prev_valid_offset, _zebra_hold_cnt
    global _uart_steer, _uart_speed, _uart_running
    global _yolo_detections, _yolo_stop_cnt, _yolo_slow_cnt

    ap = argparse.ArgumentParser()
    ap.add_argument("--speed",       type=int,   default=DRIVE_SPEED)
    ap.add_argument("--port",        default=UART_PORT)
    ap.add_argument("--smooth",      type=float, default=SMOOTH_ALPHA)
    ap.add_argument("--hsv-s-max",   type=int,   default=HSV_S_MAX,
                    help="HSV Saturation tối đa cho màu trắng (default=60). Tăng nếu miss vạch, giảm nếu nền lọt vào.")
    ap.add_argument("--hsv-v-min",   type=int,   default=HSV_V_MIN,
                    help="HSV Value tối thiểu cho màu trắng (default=160). Giảm nếu vạch tối, tăng nếu nền quá sáng.")
    ap.add_argument("--canny-low",   type=int,   default=CANNY_LOW)
    ap.add_argument("--canny-high",  type=int,   default=CANNY_HIGH)
    ap.add_argument("--roi-top",     type=float, default=ROI_TOP)
    ap.add_argument("--yolo-model",  default=YOLO_MODEL_PATH,
                    help="Path đến file .onnx (default: traffic_detector.onnx)")
    ap.add_argument("--debug",       action="store_true")
    ap.add_argument("--auto-start",  action="store_true")
    ap.add_argument("--no-camera",   action="store_true")
    ap.add_argument("--no-display",  action="store_true")
    ap.add_argument("--exposure",    type=int,   default=CAM_FIXED_EXPOSURE)
    ap.add_argument("--gain",        type=float, default=CAM_ANALOGUE_GAIN)
    ap.add_argument("--auto-awb",    action="store_true")
    args = ap.parse_args()

    ROI_TOP      = args.roi_top
    CANNY_LOW    = args.canny_low
    CANNY_HIGH   = args.canny_high
    SMOOTH_ALPHA = args.smooth
    HSV_S_MAX    = args.hsv_s_max
    HSV_V_MIN    = args.hsv_v_min

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

    # ── YOLO init ─────────────────────────────────────────────────────────
    yolo = YOLODetector(args.yolo_model)

    threading.Thread(target=_uart_thread, args=(uart, _stop_ev),
                     daemon=True, name="uart").start()

    running      = bool(args.auto_start)
    smooth_steer = 0.0
    no_lane_cnt  = 0
    yolo_frame_cnt = 0    # đếm frame để skip YOLO
    fps_t0       = time.time()
    fps_cnt      = 0
    fps          = 0.0
    debug_cnt    = 0

    with _uart_lock:
        _uart_running = running

    print("═" * 55)
    print("  CV LANE DRIVER + YOLO TRAFFIC DETECTOR          ")
    print("═" * 55)
    print("  A + Enter = Bắt đầu   |  S + Enter = Dừng")
    print("  C + Enter = Chụp ảnh  |  Q + Enter = Thoát")
    print(f"  HSV S≤{HSV_S_MAX} V≥{HSV_V_MIN}  Canny {CANNY_LOW}/{CANNY_HIGH}  ROI={ROI_TOP:.0%}")
    print(f"  YOLO: {'OK' if yolo.ok else 'DISABLED'}  skip={YOLO_SKIP_FRAMES}  conf={YOLO_CONF}")
    print("═" * 55)

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

            # ── YOLO detection (mỗi YOLO_SKIP_FRAMES frame) ───────────────
            yolo_frame_cnt += 1
            if yolo.ok and yolo_frame_cnt % YOLO_SKIP_FRAMES == 0:
                _yolo_detections = yolo.detect(frame)
                action, _ = apply_yolo_detections(_yolo_detections)
                if action == 'stop':
                    _yolo_stop_cnt = YOLO_STOP_HOLD
                    _yolo_slow_cnt = 0
                elif action == 'slow':
                    _yolo_slow_cnt = YOLO_STOP_HOLD
            else:
                # Đếm ngược hold giữa các lần detect
                if _yolo_stop_cnt > 0: _yolo_stop_cnt -= 1
                if _yolo_slow_cnt > 0: _yolo_slow_cnt -= 1

            # ── Zebra crossing detection ───────────────────────────────────
            is_zebra = detect_zebra_crossing(frame)
            if is_zebra:
                _zebra_hold_cnt = ZEBRA_HOLD_FRAMES
            elif _zebra_hold_cnt > 0:
                _zebra_hold_cnt -= 1
            in_zebra = _zebra_hold_cnt > 0

            # ── Lane detection ─────────────────────────────────────────────
            if in_zebra:
                offset  = 0.0
                dbg_img = None
            else:
                need_dbg = (tft is not None) or (args.debug and debug_cnt % 10 == 0)
                offset, dbg_img = detect_lane_offset(frame, debug=need_dbg)
                if args.debug and (debug_cnt % 10 == 0) and dbg_img is not None:
                    # Vẽ YOLO lên debug frame
                    if _yolo_detections:
                        dbg_img = draw_yolo_overlay(dbg_img, _yolo_detections)
                    cv2.imwrite(f"/tmp/cv_debug_{debug_cnt:04d}.jpg", dbg_img)

            if offset is not None:
                if abs(offset - _prev_valid_offset) > MAX_OFFSET_JUMP:
                    offset = None
                else:
                    _prev_valid_offset = offset

            if tft is not None and debug_cnt % 3 == 0:
                show_img = dbg_img if dbg_img is not None else cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if _yolo_detections and dbg_img is None:
                    show_img = draw_yolo_overlay(show_img.copy(), _yolo_detections)
                tft.show(cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB))
            debug_cnt += 1

            # ── Steer ──────────────────────────────────────────────────────
            raw_steer    = offset_to_steer(offset)
            smooth_steer = SMOOTH_ALPHA * raw_steer + (1 - SMOOTH_ALPHA) * smooth_steer
            steer_val    = int(round(smooth_steer))

            lane_found  = offset is not None
            no_lane_cnt = 0 if lane_found else no_lane_cnt + 1

            spd = args.speed if lane_found else SLOW_SPEED
            if no_lane_cnt > 20:
                spd = 0; steer_val = 0

            # ── Override theo thứ tự ưu tiên ──────────────────────────────
            # 1. YOLO stop (cao nhất) — stop_sign / no_entry / red_light
            if _yolo_stop_cnt > 0:
                spd = 0
                # Không reset steer để xe không trượt khi dừng
            # 2. Zebra crossing
            elif in_zebra:
                spd = ZEBRA_SPEED
            # 3. YOLO slow — yellow_light
            elif _yolo_slow_cnt > 0:
                spd = min(spd, SLOW_SPEED)

            with _uart_lock:
                _uart_steer = steer_val
                _uart_speed = spd if running else 0

            # ── FPS & console ──────────────────────────────────────────────
            fps_cnt += 1
            if time.time() - fps_t0 >= 2.0:
                fps     = fps_cnt / (time.time() - fps_t0)
                fps_cnt = 0; fps_t0 = time.time()

            off_str = f"{offset:+.3f}" if lane_found else " N/A "
            state_tag = "RUN " if running else "STOP"
            # Trạng thái ưu tiên cao nhất hiển thị
            if _yolo_stop_cnt > 0:
                det_names = [CLASSES[d[0]] for d in _yolo_detections]
                lane_tag  = f"YOLO:{'+'.join(det_names)}"
            elif in_zebra:
                lane_tag  = "ZEBRA"
            elif _yolo_slow_cnt > 0:
                lane_tag  = "SLOW"
            elif lane_found:
                lane_tag  = "LANE"
            else:
                lane_tag  = "NOLN"

            print(f"\r[{state_tag}] {lane_tag:<18}  fps={fps:4.1f}  "
                  f"off={off_str}  steer={steer_val:+4d}  spd={spd:2d}",
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