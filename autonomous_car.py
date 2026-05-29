"""
===================================================================
HỆ THỐNG XE TỰ HÀNH - RASPBERRY PI 4
Autonomous Car System with Lane Tracking & Traffic Sign Recognition
===================================================================
Tác giả   : Embedded AI & Computer Vision Expert
Nền tảng  : Raspberry Pi 4 + Camera Module / USB Camera
Ngôn ngữ  : Python 3.8+
Thư viện  : OpenCV 4.x, NumPy, threading, serial
===================================================================

CẤU TRÚC HỆ THỐNG:
┌─────────────────────────────────────────────────────────┐
│                    Camera Input (Video Stream)           │
└───────────────────────┬─────────────────────────────────┘
                        │
           ┌────────────▼────────────┐
           │     Frame Grabber       │  ← Thread riêng (không block)
           └────────────┬────────────┘
                        │  shared frame buffer
          ┌─────────────┴──────────────┐
          │                            │
┌─────────▼──────────┐    ┌────────────▼────────────┐
│  Lane Tracking     │    │  Traffic Sign Detection │
│  (Thread 1 - Fast) │    │  (Thread 2 - YOLO/ONNX) │
│                    │    │                         │
│  Grayscale         │    │  YOLOv8n ONNX inference │
│  Gaussian Blur     │    │  via cv2.dnn            │
│  Canny Edge        │    │                         │
│  ROI Crop          │    └────────────┬────────────┘
│  Hough Lines       │                 │ sign_result
│  Steering Angle    │    ┌────────────▼────────────┐
└────────┬───────────┘    │   Control Decision      │
         │ steering_angle └────────────┬────────────┘
         └──────────────┬──────────────┘
                        │
           ┌────────────▼────────────┐
           │  send_control_signal()  │  ← UART/I2C → MCU
           └─────────────────────────┘
"""

import cv2
import numpy as np
import threading
import time
import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from collections import deque

# ─────────────────────────────────────────────
# Cấu hình logging để dễ debug trên terminal
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# SECTION 1: DATACLASSES – Cấu trúc dữ liệu dùng chung
# ═══════════════════════════════════════════════════════════════

@dataclass
class LaneResult:
    """Kết quả từ module bám làn đường."""
    steering_angle: float = 0.0          # Góc lái tính bằng độ (-90 đến +90)
    left_line: Optional[np.ndarray] = None   # Đường thẳng làn trái [x1,y1,x2,y2]
    right_line: Optional[np.ndarray] = None  # Đường thẳng làn phải [x1,y1,x2,y2]
    lane_center_offset: float = 0.0      # Độ lệch tâm xe so với giữa làn (px)
    debug_frame: Optional[np.ndarray] = None # Frame hiển thị debug

@dataclass
class SignResult:
    """Kết quả từ module nhận diện biển báo."""
    label: str = "NONE"             # Nhãn biển báo nhận diện được
    confidence: float = 0.0         # Độ tin cậy (0.0 ~ 1.0)
    bbox: Optional[Tuple] = None    # Bounding box (x, y, w, h)
    action: str = "CONTINUE"        # Hành động đề xuất: STOP/TURN_LEFT/TURN_RIGHT/CONTINUE

@dataclass
class ControlSignal:
    """Tín hiệu điều khiển gửi xuống vi điều khiển."""
    speed: float = 0.3          # Tốc độ (0.0 ~ 1.0, đã chuẩn hóa)
    steering_angle: float = 0.0 # Góc lái (độ, âm = rẽ trái, dương = rẽ phải)
    brake: bool = False         # Phanh khẩn cấp
    timestamp: float = field(default_factory=time.time)


# ═══════════════════════════════════════════════════════════════
# SECTION 2: CẤU HÌNH THÔNG SỐ HỆ THỐNG
# ═══════════════════════════════════════════════════════════════

class Config:
    """
    Tập trung toàn bộ thông số có thể tinh chỉnh vào một chỗ.
    Thay đổi tại đây sẽ ảnh hưởng toàn bộ hệ thống.
    """

    # --- Camera ---
    CAMERA_INDEX        = 0          # 0 = webcam mặc định, hoặc đường dẫn video test
    FRAME_WIDTH         = 320        # Độ rộng khung hình (px)
    FRAME_HEIGHT        = 240        # Chiều cao khung hình (px)
    TARGET_FPS          = 30         # FPS mục tiêu của camera

    # --- Canny Edge Detection ---
    # Ngưỡng thấp: điểm biên có gradient < giá trị này bị loại
    # Ngưỡng cao: điểm biên có gradient > giá trị này chắc chắn là biên
    # TIP: Nếu phát hiện quá nhiều nhiễu → tăng cả hai ngưỡng
    #       Nếu bỏ sót đường làn → giảm ngưỡng thấp
    CANNY_THRESHOLD_LOW  = 50
    CANNY_THRESHOLD_HIGH = 150

    # --- Gaussian Blur ---
    # Kernel phải là số lẻ. Tăng lên (7,7) nếu camera bị nhiễu hạt
    GAUSSIAN_KERNEL_SIZE = (5, 5)
    GAUSSIAN_SIGMA       = 0          # 0 = tự động tính từ kernel size

    # --- ROI (Region of Interest) ---
    # Tỷ lệ theo chiều cao: chỉ xử lý từ ROI_TOP_RATIO * H đến cuối frame
    # 0.55 = lấy 45% dưới cùng. Giảm xuống nếu đường cong xuất hiện sớm hơn
    ROI_TOP_RATIO       = 0.55

    # --- Hough Line Transform ---
    # rho: độ phân giải khoảng cách (px). Thường để 1 hoặc 2
    HOUGH_RHO            = 1
    # theta: độ phân giải góc (radian). np.pi/180 = 1 độ
    HOUGH_THETA          = np.pi / 180
    # threshold: số điểm giao nhau tối thiểu để coi là đường thẳng
    # Giảm nếu hay bỏ sót làn, tăng nếu phát hiện quá nhiều nhiễu
    HOUGH_THRESHOLD      = 50
    # min_line_length: độ dài tối thiểu của đoạn thẳng (px)
    HOUGH_MIN_LINE_LEN   = 100
    # max_line_gap: khoảng cách tối đa giữa 2 đoạn thẳng để nối thành 1
    HOUGH_MAX_LINE_GAP   = 50

    # --- Lọc góc đường thẳng ---
    # Loại bỏ các đường thẳng quá nằm ngang (nhiễu), chỉ giữ đường gần dọc
    MIN_LINE_ANGLE_DEG   = 25         # Góc tối thiểu so với trục ngang (độ)

    # --- Điều hướng ---
    MAX_STEERING_ANGLE   = 30.0       # Góc lái tối đa (độ) để an toàn
    # Smoothing: trung bình của N góc lái gần nhất để tránh giật cục
    STEERING_SMOOTHING_N = 5

    # --- YOLO / ONNX ---
    YOLO_MODEL_PATH     = "yolov8n_traffic.onnx"  # Đường dẫn file .onnx
    YOLO_INPUT_SIZE     = (320, 240)               # Kích thước đầu vào của mô hình
    YOLO_CONF_THRESHOLD = 0.5                      # Ngưỡng confidence tối thiểu
    YOLO_NMS_THRESHOLD  = 0.45                     # Ngưỡng NMS (loại bỏ box trùng)
    # Tên các lớp mô hình nhận diện (phải khớp với lúc train)
    YOLO_CLASSES        = ["stop", "turn_left", "turn_right", "speed_limit", "crosswalk"]
    # Tần suất chạy YOLO: chỉ chạy mỗi N frame để tiết kiệm CPU
    YOLO_INFERENCE_EVERY_N_FRAMES = 5

    # --- UART (giao tiếp với vi điều khiển) ---
    UART_PORT            = "/dev/ttyUSB0"   # Cổng serial trên Raspberry Pi
    UART_BAUDRATE        = 115200


# ═══════════════════════════════════════════════════════════════
# SECTION 3: FRAME BUFFER – Bộ đệm frame chia sẻ giữa các thread
# ═══════════════════════════════════════════════════════════════

class SharedFrameBuffer:
    """
    Bộ đệm khung hình an toàn cho đa luồng (thread-safe).
    Thread camera ghi vào, các thread xử lý đọc ra.
    Dùng Lock để tránh race condition.
    """

    def __init__(self):
        self._frame = None
        self._lock = threading.Lock()
        self._new_frame_event = threading.Event()

    def write(self, frame: np.ndarray):
        """Ghi frame mới (gọi từ thread camera)."""
        with self._lock:
            self._frame = frame.copy()
        self._new_frame_event.set()   # Báo hiệu có frame mới

    def read(self, timeout: float = 0.1):
        """
        Đọc frame mới nhất (blocking cho đến khi có frame mới).
        Trả về None nếu timeout mà không có frame mới.
        """
        if self._new_frame_event.wait(timeout=timeout):
            self._new_frame_event.clear()
            with self._lock:
                return self._frame.copy() if self._frame is not None else None
        return None

    def read_latest(self):
        """Đọc frame mới nhất không blocking (có thể trả về frame cũ)."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None


# ═══════════════════════════════════════════════════════════════
# SECTION 4: MODULE BÁM LÀN ĐƯỜNG (LANE TRACKING)
# ═══════════════════════════════════════════════════════════════

class LaneTracker:
    """
    Module bám làn đường sử dụng Computer Vision truyền thống.
    Pipeline: Grayscale → Blur → Canny → ROI → Hough → Steering Angle
    """

    def __init__(self, config: Config):
        self.cfg = config
        # Bộ đệm lưu N góc lái gần nhất để làm mượt (moving average)
        self._angle_buffer = deque(maxlen=config.STEERING_SMOOTHING_N)
        logger.info("LaneTracker khởi tạo thành công.")

    # ─────────────────────────────────────────
    # Bước 1: Chuyển đổi sang Grayscale
    # ─────────────────────────────────────────
    def _to_grayscale(self, frame: np.ndarray) -> np.ndarray:
        """
        Chuyển frame màu (BGR) sang ảnh xám 1 kênh.
        Lý do: Canny Edge Detection chỉ hoạt động trên ảnh xám,
        và ảnh xám nhẹ hơn 3x so với BGR → xử lý nhanh hơn.
        """
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ─────────────────────────────────────────
    # Bước 2: Lọc nhiễu Gaussian Blur
    # ─────────────────────────────────────────
    def _apply_blur(self, gray: np.ndarray) -> np.ndarray:
        """
        Làm mờ ảnh để giảm nhiễu trước khi phát hiện biên.
        Không blur → Canny sẽ phát hiện rất nhiều biên giả từ texture mặt đường.
        Kernel (5,5): cân bằng giữa khử nhiễu và giữ chi tiết làn đường.
        """
        return cv2.GaussianBlur(
            gray,
            self.cfg.GAUSSIAN_KERNEL_SIZE,
            self.cfg.GAUSSIAN_SIGMA
        )

    # ─────────────────────────────────────────
    # Bước 3: Phát hiện biên Canny
    # ─────────────────────────────────────────
    def _apply_canny(self, blurred: np.ndarray) -> np.ndarray:
        """
        Phát hiện biên cạnh bằng thuật toán Canny.
        - threshold1 (low): điểm dưới ngưỡng này không phải biên
        - threshold2 (high): điểm trên ngưỡng này chắc chắn là biên
        - Điểm ở giữa: chỉ là biên nếu kết nối với điểm biên chắc chắn

        TINH CHỈNH:
        - Ngày nắng, làn rõ ràng: low=50, high=150
        - Ngày mưa, làn mờ:       low=30, high=100
        """
        return cv2.Canny(
            blurred,
            self.cfg.CANNY_THRESHOLD_LOW,
            self.cfg.CANNY_THRESHOLD_HIGH
        )

    # ─────────────────────────────────────────
    # Bước 4: Cắt vùng quan tâm (ROI)
    # ─────────────────────────────────────────
    def _apply_roi(self, edges: np.ndarray) -> np.ndarray:
        """
        Giữ lại chỉ phần dưới của ảnh (nơi làn đường thực sự hiện diện).
        Phần trên thường là trời, cây cối → gây nhiễu cho Hough.

        Vùng ROI được định nghĩa là hình thang:
             ┌──────────────────────────────┐
             │                              │  ← bỏ qua (trời, cảnh quan)
             │    [tl]──────────────[tr]    │  ← ROI_TOP_RATIO * H
             │   /                       \  │
             │  /   ← vùng ROI giữ lại →  \ │
             │ [bl]──────────────────────[br]│  ← đáy frame
             └──────────────────────────────┘
        """
        H, W = edges.shape
        top_y = int(H * self.cfg.ROI_TOP_RATIO)

        # Định nghĩa 4 đỉnh của đa giác ROI hình thang
        # Tinh chỉnh các giá trị x nếu camera bị xoay hoặc đặt lệch
        roi_vertices = np.array([[
            (0, H),            # Góc dưới trái (bl)
            (0, top_y),        # Góc trên trái (tl)
            (W, top_y),        # Góc trên phải (tr)
            (W, H),            # Góc dưới phải (br)
        ]], dtype=np.int32)

        # Tạo mask đen toàn bộ, rồi tô trắng vùng ROI
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, roi_vertices, 255)

        # Chỉ giữ lại điểm biên nằm trong ROI
        return cv2.bitwise_and(edges, mask)

    # ─────────────────────────────────────────
    # Bước 5: Hough Line Transform
    # ─────────────────────────────────────────
    def _detect_lines(self, roi_edges: np.ndarray):
        """
        Tìm tất cả đoạn thẳng trong vùng ROI bằng Probabilistic Hough Transform.
        Trả về mảng [[x1,y1,x2,y2], ...] hoặc None nếu không tìm thấy.
        """
        lines = cv2.HoughLinesP(
            roi_edges,
            rho=self.cfg.HOUGH_RHO,
            theta=self.cfg.HOUGH_THETA,
            threshold=self.cfg.HOUGH_THRESHOLD,
            minLineLength=self.cfg.HOUGH_MIN_LINE_LEN,
            maxLineGap=self.cfg.HOUGH_MAX_LINE_GAP
        )
        return lines

    # ─────────────────────────────────────────
    # Bước 6: Phân loại và trung bình hóa đường thẳng
    # ─────────────────────────────────────────
    def _average_lines(self, lines: np.ndarray, frame_width: int, frame_height: int):
        """
        Phân loại đường thẳng thành làn trái / làn phải dựa trên:
        - Độ dốc (slope): âm → làn trái, dương → làn phải
        - Lọc bỏ đường nằm ngang (nhiễu)

        Sau đó ngoại suy (extrapolate) mỗi nhóm thành một đường thẳng duy nhất
        kéo dài từ đáy frame đến đỉnh ROI.
        """
        left_points  = []   # Tập điểm thuộc làn trái
        right_points = []   # Tập điểm thuộc làn phải

        min_angle_rad = math.radians(self.cfg.MIN_LINE_ANGLE_DEG)

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Tránh chia cho 0 khi đường thẳng đứng hoàn toàn
            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)
            angle = abs(math.atan(slope))

            # Lọc bỏ đường gần như nằm ngang (vệt nứt ngang mặt đường, v.v.)
            if angle < min_angle_rad:
                continue

            # Trong OpenCV, trục Y tăng từ trên xuống dưới
            # → Làn trái:  slope < 0 (đường đi từ dưới-trái lên trên-phải)
            # → Làn phải: slope > 0 (đường đi từ dưới-phải lên trên-trái)
            if slope < 0:
                left_points.extend([(x1, y1), (x2, y2)])
            else:
                right_points.extend([(x1, y1), (x2, y2)])

        def fit_line(points, y_top, y_bottom):
            """
            Dùng numpy polyfit bậc 1 để tìm đường thẳng tốt nhất qua tập điểm,
            rồi tính tọa độ tại y_top và y_bottom.
            """
            if len(points) < 2:
                return None
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            try:
                # polyfit trả về [slope, intercept] cho đường x = slope*y + intercept
                # (đảo x,y vì đường làn gần dọc hơn ngang)
                coef = np.polyfit(ys, xs, 1)
                poly = np.poly1d(coef)
                x_bottom = int(poly(y_bottom))
                x_top    = int(poly(y_top))
                return np.array([x_bottom, y_bottom, x_top, y_top])
            except (np.RankWarning, ValueError):
                return None

        y_bottom = frame_height
        y_top    = int(frame_height * self.cfg.ROI_TOP_RATIO)

        left_line  = fit_line(left_points,  y_top, y_bottom)
        right_line = fit_line(right_points, y_top, y_bottom)

        return left_line, right_line

    # ─────────────────────────────────────────
    # Bước 7: Tính góc đánh lái
    # ─────────────────────────────────────────
    def _compute_steering_angle(self, left_line, right_line, frame_width, frame_height) -> float:
        """
        Tính góc đánh lái dựa trên vị trí tâm làn đường so với tâm camera.

        Logic:
        1. Xác định điểm giữa của làn đường ở đáy frame
        2. So sánh với tâm frame (frame_width / 2)
        3. Tính góc lái cần thiết để đưa xe về giữa làn

        Góc trả về: âm = rẽ trái, dương = rẽ phải, 0 = thẳng
        """
        frame_cx = frame_width // 2
        y_ref    = int(frame_height * 0.9)  # Điểm tham chiếu ở gần đáy frame

        def x_at_y(line, y):
            """Nội suy tọa độ x của đường thẳng tại chiều cao y."""
            if line is None:
                return None
            x1, y1, x2, y2 = line
            if y2 == y1:
                return x1
            return int(x1 + (y - y1) * (x2 - x1) / (y2 - y1))

        left_x  = x_at_y(left_line,  y_ref)
        right_x = x_at_y(right_line, y_ref)

        # Xác định tâm làn đường
        if left_x is not None and right_x is not None:
            lane_center = (left_x + right_x) // 2
        elif left_x is not None:
            # Chỉ thấy làn trái: ước tính tâm (xe đang lệch phải)
            lane_center = left_x + 160
        elif right_x is not None:
            # Chỉ thấy làn phải: ước tính tâm (xe đang lệch trái)
            lane_center = right_x - 160
        else:
            # Không thấy làn nào → giữ nguyên góc lái cũ
            return self._angle_buffer[-1] if self._angle_buffer else 0.0

        # Độ lệch tâm xe so với tâm làn (px)
        offset_px = lane_center - frame_cx

        # Chuyển đổi độ lệch pixel → góc lái (độ)
        # Công thức: angle = arctan(offset / look_ahead_distance)
        # look_ahead_distance: khoảng nhìn trước (px), điều chỉnh tùy tốc độ xe
        look_ahead = frame_height * 0.4
        raw_angle  = math.degrees(math.atan2(offset_px, look_ahead))

        # Giới hạn góc lái trong khoảng an toàn
        clamped = max(-self.cfg.MAX_STEERING_ANGLE,
                      min(self.cfg.MAX_STEERING_ANGLE, raw_angle))

        # Làm mượt góc lái bằng moving average
        self._angle_buffer.append(clamped)
        smoothed = sum(self._angle_buffer) / len(self._angle_buffer)

        return round(smoothed, 2)

    # ─────────────────────────────────────────
    # Vẽ kết quả lên frame để debug
    # ─────────────────────────────────────────
    def _draw_debug(self, frame: np.ndarray, left_line, right_line, steering_angle: float) -> np.ndarray:
        """Vẽ đường làn, góc lái và thông số debug lên frame."""
        debug = frame.copy()
        overlay = np.zeros_like(debug)

        def draw_line(line, color):
            if line is not None:
                x1, y1, x2, y2 = line
                cv2.line(debug, (x1, y1), (x2, y2), color, 4)

        # Vẽ đường làn: xanh lá = trái, đỏ = phải
        draw_line(left_line,  (0, 255, 0))
        draw_line(right_line, (0, 0, 255))

        # Vẽ vùng làn đường (hình thang bán trong suốt màu xanh lam)
        if left_line is not None and right_line is not None:
            pts = np.array([
                [left_line[0],  left_line[1]],
                [left_line[2],  left_line[3]],
                [right_line[2], right_line[3]],
                [right_line[0], right_line[1]],
            ], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (255, 150, 0))
            cv2.addWeighted(debug, 1.0, overlay, 0.3, 0, debug)

        # Hiển thị thông số
        H, W = debug.shape[:2]
        cv2.putText(debug, f"Steering: {steering_angle:+.1f} deg",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Vẽ mũi tên chỉ hướng lái từ tâm frame
        cx, cy = W // 2, int(H * 0.85)
        arrow_len = 80
        end_x = int(cx + arrow_len * math.sin(math.radians(steering_angle)))
        end_y = int(cy - arrow_len * math.cos(math.radians(steering_angle)))
        cv2.arrowedLine(debug, (cx, cy), (end_x, end_y), (0, 255, 255), 3, tipLength=0.3)
        cv2.circle(debug, (cx, cy), 6, (0, 255, 255), -1)

        return debug

    # ─────────────────────────────────────────
    # Hàm chính: xử lý một frame
    # ─────────────────────────────────────────
    def process_lane(self, frame: np.ndarray) -> LaneResult:
        """
        Pipeline hoàn chỉnh bám làn đường cho một frame.
        Gọi hàm này từ thread bám làn.
        """
        H, W = frame.shape[:2]
        result = LaneResult()

        # ── Bước 1: Grayscale ──────────────────
        gray = self._to_grayscale(frame)

        # ── Bước 2: Gaussian Blur ───────────────
        blurred = self._apply_blur(gray)

        # ── Bước 3: Canny Edge Detection ────────
        edges = self._apply_canny(blurred)

        # ── Bước 4: ROI ─────────────────────────
        roi_edges = self._apply_roi(edges)

        # ── Bước 5: Hough Lines ──────────────────
        lines = self._detect_lines(roi_edges)

        if lines is None:
            # Không tìm thấy đường thẳng nào → giữ nguyên góc cũ
            result.steering_angle = self._angle_buffer[-1] if self._angle_buffer else 0.0
            result.debug_frame    = frame
            return result

        # ── Bước 6: Trung bình hóa đường thẳng ─
        left_line, right_line = self._average_lines(lines, W, H)

        # ── Bước 7: Tính góc lái ──────────────
        steering = self._compute_steering_angle(left_line, right_line, W, H)

        # ── Vẽ debug ─────────────────────────
        debug_frame = self._draw_debug(frame, left_line, right_line, steering)

        result.steering_angle = steering
        result.left_line      = left_line
        result.right_line     = right_line
        result.debug_frame    = debug_frame
        return result


# ═══════════════════════════════════════════════════════════════
# SECTION 5: MODULE NHẬN DIỆN BIỂN BÁO (TRAFFIC SIGN DETECTION)
# ═══════════════════════════════════════════════════════════════

class TrafficSignDetector:
    """
    Nhận diện biển báo giao thông dùng YOLOv8n ONNX qua cv2.dnn.
    Chạy trong thread riêng biệt để không block luồng bám làn đường.

    Hướng dẫn export mô hình:
        pip install ultralytics
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        model.export(format="onnx", imgsz=640, opset=12)
    """

    # Ánh xạ từ tên nhãn → hành động điều khiển
    LABEL_TO_ACTION = {
        "stop":        "STOP",
        "turn_left":   "TURN_LEFT",
        "turn_right":  "TURN_RIGHT",
        "speed_limit": "SLOW_DOWN",
        "crosswalk":   "SLOW_DOWN",
    }

    def __init__(self, config: Config):
        self.cfg = config
        self.net = None
        self._frame_count = 0
        self._last_result = SignResult()
        self._load_model()

    def _load_model(self):
        """
        Load mô hình ONNX vào bộ nhớ thông qua cv2.dnn.
        cv2.dnn hỗ trợ tăng tốc bằng OpenCL (GPU/NPU) nếu có.
        """
        try:
            logger.info(f"Đang load YOLO model: {self.cfg.YOLO_MODEL_PATH}")
            self.net = cv2.dnn.readNetFromONNX(self.cfg.YOLO_MODEL_PATH)

            # Thử dùng OpenCL để tăng tốc (nếu có GPU/NPU)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            # Uncomment để dùng OpenCL trên GPU (nếu hỗ trợ):
            # self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

            logger.info("YOLO model load thanh cong.")
        except Exception as e:
            logger.error(f"Khong load duoc model YOLO: {e}")
            logger.warning("  -> Kiem tra duong dan: " + self.cfg.YOLO_MODEL_PATH)
            self.net = None

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Tiền xử lý frame cho YOLO:
        1. Resize về kích thước đầu vào của mô hình (640x640)
        2. Chuẩn hóa pixel về [0, 1]
        3. Chuyển từ HWC sang NCHW (batch, channel, height, width)
        """
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0 / 255.0,    # Chuẩn hóa [0,255] → [0,1]
            size=self.cfg.YOLO_INPUT_SIZE,
            mean=(0, 0, 0),              # Không trừ mean (YOLOv8 không cần)
            swapRB=True,                 # BGR → RGB
            crop=False
        )
        return blob

    def _postprocess(self, outputs: np.ndarray, orig_w: int, orig_h: int) -> SignResult:
        """
        Hậu xử lý đầu ra của YOLOv8 ONNX:
        - YOLOv8 output shape: (1, num_classes+4, num_anchors)
        - Mỗi anchor: [cx, cy, w, h, conf_class0, conf_class1, ...]

        Các bước:
        1. Transpose output để duyệt theo từng detection
        2. Lọc theo confidence threshold
        3. Áp dụng Non-Maximum Suppression (NMS)
        4. Trả về detection có confidence cao nhất
        """
        output = outputs[0]             # Bỏ batch dimension
        output = np.squeeze(output)     # (4+num_cls, num_det) hoặc (num_det, 4+num_cls)

        # Transpose nếu cần để có shape (num_det, 4+num_cls)
        if output.shape[0] < output.shape[1]:
            output = output.T

        num_classes = len(self.cfg.YOLO_CLASSES)
        input_w, input_h = self.cfg.YOLO_INPUT_SIZE

        # Tỷ lệ scale để chuyển tọa độ từ không gian mô hình về không gian ảnh gốc
        scale_x = orig_w / input_w
        scale_y = orig_h / input_h

        boxes, scores, class_ids = [], [], []

        for det in output:
            class_scores = det[4:4 + num_classes]
            max_score    = float(np.max(class_scores))
            class_id     = int(np.argmax(class_scores))

            if max_score < self.cfg.YOLO_CONF_THRESHOLD:
                continue

            # Chuyển từ [cx, cy, w, h] → [x, y, w, h] (góc trên trái)
            cx, cy, bw, bh = det[:4]
            x = int((cx - bw / 2) * scale_x)
            y = int((cy - bh / 2) * scale_y)
            w = int(bw * scale_x)
            h = int(bh * scale_y)

            boxes.append([x, y, w, h])
            scores.append(max_score)
            class_ids.append(class_id)

        if not boxes:
            return SignResult()

        # NMS: loại bỏ các bounding box trùng lặp
        indices = cv2.dnn.NMSBoxes(
            boxes, scores,
            self.cfg.YOLO_CONF_THRESHOLD,
            self.cfg.YOLO_NMS_THRESHOLD
        )

        if len(indices) == 0:
            return SignResult()

        # Chọn detection có confidence cao nhất sau NMS
        best_idx  = indices[0] if isinstance(indices[0], (int, np.integer)) else indices[0][0]
        best_cls  = class_ids[best_idx]
        best_conf = scores[best_idx]
        best_box  = boxes[best_idx]
        label     = self.cfg.YOLO_CLASSES[best_cls]
        action    = self.LABEL_TO_ACTION.get(label, "CONTINUE")

        return SignResult(
            label=label,
            confidence=best_conf,
            bbox=tuple(best_box),
            action=action
        )

    def detect_signs(self, frame: np.ndarray) -> SignResult:
        """
        Phát hiện biển báo trong một frame.
        Được gọi mỗi N frame (YOLO_INFERENCE_EVERY_N_FRAMES) để tiết kiệm CPU.
        Giữa các lần inference, trả về kết quả lần trước.
        """
        self._frame_count += 1

        # Skip frame để giảm tải CPU (YOLO tốn ~80-200ms/frame trên RPi4)
        if self._frame_count % self.cfg.YOLO_INFERENCE_EVERY_N_FRAMES != 0:
            return self._last_result

        if self.net is None:
            return SignResult()

        H, W = frame.shape[:2]

        try:
            blob    = self._preprocess(frame)
            self.net.setInput(blob)

            # Forward pass – đây là bước tốn thời gian nhất
            t_start = time.perf_counter()
            outputs = self.net.forward()
            t_end   = time.perf_counter()
            logger.debug(f"YOLO inference: {(t_end - t_start)*1000:.1f}ms")

            result = self._postprocess(outputs, W, H)
            self._last_result = result
            return result

        except Exception as e:
            logger.warning(f"Loi YOLO inference: {e}")
            return self._last_result


# ═══════════════════════════════════════════════════════════════
# SECTION 6: GIAO TIẾP PHẦN CỨNG (HARDWARE INTERFACE)
# ═══════════════════════════════════════════════════════════════

def send_control_signal(speed: float, steering_angle: float, brake: bool = False):
    """
    [PLACEHOLDER] Gửi tín hiệu điều khiển xuống vi điều khiển qua UART/I2C.

    Trong hệ thống thực, hàm này sẽ:
    1. Đóng gói dữ liệu (speed, steering_angle, brake) vào giao thức tùy chỉnh
    2. Gửi qua UART (pyserial) hoặc I2C (smbus2)

    Ví dụ giao thức UART:
        Packet: [0xAA] [speed_byte] [steering_byte] [brake_byte] [checksum] [0x55]

    Triển khai UART thực (bỏ comment khi ready):
    ---------------------------------------------------
        import serial
        ser = serial.Serial(Config.UART_PORT, Config.UART_BAUDRATE, timeout=0.01)

        speed_byte    = int(speed * 255)
        steering_byte = int((steering_angle + 90) * 255 / 180)
        brake_byte    = 0xFF if brake else 0x00
        checksum      = (speed_byte + steering_byte + brake_byte) & 0xFF

        packet = bytes([0xAA, speed_byte, steering_byte, brake_byte, checksum, 0x55])
        ser.write(packet)
    ---------------------------------------------------

    Args:
        speed:          Tốc độ chuẩn hóa [0.0, 1.0]. 0 = dừng, 1 = tối đa
        steering_angle: Góc lái (độ). -90 = rẽ trái hết, 0 = thẳng, +90 = rẽ phải hết
        brake:          True = phanh khẩn cấp
    """
    signal = ControlSignal(speed=speed, steering_angle=steering_angle, brake=brake)

    if brake:
        logger.warning(f"BRAKE! speed={speed:.2f}, angle={steering_angle:+.1f} deg")
    else:
        logger.debug(
            f"Ctrl: speed={signal.speed:.2f} | "
            f"steer={signal.steering_angle:+.1f} deg | "
            f"brake={signal.brake}"
        )


# ═══════════════════════════════════════════════════════════════
# SECTION 7: LOGIC QUYẾT ĐỊNH (DECISION MAKING)
# ═══════════════════════════════════════════════════════════════

class DecisionMaker:
    """
    Kết hợp kết quả bám làn và nhận diện biển báo
    để đưa ra quyết định điều khiển xe.
    """

    BASE_SPEED = 0.4        # Tốc độ bình thường (40% công suất)
    SLOW_SPEED = 0.2        # Tốc độ chậm (20%) khi có biển cảnh báo
    STOP_SPEED = 0.0        # Dừng hẳn

    def decide(self, lane_result: LaneResult, sign_result: SignResult) -> ControlSignal:
        """
        Kết hợp tín hiệu làn đường + biển báo → tín hiệu điều khiển.
        Biển báo luôn có độ ưu tiên cao hơn bám làn.
        """
        action   = sign_result.action
        steering = lane_result.steering_angle
        brake    = False

        if action == "STOP":
            speed    = self.STOP_SPEED
            brake    = True
            steering = 0.0
        elif action == "TURN_LEFT":
            speed    = self.SLOW_SPEED
            steering = -45.0
        elif action == "TURN_RIGHT":
            speed    = self.SLOW_SPEED
            steering = +45.0
        elif action == "SLOW_DOWN":
            speed = self.SLOW_SPEED
        else:
            speed = self.BASE_SPEED

        return ControlSignal(speed=speed, steering_angle=steering, brake=brake)


# ═══════════════════════════════════════════════════════════════
# SECTION 8: PIPELINE CHÍNH (MAIN PIPELINE)
# ═══════════════════════════════════════════════════════════════

class AutonomousCarPipeline:
    """
    Pipeline chính điều phối toàn bộ hệ thống xe tự hành.

    Kiến trúc luồng:
    - Thread Camera:  đọc frame từ camera → SharedFrameBuffer
    - Thread Lane:    đọc frame → LaneTracker (nhanh, ~30fps)
    - Thread Sign:    đọc frame → YOLO (chậm, ~5fps, không block Lane)
    - Thread Control: kết hợp kết quả → send_control_signal (20Hz)
    - Main Thread:    hiển thị debug, bắt phím thoát
    """

    def __init__(self, config: Config):
        self.cfg     = config
        self.running = threading.Event()
        self.running.set()

        self.frame_buffer   = SharedFrameBuffer()
        self.display_buffer = SharedFrameBuffer()

        self.lane_tracker   = LaneTracker(config)
        self.sign_detector  = TrafficSignDetector(config)
        self.decision_maker = DecisionMaker()

        self._lane_result = LaneResult()
        self._sign_result = SignResult()
        self._lock_lane   = threading.Lock()
        self._lock_sign   = threading.Lock()

        self._lane_fps_counter = 0
        self._lane_fps_time    = time.time()
        self._lane_fps         = 0.0

    def _camera_thread(self):
        """Thread chuyên đọc frame từ camera."""
        logger.info("Camera thread bat dau.")
        cap = cv2.VideoCapture(self.cfg.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.cfg.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          self.cfg.TARGET_FPS)

        if not cap.isOpened():
            logger.error("Khong mo duoc camera! Kiem tra CAMERA_INDEX.")
            self.running.clear()
            return

        logger.info("Camera mo thanh cong.")

        while self.running.is_set():
            ret, frame = cap.read()
            if not ret:
                logger.warning("Doc frame that bai, thu lai...")
                time.sleep(0.01)
                continue
            self.frame_buffer.write(frame)

        cap.release()
        logger.info("Camera thread dung.")

    def _lane_tracking_thread(self):
        """
        Thread chạy liên tục, xử lý bám làn đường (~25-30fps trên RPi4).
        """
        logger.info("Lane tracking thread bat dau.")

        while self.running.is_set():
            frame = self.frame_buffer.read(timeout=0.1)
            if frame is None:
                continue

            result = self.lane_tracker.process_lane(frame)

            with self._lock_lane:
                self._lane_result = result

            if result.debug_frame is not None:
                self.display_buffer.write(result.debug_frame)

            # Tính FPS
            self._lane_fps_counter += 1
            elapsed = time.time() - self._lane_fps_time
            if elapsed >= 2.0:
                self._lane_fps = self._lane_fps_counter / elapsed
                self._lane_fps_counter = 0
                self._lane_fps_time = time.time()
                logger.info(f"Lane FPS: {self._lane_fps:.1f}")

        logger.info("Lane tracking thread dung.")

    def _sign_detection_thread(self):
        """
        Thread nhận diện biển báo YOLO.
        Chạy chậm hơn nhưng KHÔNG ảnh hưởng đến luồng bám làn.
        """
        logger.info("Sign detection thread bat dau.")

        while self.running.is_set():
            frame = self.frame_buffer.read_latest()
            if frame is None:
                time.sleep(0.05)
                continue

            result = self.sign_detector.detect_signs(frame)

            if result.label != "NONE":
                logger.info(
                    f"Bien bao: [{result.label}] "
                    f"conf={result.confidence:.2f} "
                    f"action={result.action}"
                )

            with self._lock_sign:
                self._sign_result = result

            # Nghỉ ngắn để nhường CPU cho lane tracking
            time.sleep(0.05)

        logger.info("Sign detection thread dung.")

    def _control_thread(self):
        """
        Thread gửi tín hiệu điều khiển ở tần suất cố định 20Hz.
        """
        logger.info("Control thread bat dau.")
        control_interval = 0.05  # 20Hz

        while self.running.is_set():
            t_start = time.perf_counter()

            with self._lock_lane:
                lane = self._lane_result
            with self._lock_sign:
                sign = self._sign_result

            control = self.decision_maker.decide(lane, sign)

            send_control_signal(
                speed=control.speed,
                steering_angle=control.steering_angle,
                brake=control.brake
            )

            elapsed    = time.perf_counter() - t_start
            sleep_time = max(0.0, control_interval - elapsed)
            time.sleep(sleep_time)

        logger.info("Control thread dung.")

    def main_pipeline(self):
        """
        Khởi động tất cả các thread và giám sát hệ thống.
        Nhấn 'q' để thoát an toàn.
        """
        logger.info("=" * 60)
        logger.info("He thong xe tu hanh dang khoi dong...")
        logger.info("=" * 60)

        threads = [
            threading.Thread(target=self._camera_thread,
                             name="Camera",     daemon=True),
            threading.Thread(target=self._lane_tracking_thread,
                             name="LaneTrack",  daemon=True),
            threading.Thread(target=self._sign_detection_thread,
                             name="SignDetect", daemon=True),
            threading.Thread(target=self._control_thread,
                             name="Control",    daemon=True),
        ]

        for t in threads:
            t.start()
            logger.info(f"  Thread '{t.name}' da khoi dong.")

        logger.info("\nHe thong dang chay. Nhan 'q' de thoat.\n")

        try:
            while self.running.is_set():
                display_frame = self.display_buffer.read(timeout=0.05)

                if display_frame is not None:
                    with self._lock_sign:
                        sign = self._sign_result

                    if sign.label != "NONE":
                        cv2.putText(
                            display_frame,
                            f"Sign: {sign.label} ({sign.confidence:.0%})",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 165, 255), 2
                        )
                        if sign.bbox:
                            x, y, w, h = sign.bbox
                            cv2.rectangle(display_frame,
                                          (x, y), (x+w, y+h),
                                          (0, 165, 255), 2)

                    cv2.putText(
                        display_frame,
                        f"Lane FPS: {self._lane_fps:.1f}",
                        (display_frame.shape[1] - 160, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
                    )

                    cv2.imshow("Autonomous Car - Debug View", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Nhan 'q' - Dang dung he thong...")
                    self.running.clear()
                    break

        except KeyboardInterrupt:
            logger.info("Ctrl+C - Dang dung he thong...")
            self.running.clear()

        finally:
            for t in threads:
                t.join(timeout=2.0)
                if t.is_alive():
                    logger.warning(f"Thread '{t.name}' chua ket thuc sau 2s.")

            cv2.destroyAllWindows()
            logger.info("He thong da dung an toan.")


# ═══════════════════════════════════════════════════════════════
# SECTION 9: ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Điểm vào chương trình.

    Trước khi chạy, đảm bảo:
    1. Đã cài thư viện: pip3 install opencv-python-headless numpy
    2. Đã có file model: yolov8n_traffic.onnx (xem hướng dẫn trong TrafficSignDetector)
    3. Camera được kết nối đúng CAMERA_INDEX

    Chạy thử với video file (không cần camera thật):
        Thay CAMERA_INDEX = "test_video.mp4" trong Config
    """
    config   = Config()
    pipeline = AutonomousCarPipeline(config)
    pipeline.main_pipeline()
