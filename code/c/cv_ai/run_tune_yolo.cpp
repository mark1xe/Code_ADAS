#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <csignal>
#include <ctime>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>   

using namespace std;
using namespace cv;

// ============================================================
//  THÔNG SỐ CAMERA & ÁNH SÁNG  
// ============================================================
constexpr int   CAM_W              = 320;
constexpr int   CAM_H              = 240;
constexpr int   CAM_FIXED_EXPOSURE = 30000;
constexpr float CAM_ANALOGUE_GAIN  = 4.0f;
constexpr float SW_BRIGHTNESS      = 1.2f;

constexpr int HSV_H_MIN = 0,   HSV_H_MAX = 180;
constexpr int HSV_S_MIN = 0,   HSV_S_MAX = 85;
constexpr int HSV_V_MIN = 60,  HSV_V_MAX = 255; // Hạ V_MIN xuống 60 cho sáng

// ============================================================
//  THÔNG SỐ XE  
// ============================================================
constexpr int   DRIVE_SPEED        = 12;
constexpr int   SLOW_SPEED         = 6;
constexpr int   ZEBRA_SPEED        = 8;
constexpr int   PERSON_SPEED       = 4;  
constexpr int   ZEBRA_HOLD_FRAMES  = 60;
constexpr float SMOOTH_ALPHA       = 0.75f;
constexpr float DEADBAND           = 0.04f;
constexpr float MAX_OFFSET_JUMP    = 0.65f;

// ============================================================
//  ROI & CV PARAMS  
// ============================================================
constexpr float ROI_TOP_LEFT_X  = 0.28f;
constexpr float ROI_TOP_RIGHT_X = 0.85f;
constexpr float ROI_TOP_Y       = 0.42f;
constexpr float LANE_SLOPE_MIN  = 0.47f;
constexpr float LANE_SLOPE_MAX  = 5.67f;
constexpr int   CANNY_LOW       = 40,  CANNY_HIGH    = 100;
constexpr int   HOUGH_THR       = 30,  HOUGH_MIN_LEN = 20, HOUGH_MAX_GAP = 50;

// =============================================================================
//  ZEBRA DETECTION PARAMS
// =============================================================================
constexpr float ZEBRA_ROI_X0         = 0.25f;
constexpr float ZEBRA_ROI_X1         = 0.75f;
constexpr float ZEBRA_ROI_Y0         = 0.50f;
constexpr float ZEBRA_ROI_Y1         = 0.92f;
constexpr int   ZEBRA_MIN_STRIPES    = 3;
constexpr float ZEBRA_STRIPE_ASPECT  = 1.8f;
constexpr float ZEBRA_MIN_AREA       = 200.0f;
constexpr float ZEBRA_GAP_STD_MAX    = 8.0f;
constexpr int   ZEBRA_CONFIRM_FRAMES = 2;

// ============================================================
//  YOLO CONFIG
// ============================================================
constexpr int   YOLO_INPUT_W    = 320;
constexpr int   YOLO_INPUT_H    = 320;
constexpr float YOLO_CONF_THR   = 0.25f;   
constexpr float YOLO_NMS_THR    = 0.45f;   
constexpr int   YOLO_NUM_CLASS  = 6;
const char* YOLO_MODEL_PATH = "/home/leanhquan/export/best.onnx";

// Class names 
const vector<string> CLASS_NAMES = {
    "stop_sign", "no_entry", "red_light",
    "yellow_light", "green_light", "person"
};
// Màu debug cho từng class (BGR)
const vector<Scalar> CLASS_COLORS = {
    Scalar(0,0,220),     // stop_sign
    Scalar(0,60,200),    // no_entry
    Scalar(0,0,255),     // red_light
    Scalar(0,200,255),   // yellow_light
    Scalar(0,200,50),    // green_light
    Scalar(255,150,0),   // person
};

// ============================================================
//  KỊCH BẢN HÀNH VI (Vehicle Behavior Scenario)
// ============================================================
enum class Scenario {
    NORMAL      = 0,
    STOP_SIGN   = 1,
    NO_ENTRY    = 2,
    RED_LIGHT   = 3,
    YELLOW_LIGHT= 4,
    GREEN_LIGHT = 5,
    PERSON_AHEAD= 6,
};

// ============================================================
//  SHARED STATE
// ============================================================
atomic<int>   shared_steer(0);
atomic<int>   shared_speed(0);
atomic<bool>  shared_running(false);
atomic<bool>  app_exit(false);

// Camera → Vision
Mat        cap_frame;
mutex      cap_mtx;
condition_variable cap_cv;
bool       cap_ready = false;

// Kết quả YOLO detection
struct YoloDetection {
    int   class_id;
    float conf;
    Rect  box;
};

struct VisionResult {
    bool  lane_found  = false;
    float lane_offset = 0.0f;
    bool  is_zebra    = false;
    vector<YoloDetection> detections;
    Scenario scenario    = Scenario::NORMAL;
    Mat   raw_frame;
    Mat   debug_frame;
    Mat   mask_frame;
};
VisionResult vis_result;
mutex        vis_mtx;
condition_variable vis_cv;
bool         vis_ready = false;

mutex algo_mtx;
float _ema_lane_width    = 160.0f;
float _prev_valid_offset = 0.0f;
int   _zebra_hold_cnt    = 0;

// ============================================================
//  YOLO THREAD — shared queue (vision → yolo thread)
// ============================================================
Mat        yolo_input_frame;
mutex      yolo_in_mtx;
condition_variable yolo_in_cv;
bool       yolo_in_ready = false;

vector<YoloDetection> yolo_last_dets;
Scenario              yolo_last_sc  = Scenario::NORMAL;
mutex                 yolo_out_mtx;

int uart_fd = -1;

// ============================================================
//  UART THREAD
// ============================================================
bool init_uart(const char* port) {
    uart_fd = open(port, O_RDWR | O_NOCTTY | O_NDELAY);
    if (uart_fd < 0) return false;
    struct termios opt;
    tcgetattr(uart_fd, &opt);
    cfsetispeed(&opt, B115200);
    cfsetospeed(&opt, B115200);
    opt.c_cflag |= (CLOCAL | CREAD);
    opt.c_cflag &= ~(PARENB | CSTOPB | CSIZE); opt.c_cflag |= CS8;
    opt.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    opt.c_oflag &= ~OPOST;
    opt.c_cc[VMIN] = 0; opt.c_cc[VTIME] = 1;
    tcsetattr(uart_fd, TCSANOW, &opt);
    return true;
}

void uart_thread_func() {
    using clk = chrono::steady_clock;
    auto next = clk::now();
    while (!app_exit) {
        next += chrono::milliseconds(100);
        this_thread::sleep_until(next);
        if (uart_fd < 0) continue;
        int  st  = shared_steer.load(memory_order_relaxed);
        int  sp  = shared_speed.load(memory_order_relaxed);
        bool run = shared_running.load(memory_order_relaxed);
        char buf[32];
        int  n = run ? snprintf(buf, sizeof(buf), "F,%d,%d\n", st, sp)
                     : snprintf(buf, sizeof(buf), "S,0,0\n");
        write(uart_fd, buf, n);
    }
}

// ============================================================
//  CAMERA CAPTURE THREAD
// ============================================================
void capture_thread_func() {
    // TỐI ƯU CAMERA: CAP_ANY để C++ tự động chọn backend y hệt Python
    VideoCapture cap(0, CAP_ANY); 
    
    if (!cap.isOpened()) {
        cerr << "[CAM] Không mở được camera!\n";
        app_exit = true; 
        cap_cv.notify_all(); vis_cv.notify_all(); yolo_in_cv.notify_all();
        return;
    }
    
    // TỐI ƯU CROP: Set FOV rộng 640x480 để tránh libcamerify Center Crop
    cap.set(CAP_PROP_FRAME_WIDTH,  640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(CAP_PROP_FPS, 30);
    
    // Tuyệt đối không set BUFFERSIZE trên Pi OS Bookworm
    // Tuyệt đối không dùng v4l2-ctl để tránh chết luồng I2C

    Mat tmp;
    while (!app_exit) {
        if (!cap.read(tmp) || tmp.empty()) {
            cerr << "[CAM] Đang chờ frame...\n";
            this_thread::sleep_for(chrono::milliseconds(100));
            continue;
        }
        
        // TỐI ƯU RESIZE: Dùng INTER_NEAREST siêu tốc
        if (tmp.cols != CAM_W || tmp.rows != CAM_H) {
            resize(tmp, tmp, Size(CAM_W, CAM_H), 0, 0, INTER_NEAREST);
        }
            
        {
            lock_guard<mutex> lk(cap_mtx);
            tmp.copyTo(cap_frame);
            cap_ready = true;
        }
        cap_cv.notify_one();
    }
    cap.release();
}

// ============================================================
//  VISION — LANE DETECTION
// ============================================================
Mat compute_white_mask(const Mat& bgr) {
    Mat hsv, blur_img, mask;
    
    // TỐI ƯU CPU: Bỏ CLAHE, dùng Blur nhẹ 3x3
    GaussianBlur(bgr, blur_img, Size(3, 3), 0); 
    cvtColor(blur_img, hsv, COLOR_BGR2HSV);
    
    inRange(hsv,
            Scalar(HSV_H_MIN, HSV_S_MIN, HSV_V_MIN),
            Scalar(HSV_H_MAX, HSV_S_MAX, HSV_V_MAX),
            mask);
    return mask;
}

// =============================================================================
//  ZEBRA DETECTION
// =============================================================================
bool detect_zebra_from_mask(const Mat& white_mask, int W, int H) {
    int x0 = int(W * ZEBRA_ROI_X0);
    int x1 = int(W * ZEBRA_ROI_X1);
    int y0 = int(H * ZEBRA_ROI_Y0);
    int y1 = int(H * ZEBRA_ROI_Y1);
    int rw = x1 - x0, rh = y1 - y0;
    if (rw <= 0 || rh <= 0) return false;

    Mat roi = white_mask(Rect(x0, y0, rw, rh));

    Mat roi_clean;
    Mat kernel_h = getStructuringElement(MORPH_RECT, Size(9, 1));
    Mat kernel_v = getStructuringElement(MORPH_RECT, Size(1, 3));
    morphologyEx(roi, roi_clean, MORPH_CLOSE,  kernel_h);
    morphologyEx(roi_clean, roi_clean, MORPH_OPEN, kernel_v);

    vector<vector<Point>> contours;
    findContours(roi_clean, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<float> stripe_cy;
    for (const auto& cnt : contours) {
        double area = contourArea(cnt);
        if (area < ZEBRA_MIN_AREA) continue;
        Rect bb = boundingRect(cnt);
        float aspect = (float)bb.width / max(bb.height, 1);
        if (aspect < ZEBRA_STRIPE_ASPECT) continue;
        if (bb.height < 5) continue;
        if (bb.x <= 2 || (bb.x + bb.width) >= rw - 2) continue;
        stripe_cy.push_back(bb.y + bb.height / 2.0f);
    }

    if ((int)stripe_cy.size() < ZEBRA_MIN_STRIPES) return false;

    sort(stripe_cy.begin(), stripe_cy.end());
    vector<float> gaps;
    for (int i = 1; i < (int)stripe_cy.size(); i++)
        gaps.push_back(stripe_cy[i] - stripe_cy[i-1]);

    float mean_gap = 0.0f;
    for (float g : gaps) mean_gap += g;
    mean_gap /= gaps.size();
    if (mean_gap < 4.0f) return false;

    float var = 0.0f;
    for (float g : gaps) var += (g - mean_gap) * (g - mean_gap);
    if (sqrtf(var / gaps.size()) > ZEBRA_GAP_STD_MAX) return false;

    static int confirm_cnt = 0;
    confirm_cnt++;
    if (confirm_cnt >= ZEBRA_CONFIRM_FRAMES) return true;
    return false;
}

static void reset_zebra_debounce() {
    static int& cnt = *[]() -> int* { static int c = 0; return &c; }();
    cnt = 0;
}

struct LaneResult { bool found; float offset; };

LaneResult detect_lane_from_mask(const Mat& bgr, const Mat& white_mask,
                                  Mat& dbg, int W, int H, bool zebra_active) {
    LaneResult res{false, 0.0f};
    Mat edges, roi_mask, masked;
    Canny(white_mask, edges, CANNY_LOW, CANNY_HIGH);

    Point tl(int(W * ROI_TOP_LEFT_X),  int(H * ROI_TOP_Y));
    Point tr(int(W * ROI_TOP_RIGHT_X), int(H * ROI_TOP_Y));
    Point bl(0, H), br(W, H);

    roi_mask = Mat::zeros(edges.size(), CV_8UC1);
    Point pts[4] = {bl, tl, tr, br};
    const Point* ppt[1] = {pts};
    int npt[] = {4};
    fillPoly(roi_mask, ppt, npt, 1, Scalar(255));
    bitwise_and(edges, roi_mask, masked);

    dbg = bgr.clone();
    vector<Point> roi_pts = {bl, tl, tr, br};
    polylines(dbg, roi_pts, true, Scalar(255, 0, 255), 1);
    int lookahead_y = int(H * 0.75f);
    line(dbg, Point(0, lookahead_y), Point(W, lookahead_y), Scalar(50, 50, 50), 1);

    vector<Vec4i> lines;
    HoughLinesP(masked, lines, 1, CV_PI / 180.0, HOUGH_THR, HOUGH_MIN_LEN, HOUGH_MAX_GAP);
    if (lines.empty()) return res;

    float cx = W / 2.0f;
    float ly_near = H * 0.85f;
    float ly_far  = H * 0.55f;
    float ew_val;
    { lock_guard<mutex> lk(algo_mtx); ew_val = _ema_lane_width; }

    vector<pair<float,float>> left_xs, right_xs;
    vector<Vec4i> left_lines, right_lines;

    for (const auto& l : lines) {
        float dx = l[2] - l[0], dy = l[3] - l[1];
        float len = hypotf(dx, dy);
        if (len < 1.0f || fabsf(dx) < 0.5f) continue;
        float sa = fabsf(dy / dx);
        if (sa < LANE_SLOPE_MIN || sa > LANE_SLOPE_MAX) continue;
        float xn = l[0] + (ly_near - l[1]) * dx / dy;
        float xf = l[0] + (ly_far  - l[1]) * dx / dy;
        float x_look = 0.6f * xn + 0.4f * xf;
        if (zebra_active && fabsf(x_look - cx) < (ew_val * 0.35f)) continue;
        if (x_look < cx) { left_xs.push_back({x_look, len}); left_lines.push_back(l); }
        else             { right_xs.push_back({x_look, len}); right_lines.push_back(l); }
    }

    auto wmean = [](const vector<pair<float,float>>& v) {
        float sx = 0, sw = 0;
        for (auto& p : v) { sx += p.first * p.second; sw += p.second; }
        return sw > 0 ? sx / sw : 0.0f;
    };

    bool hl = !left_xs.empty(), hr = !right_xs.empty();
    if (!hl && !hr) return res;

    float xl = hl ? wmean(left_xs)  : 0.0f;
    float xr = hr ? wmean(right_xs) : 0.0f;
    float lane_center; bool single = false;

    if (hl && hr) {
        float cw = xr - xl;
        { lock_guard<mutex> lk(algo_mtx);
          if (cw > 80 && cw < 280) _ema_lane_width = 0.1f * cw + 0.9f * _ema_lane_width; }
        lane_center = (xl + xr) / 2.0f;
    } else {
        float ew; { lock_guard<mutex> lk(algo_mtx); ew = _ema_lane_width; }
        lane_center = hl ? xl + ew / 2.0f : xr - ew / 2.0f;
        single = true;
    }

    float offset = (lane_center - cx) / cx;
    offset = max(-1.0f, min(1.0f, offset));
    if (single) offset = max(-1.0f, min(1.0f, offset * 1.3f));
    res.found = true; res.offset = offset;

    for (const auto& l : left_lines)  line(dbg, Point(l[0],l[1]), Point(l[2],l[3]), Scalar(0,255,0), 2);
    for (const auto& l : right_lines) line(dbg, Point(l[0],l[1]), Point(l[2],l[3]), Scalar(0,100,255), 2);
    if (hl) circle(dbg, Point(int(xl), lookahead_y), 4, Scalar(255,255,255), -1);
    if (hr) circle(dbg, Point(int(xr), lookahead_y), 4, Scalar(255,255,255), -1);
    line(dbg, Point(int(lane_center), H-10), Point(int(lane_center), int(H*ROI_TOP_Y)), Scalar(0,255,255), 2);
    line(dbg, Point(int(cx),          H-10), Point(int(cx),          int(H*ROI_TOP_Y)), Scalar(128,128,128), 1);

    char tag[50]; int z_cnt;
    { lock_guard<mutex> lk(algo_mtx); ew_val = _ema_lane_width; z_cnt = _zebra_hold_cnt; }
    snprintf(tag, sizeof(tag), "off=%+5.3f W=%d", offset, int(ew_val));
    putText(dbg, tag, Point(5, H-5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,0), 1);
    if (z_cnt > 0) {
        char z_tag[30]; snprintf(z_tag, sizeof(z_tag), "ZEBRA hold=%d", z_cnt);
        putText(dbg, z_tag, Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(0,0,255), 2);
    }
    return res;
}

// ============================================================
//  YOLO ONNX INFERENCE
// ============================================================
static dnn::Net g_yolo_net;
static bool     g_yolo_ok = false;

bool init_yolo(const char* model_path) {
    try {
        g_yolo_net = dnn::readNetFromONNX(model_path);
        g_yolo_net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
        g_yolo_net.setPreferableTarget(dnn::DNN_TARGET_CPU);
        g_yolo_ok = true;

        Mat dummy(YOLO_INPUT_H, YOLO_INPUT_W, CV_8UC3, Scalar(128,128,128));
        Mat test_blob;
        dnn::blobFromImage(dummy, test_blob, 1.0/255.0,
                           Size(YOLO_INPUT_W, YOLO_INPUT_H), Scalar(), true, false, CV_32F);
        g_yolo_net.setInput(test_blob);
        Mat test_out = g_yolo_net.forward();
        printf("[YOLO] Output shape: [%d, %d, %d]\n",
               test_out.size[0], test_out.size[1], test_out.size[2]);
        cout << "[YOLO] Loaded: " << model_path << "\n";
    } catch (const Exception& e) {
        cerr << "[YOLO] Load failed: " << e.what() << "\n";
        g_yolo_ok = false;
    }
    return g_yolo_ok;
}

vector<YoloDetection> run_yolo(const Mat& bgr) {
    vector<YoloDetection> results;
    if (!g_yolo_ok || bgr.empty()) return results;

    // 1. PADDING CHUẨN (Letterbox)
    Mat square_img(YOLO_INPUT_H, YOLO_INPUT_W, CV_8UC3, Scalar(114, 114, 114));
    int pad_y = (YOLO_INPUT_H - bgr.rows) / 2; // (320 - 240) / 2 = 40
    bgr.copyTo(square_img(Rect(0, pad_y, bgr.cols, bgr.rows)));

    // 2. CHẠY MODEL
    Mat blob;
    dnn::blobFromImage(square_img, blob, 1.0 / 255.0,
                       Size(YOLO_INPUT_W, YOLO_INPUT_H),
                       Scalar(), true, false, CV_32F);
    g_yolo_net.setInput(blob);
    Mat raw_out = g_yolo_net.forward();

    // 3. ĐỌC TENSOR AN TOÀN BẰNG MAT.T()
    int dim1 = raw_out.size[1];
    int dim2 = raw_out.size[2];
    
    Mat out_tensor(dim1, dim2, CV_32F, raw_out.ptr<float>());
    
    if (dim1 < dim2) {
        out_tensor = out_tensor.t(); 
    }

    vector<int>   class_ids;
    vector<float> confidences;
    vector<Rect>  boxes;
    float frame_max_conf = 0.0f; 
    int frame_best_cls = -1;
    
    for (int i = 0; i < out_tensor.rows; ++i) {
        float* row = out_tensor.ptr<float>(i);
        
        float max_conf = 0.0f; 
        int best_cls = -1;
        for (int c = 0; c < YOLO_NUM_CLASS; ++c) {
            if (row[4 + c] > max_conf) {
                max_conf = row[4 + c];
                best_cls = c;
            }
        }
        
        if (max_conf > frame_max_conf) {
            frame_max_conf = max_conf;
            frame_best_cls = best_cls;
        }

        if (max_conf < YOLO_CONF_THR || best_cls < 0) continue;

        float cx = row[0];
        float cy = row[1] - pad_y; 
        float w  = row[2];
        float h  = row[3];

        int x1 = max(0, (int)(cx - w / 2));
        int y1 = max(0, (int)(cy - h / 2));
        int bw = min((int)w, bgr.cols - x1);
        int bh = min((int)h, bgr.rows - y1);
        if (bw <= 0 || bh <= 0) continue;

        class_ids.push_back(best_cls);
        confidences.push_back(max_conf);
        boxes.push_back(Rect(x1, y1, bw, bh));
    }

    // In debug để tinh chỉnh Threshold
    // string best_name = (frame_best_cls >= 0 && frame_best_cls < CLASS_NAMES.size()) 
    //                    ? CLASS_NAMES[frame_best_cls] : "none";
    // printf("\n[YOLO Debug] Max Conf: %.3f (%s) | Boxes: %d\n", 
    //         frame_max_conf, best_name.c_str(), (int)boxes.size());

    vector<int> keep;
    dnn::NMSBoxes(boxes, confidences, YOLO_CONF_THR, YOLO_NMS_THR, keep);
    for (int idx : keep)
        results.push_back({class_ids[idx], confidences[idx], boxes[idx]});
        
    return results;
}

void draw_detections(Mat& dbg, const vector<YoloDetection>& dets) {
    for (const auto& d : dets) {
        if (d.class_id < 0 || d.class_id >= (int)CLASS_NAMES.size()) continue;
        const Scalar& col = CLASS_COLORS[d.class_id];
        rectangle(dbg, d.box, col, 2);
        char label[64];
        snprintf(label, sizeof(label), "%s %.2f", CLASS_NAMES[d.class_id].c_str(), d.conf);
        int base; Size ts = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.45, 1, &base);
        Point tl(d.box.x, max(0, d.box.y - ts.height - 3));
        rectangle(dbg, tl, Point(tl.x + ts.width + 2, tl.y + ts.height + base + 3), col, FILLED);
        putText(dbg, label, Point(tl.x + 1, tl.y + ts.height + 1),
                FONT_HERSHEY_SIMPLEX, 0.45, Scalar(255,255,255), 1);
    }
}

// ============================================================
//  KỊCH BẢN XE
// ============================================================
Scenario determine_scenario(const vector<YoloDetection>& dets) {
    constexpr int MIN_BOX_AREA = 150;
    bool has_no_entry = false, has_stop  = false, has_red    = false;
    bool has_yellow   = false, has_green = false, has_person = false;

    for (const auto& d : dets) {
        if (d.box.area() < MIN_BOX_AREA) continue;
        switch (d.class_id) {
            case 0: has_stop     = true; break;
            case 1: has_no_entry = true; break;
            case 2: has_red      = true; break;
            case 3: has_yellow   = true; break;
            case 4: has_green    = true; break;
            case 5: has_person   = true; break;
        }
    }
    if (has_no_entry) return Scenario::NO_ENTRY;
    if (has_stop)     return Scenario::STOP_SIGN;
    if (has_red)      return Scenario::RED_LIGHT;
    if (has_person)   return Scenario::PERSON_AHEAD;
    if (has_yellow)   return Scenario::YELLOW_LIGHT;
    if (has_green)    return Scenario::GREEN_LIGHT;
    return Scenario::NORMAL;
}

const char* scenario_name(Scenario s) {
    switch (s) {
        case Scenario::STOP_SIGN:    return "STOP_SIGN";
        case Scenario::NO_ENTRY:     return "NO_ENTRY!";
        case Scenario::RED_LIGHT:    return "RED_LIGHT";
        case Scenario::YELLOW_LIGHT: return "YELLOW   ";
        case Scenario::GREEN_LIGHT:  return "GREEN    ";
        case Scenario::PERSON_AHEAD: return "PERSON!  ";
        default:                     return "NORMAL   ";
    }
}

// ============================================================
//  YOLO THREAD 
// ============================================================
void yolo_thread_func() {
    while (!app_exit) {
        Mat frame;
        {
            unique_lock<mutex> lk(yolo_in_mtx);
            yolo_in_cv.wait_for(lk, chrono::milliseconds(50),
                                []{ return yolo_in_ready || app_exit.load(); });
            if (app_exit) break;
            if (!yolo_in_ready) continue;
            frame = yolo_input_frame.clone();
            yolo_in_ready = false;
        }
        if (frame.empty()) continue;

        // Xóa hoàn toàn Gamma correction để tránh Domain Shift
        auto dets = run_yolo(frame);
        Scenario sc = determine_scenario(dets);

        {
            lock_guard<mutex> lk(yolo_out_mtx);
            yolo_last_dets = dets;
            yolo_last_sc   = sc;
        }
    }
}

// ============================================================
//  VISION THREAD  
// ============================================================
void vision_thread_func() {
    Mat frame;
    int feed_cnt = 0;   

    while (!app_exit) {
        {
            unique_lock<mutex> lk(cap_mtx);
            bool got = cap_cv.wait_for(lk, chrono::milliseconds(30),
                           []{ return cap_ready || app_exit.load(); });
            if (!got || app_exit) continue;
            cap_frame.copyTo(frame);
            cap_ready = false;
        }
        if (frame.empty()) continue;

        Mat lane_frame;
        if (SW_BRIGHTNESS > 1.0f)
            frame.convertTo(lane_frame, -1, SW_BRIGHTNESS, 0);
        else
            lane_frame = frame;

        Mat wmask = compute_white_mask(lane_frame);
        bool in_zebra;
        { lock_guard<mutex> lk(algo_mtx); in_zebra = (_zebra_hold_cnt > 0); }

        bool zebra = detect_zebra_from_mask(wmask, frame.cols, frame.rows);
        if (!zebra) reset_zebra_debounce();

        Mat dbg;
        LaneResult lane = detect_lane_from_mask(lane_frame, wmask, dbg,
                                                 frame.cols, frame.rows, in_zebra);

        // TỐI ƯU THROTTLE: Chỉ gửi frame cho YOLO mỗi 5 frames
        feed_cnt++;
        if (feed_cnt >= 5) {
            feed_cnt = 0;
            lock_guard<mutex> lk(yolo_in_mtx);
            frame.copyTo(yolo_input_frame);  
            yolo_in_ready = true;
            yolo_in_cv.notify_one();
        }

        vector<YoloDetection> cur_dets;
        Scenario sc;
        {
            lock_guard<mutex> lk(yolo_out_mtx);
            cur_dets = yolo_last_dets;
            sc       = yolo_last_sc;
        }

        draw_detections(dbg, cur_dets);

        {
            Scalar col = (sc == Scenario::NORMAL) ? Scalar(0,200,0) : Scalar(0,0,255);
            char sc_label[32];
            snprintf(sc_label, sizeof(sc_label), "SC:%s", scenario_name(sc));
            putText(dbg, sc_label, Point(frame.cols - 120, 15),
                    FONT_HERSHEY_SIMPLEX, 0.5, col, 2);
        }

        {
            lock_guard<mutex> lk(vis_mtx);
            vis_result.lane_found  = lane.found;
            vis_result.lane_offset = lane.offset;
            vis_result.is_zebra    = zebra;
            vis_result.detections  = cur_dets;
            vis_result.scenario    = sc;
            frame.copyTo(vis_result.raw_frame);
            dbg.copyTo(vis_result.debug_frame);
            wmask.copyTo(vis_result.mask_frame);
            vis_ready = true;
        }
        vis_cv.notify_one();
    }
}

// ============================================================
//  STEERING CALC 
// ============================================================
int offset_to_steer(float offset) {
    if (fabsf(offset) < DEADBAND) return 0;
    int   sign = (offset > 0) ? 1 : -1;
    float x    = min((fabsf(offset) - DEADBAND) / (1.0f - DEADBAND), 1.0f);
    float mag  = 5.0f + 120.0f * powf(x, 1.2f);
    return (int)roundf(sign * min(mag, 100.0f));
}

// ============================================================
//  MAIN 
// ============================================================
int main() {
    cout << "==============================================\n"
         << "  C++ LANE + YOLO DRIVER — TRAFFIC SCENARIOS \n"
         << "==============================================\n"
         << "  a = Bắt đầu tự lái  |  s = Dừng\n"
         << "  c = Chụp ảnh        |  q = Thoát\n"
         << "  r = Reset kịch bản (NO_ENTRY / STOP)\n"
         << "==============================================\n";

    // TỐI ƯU OPENCV: Ép chỉ dùng 2 luồng CPU
    cv::setNumThreads(2);

    init_yolo(YOLO_MODEL_PATH);

    const char* ports[] = {"/dev/serial0", "/dev/ttyAMA0", "/dev/ttyS0"};
    bool uart_ok = false;
    for (auto p : ports)
        if (init_uart(p)) { cout << "[UART] " << p << " @ 115200\n"; uart_ok = true; break; }
    if (!uart_ok) { cerr << "[UART] Lỗi cổng!\n"; return 1; }

    struct termios oldt;
    tcgetattr(STDIN_FILENO, &oldt);
    {
        struct termios newt = oldt;
        newt.c_lflag    &= ~(ICANON | ECHO);
        newt.c_cc[VMIN]  = 0;
        newt.c_cc[VTIME] = 0;
        tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    }

    thread t_capture(capture_thread_func);
    thread t_vision(vision_thread_func);
    thread t_yolo(yolo_thread_func);      
    thread t_uart(uart_thread_func);

    Scenario active_sc     = Scenario::NORMAL;
    bool     sc_locked     = false;
    int      stop_hold_cnt = 0;
    bool     no_entry_halt = false;
    bool     green_released= false;
    int      person_hold   = 0;

    float smooth_steer = 0.0f;
    int   no_lane_cnt  = 0;
    int   zebra_hold   = 0;
    int   fps_cnt      = 0;
    float fps          = 0.0f;
    int   stream_skip  = 0;
    auto  fps_t0       = chrono::steady_clock::now();

    while (!app_exit) {
        char ch = 0;
        read(STDIN_FILENO, &ch, 1);
        if (ch == 'a') {
            shared_running.store(true);
            smooth_steer   = 0.0f;
            no_entry_halt  = false;
            sc_locked      = false;
            stop_hold_cnt  = 0;
            printf("\n>>> TỰ LÁI BẮT ĐẦU\n");
        } else if (ch == 's') {
            shared_running.store(false);
            shared_steer.store(0); shared_speed.store(0);
            smooth_steer = 0.0f;
            printf("\n>>> DỪNG LÁI\n");
        } else if (ch == 'r') {
            no_entry_halt = false;
            sc_locked     = false;
            stop_hold_cnt = 0;
            green_released= false;
            person_hold   = 0;
            printf("\n>>> RESET kịch bản\n");
        } else if (ch == 'q') {
            app_exit = true; break;
        } else if (ch == 'c') {
            time_t rawtime; struct tm* ti; char tb[20];
            time(&rawtime); ti = localtime(&rawtime);
            strftime(tb, sizeof(tb), "%H%M%S", ti);
            string ts(tb);
            VisionResult vr;
            { lock_guard<mutex> lk(vis_mtx); vr = vis_result; }
            if (!vr.raw_frame.empty()) {
                imwrite("snap_" + ts + "_raw.jpg",   vr.raw_frame);
                imwrite("snap_" + ts + "_debug.jpg", vr.debug_frame);
                imwrite("snap_" + ts + "_mask.jpg",  vr.mask_frame);
                printf("\n>>> [SCREENSHOT] Đã lưu.\n");
            }
        }

        VisionResult vr;
        {
            unique_lock<mutex> lk(vis_mtx);
            bool got = vis_cv.wait_for(lk, chrono::milliseconds(30),
                           []{ return vis_ready || app_exit.load(); });
            if (!got) continue;
            vr = vis_result;
            vis_ready = false;
        }
        if (app_exit) break;

        if (!vr.debug_frame.empty()) {
            stream_skip++;
            if (stream_skip % 2 == 0) {
                imshow("DEBUG_STREAM", vr.debug_frame);
                if (waitKey(1) == 'q') { app_exit = true; break; }
            }
        }

        if (vr.is_zebra) zebra_hold = ZEBRA_HOLD_FRAMES;
        else if (zebra_hold > 0) zebra_hold--;
        bool in_zebra = zebra_hold > 0;
        { lock_guard<mutex> lk(algo_mtx); _zebra_hold_cnt = zebra_hold; }

        Scenario detected_sc = vr.scenario;
        int  spd   = DRIVE_SPEED;
        bool do_stop = false;

        if (detected_sc == Scenario::NO_ENTRY && !no_entry_halt) {
            no_entry_halt = true;
            printf("\n>>> [NO_ENTRY] DỪNG hoàn toàn!\n");
        }
        if (no_entry_halt) {
            do_stop = true; spd = 0; active_sc = Scenario::NO_ENTRY;
        }
        else if (detected_sc == Scenario::STOP_SIGN && !sc_locked) {
            sc_locked = true; stop_hold_cnt = 90;
            printf("\n>>> [STOP_SIGN] Dừng 3 giây...\n");
            active_sc = Scenario::STOP_SIGN;
        }
        if (sc_locked && stop_hold_cnt > 0) {
            do_stop = true; spd = 0; stop_hold_cnt--;
            if (stop_hold_cnt == 0) {
                sc_locked = false; active_sc = Scenario::NORMAL;
                printf("\n>>> [STOP_SIGN] Tiếp tục chạy.\n");
            }
        }
        else if (detected_sc == Scenario::RED_LIGHT) {
            do_stop = true; spd = 0; green_released = false;
            active_sc = Scenario::RED_LIGHT;
        }
        else if (detected_sc == Scenario::GREEN_LIGHT) {
            green_released = true; active_sc = Scenario::GREEN_LIGHT; spd = DRIVE_SPEED;
        }
        else if (detected_sc == Scenario::YELLOW_LIGHT) {
            spd = SLOW_SPEED; active_sc = Scenario::YELLOW_LIGHT;
        }
        else if (detected_sc == Scenario::PERSON_AHEAD) {
            person_hold = 60; active_sc = Scenario::PERSON_AHEAD;
        }

        if (person_hold > 0) {
            spd = PERSON_SPEED; person_hold--;
            if (person_hold == 0) active_sc = Scenario::NORMAL;
        }

        if (in_zebra && spd > ZEBRA_SPEED) spd = ZEBRA_SPEED;

        float offset = vr.lane_offset;
        bool  found  = vr.lane_found;

        if (found) {
            lock_guard<mutex> lk(algo_mtx);
            if (!in_zebra && fabsf(offset - _prev_valid_offset) > MAX_OFFSET_JUMP)
                found = false;
            else
                _prev_valid_offset = offset;
        }

        int raw_steer = offset_to_steer(found ? offset : 0.0f);
        smooth_steer  = SMOOTH_ALPHA * raw_steer + (1.0f - SMOOTH_ALPHA) * smooth_steer;
        int steer_val = (int)roundf(smooth_steer);

        if (found) no_lane_cnt = 0; else no_lane_cnt++;
        if (!found && spd > SLOW_SPEED) spd = SLOW_SPEED;
        if (no_lane_cnt > 30) spd = 0;
        if (do_stop) spd = 0;

        shared_steer.store(steer_val, memory_order_relaxed);
        shared_speed.store(shared_running.load() ? spd : 0, memory_order_relaxed);

        fps_cnt++;
        auto now = chrono::steady_clock::now();
        chrono::duration<float> dt = now - fps_t0;
        if (dt.count() >= 1.0f) { fps = fps_cnt / dt.count(); fps_cnt = 0; fps_t0 = now; }

        printf("\r[%s] SC:%-10s fps=%5.1f off=%+5.3f str=%+4d spd=%2d noln=%3d zbr=%2d",
               shared_running.load() ? "RUN " : "STOP",
               scenario_name(active_sc), fps,
               found ? offset : 0.0f, steer_val, spd,
               no_lane_cnt, zebra_hold);
        fflush(stdout);
    }

    app_exit = true;
    cap_cv.notify_all();
    vis_cv.notify_all();
    yolo_in_cv.notify_all();
    if (t_capture.joinable()) t_capture.join();
    if (t_vision.joinable())  t_vision.join();
    if (t_yolo.joinable())    t_yolo.join();
    if (t_uart.joinable())    t_uart.join();

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    if (uart_fd >= 0) { write(uart_fd, "S,0,0\n", 6); close(uart_fd); }
    printf("\nĐã đóng luồng an toàn.\n");
    return 0;
}