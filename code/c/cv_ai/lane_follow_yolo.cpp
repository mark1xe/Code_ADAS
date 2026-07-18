// =============================================================
//  LANE FOLLOW + YOLO  —  Bám làn + nhận diện biển/đèn
//
//  Hành vi:
//    - RED_LIGHT / NO_ENTRY / PERSON : DỪNG khi còn THẤY (tự đi lại khi hết).
//    - STOP_SIGN : DỪNG ~2s rồi ĐI tiếp, bỏ qua biển đó trong ~6s (cooldown).
//    - GREEN / YELLOW / NORMAL : chạy bình thường.
//    - Lái LUÔN bám làn (kể cả lúc dừng) -> resume là đi đúng hướng.
//
//  Giao thức UART:
//    Pi -> ESP32 : "F,<steer>,<speed>\n" / "S,0,0\n"
//    ESP32 -> Pi : "T,<encLeft>,<encRight>,<t_ms_esp32>\n"  (telemetry, ~10Hz, xem main.c)
//
//  Log CSV (1 dòng/khung hình, phục vụ đánh giá định lượng — mục 2/5/6 luận văn):
//    frame,t_ms,fps_vision,fps_capture,fps_yolo,latency_compute_ms,lane_found,off,
//    nbands,steer,spd,no_lane,scenario,force_stop,on_zebra,enc_left,enc_right,enc_t_ms
//
//  Build:
//    g++ -O3 -std=c++17 lane_follow_yolo.cpp -o lane_follow_yolo \
//        `pkg-config --cflags --libs opencv4` -lpthread
//  Chạy:
//    libcamerify ./lane_follow_yolo [đường_dẫn_model.onnx] [đường_dẫn_log.csv]
//    Phím: a=chạy  s=dừng  c=chụp  q=thoát
// =============================================================
#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <ctime>
#include <cstring>
#include <csignal>
#include <sys/socket.h>
#include <netinet/in.h>
#include <fstream>
#include <iomanip>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// ============================================================
//  CẤU HÌNH CHUNG
// ============================================================
constexpr int   CAM_W = 320, CAM_H = 240;    
#define SHOW_WINDOW 0

// --- Vùng quan tâm: chỉ xét nửa dưới ảnh ---
constexpr float ROI_TOP_Y   = 0.45f;

// --- Adaptive threshold ---
constexpr int   ADAPT_BLOCK = 41;
constexpr double ADAPT_C    = -15.0;

// --- Lọc theo SÀN tối ---
constexpr int   FLOOR_MAX   = 70;
constexpr int   FLOOR_DILATE= 25;
constexpr int   MIN_BLOB    = 80;

// --- Sliding window ---
constexpr int   N_BANDS       = 8;
constexpr int   MIN_PIX       = 8;
constexpr float HALF_LANE     = 0.34f;
constexpr float HUG_LANE      = 0.20f;   // chỉ thấy 1 vạch ở CUA -> ôm sát mép trong (<HALF_LANE: cua gắt hơn; giảm nếu vẫn lố)
constexpr float LEAN_TH       = 8.0f;    // px: độ nghiêng vạch (gần->xa) tối thiểu để coi là ĐANG CUA
constexpr float SEARCH_MARGIN = 0.18f;
constexpr int   COL_MIN       = 4;
constexpr int   MIN_BANDS     = 2;
constexpr float INNER_BIAS    = 0.0f;
constexpr float CURVE_TH      = 0.04f;

// --- Lái ---
constexpr float DEADBAND     = 0.04f;
constexpr float STEER_GAIN   = 120.0f;
constexpr float STEER_POW    = 1.0f;
constexpr float SMOOTH_ALPHA = 0.4f;
constexpr float SPEED_STEP   = 1.0f;
#define STEER_INVERT 0

// --- Tốc độ ---
constexpr int   DRIVE_SPEED  = 8;
constexpr int   CURVE_SPEED  = 5;
constexpr int   SLOW_SPEED   = 5;
constexpr bool  ENABLE_CURVE_SPEED = false;
constexpr int   NO_LANE_STOP = 30;

// ============================================================
//  CẤU HÌNH YOLO
// ============================================================
constexpr int   YOLO_INPUT_W   = 320;
constexpr int   YOLO_INPUT_H   = 320;
constexpr float YOLO_CONF_THR  = 0.25f;
constexpr float YOLO_NMS_THR   = 0.45f;
constexpr int   YOLO_NUM_CLASS = 6;

string YOLO_MODEL_PATH = "/home/leanhquan/export/best.onnx";

// Diện tích khung tối thiểu (px^2) để coi đối tượng đủ GẦN -> mới phản ứng
constexpr int   MIN_BOX_AREA   = 200;   // bắt biển/đèn TỪ XA -> kịp dừng trước khi tới 
// Ngưỡng "phiếu" để xác nhận có vật cần DỪNG 
constexpr int   YOLO_STOP_VOTES = 2;
// Đèn đỏ: đã thấy đỏ thì CHỐT dừng và giữ; tự nhả sau ngần này (ms) nếu không thấy xanh (an toàn)
constexpr int   RED_LATCH_TIMEOUT_MS = 8000;

// --- Vạch người đi bộ (zebra): các sọc trắng ngang đều nhau ---
constexpr float ZEBRA_ROI_X0 = 0.25f, ZEBRA_ROI_X1 = 0.75f;   
constexpr float ZEBRA_ROI_Y0 = 0.50f, ZEBRA_ROI_Y1 = 0.92f;   
constexpr int   ZEBRA_MIN_STRIPES   = 3;     
constexpr float ZEBRA_STRIPE_ASPECT = 1.8f;  
constexpr float ZEBRA_MIN_AREA      = 200.0f; 
constexpr float ZEBRA_GAP_STD_MAX   = 8.0f;  
constexpr int   ZEBRA_CONFIRM       = 2;      
constexpr int   ZEBRA_HOLD_FRAMES   = 40;     
constexpr int   ZEBRA_SPEED         = 5;      
constexpr bool  ENABLE_ZEBRA        = false;  

const vector<string> CLASS_NAMES = {
    "stop_sign", "no_entry", "red_light",
    "yellow_light", "green_light", "person"
};
const vector<Scalar> CLASS_COLORS = {
    Scalar(0,0,220), Scalar(0,60,200), Scalar(0,0,255),
    Scalar(0,200,255), Scalar(0,200,50), Scalar(255,150,0)
};

enum class Scenario { NORMAL=0, STOP_SIGN, NO_ENTRY, RED_LIGHT, YELLOW_LIGHT, GREEN_LIGHT, PERSON_AHEAD };

struct YoloDetection { int class_id; float conf; Rect box; };

// ============================================================
//  TRẠNG THÁI CHIA SẺ
// ============================================================
atomic<int>  shared_steer(0);
atomic<int>  shared_speed(0);
atomic<bool> shared_running(false);
atomic<bool> app_exit(false);

Mat   cap_frame;  mutex cap_mtx;  bool cap_ready = false;
int64_t cap_frame_us = 0;                              
int   uart_fd = -1;
Mat   stream_frame;  mutex stream_mtx;

// YOLO: đầu vào (main -> yolo thread)
Mat   yolo_input_frame;  mutex yolo_in_mtx;  condition_variable yolo_in_cv;  bool yolo_in_ready = false;
// YOLO: đầu ra (yolo thread -> main)
vector<YoloDetection> yolo_last_dets;  Scenario yolo_last_sc = Scenario::NORMAL;  mutex yolo_out_mtx;

// ============================================================
//  ĐO LƯỜNG ĐỊNH LƯỢNG 
// ============================================================
atomic<float> g_fps_capture(0.0f);
atomic<float> g_fps_yolo(0.0f);

atomic<int32_t> g_enc_left(-1), g_enc_right(-1);
atomic<int64_t> g_enc_t_ms(-1);        
atomic<int64_t> g_enc_recv_us(0);      


ofstream g_log;
long     g_log_frame_idx = 0;
chrono::steady_clock::time_point g_program_start;

static inline int64_t now_us() {
    return chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now().time_since_epoch()).count();
}
string default_log_name() {
    time_t rt; time(&rt); char tb[32]; strftime(tb, sizeof(tb), "%Y%m%d_%H%M%S", localtime(&rt));
    return string("log_") + tb + ".csv";
}

// ============================================================
//  MJPEG SERVER
// ============================================================
constexpr int MJPEG_PORT = 8080;
void mjpeg_server_thread() {
    int srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) { cerr << "[MJPEG] socket lỗi\n"; return; }
    int opt = 1; setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_addr.s_addr = INADDR_ANY; addr.sin_port = htons(MJPEG_PORT);
    if (bind(srv, (sockaddr*)&addr, sizeof(addr)) < 0) { cerr << "[MJPEG] bind lỗi cổng " << MJPEG_PORT << "\n"; close(srv); return; }
    listen(srv, 1);
    cout << "[MJPEG] Mo trinh duyet: http://<IP-Pi>:" << MJPEG_PORT << "\n";
    while (!app_exit) {
        int cli = accept(srv, nullptr, nullptr);
        if (cli < 0) continue;
        const char* hdr = "HTTP/1.0 200 OK\r\nCache-Control: no-cache\r\nPragma: no-cache\r\n"
                          "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
        if (write(cli, hdr, strlen(hdr)) < 0) { close(cli); continue; }
        vector<uchar> buf;
        while (!app_exit) {
            Mat f;
            { lock_guard<mutex> lk(stream_mtx); if (!stream_frame.empty()) stream_frame.copyTo(f); }
            if (f.empty()) { this_thread::sleep_for(chrono::milliseconds(30)); continue; }
            imencode(".jpg", f, buf, {IMWRITE_JPEG_QUALITY, 70});
            char part[128];
            int n = snprintf(part, sizeof(part), "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %zu\r\n\r\n", buf.size());
            if (write(cli, part, n) < 0) break;
            if (write(cli, (const char*)buf.data(), buf.size()) < 0) break;
            if (write(cli, "\r\n", 2) < 0) break;
            this_thread::sleep_for(chrono::milliseconds(50));
        }
        close(cli);
    }
    close(srv);
}

// ============================================================
//  UART
// ============================================================
bool init_uart(const char* port) {
    uart_fd = open(port, O_RDWR | O_NOCTTY | O_NDELAY);
    if (uart_fd < 0) return false;
    termios opt; tcgetattr(uart_fd, &opt);
    cfsetispeed(&opt, B115200); cfsetospeed(&opt, B115200);
    opt.c_cflag |= (CLOCAL | CREAD);
    opt.c_cflag &= ~(PARENB | CSTOPB | CSIZE); opt.c_cflag |= CS8;
    opt.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    opt.c_oflag &= ~OPOST;
    opt.c_cc[VMIN] = 0; opt.c_cc[VTIME] = 1;
    tcsetattr(uart_fd, TCSANOW, &opt);
    return true;
}
void uart_thread_func() {
    using clk = chrono::steady_clock; auto next = clk::now();
    while (!app_exit) {
        next += chrono::milliseconds(50);
        this_thread::sleep_until(next);
        if (uart_fd < 0) continue;
        int st = shared_steer.load(memory_order_relaxed);
        int sp = shared_speed.load(memory_order_relaxed);
        bool run = shared_running.load(memory_order_relaxed);
        char buf[32];
        int n = run ? snprintf(buf, sizeof(buf), "F,%d,%d\n", st, sp)
                    : snprintf(buf, sizeof(buf), "S,0,0\n");
        write(uart_fd, buf, n);
    }
}

// Luồng RIÊNG chỉ để đọc telemetry encoder ESP32 gửi về khung "T,<encL>,<encR>,<t_ms>\n"
void uart_rx_thread_func() {
    string line; line.reserve(64);
    while (!app_exit) {
        if (uart_fd < 0) { this_thread::sleep_for(chrono::milliseconds(100)); continue; }
        char c;
        int n = read(uart_fd, &c, 1);
        if (n <= 0) continue;  
        if (c == '\n') {
            if (!line.empty() && line[0] == 'T') {
                long eL = 0, eR = 0; long long tms = 0;
                if (sscanf(line.c_str(), "T,%ld,%ld,%lld", &eL, &eR, &tms) == 3) {
                    g_enc_left.store((int32_t)eL, memory_order_relaxed);
                    g_enc_right.store((int32_t)eR, memory_order_relaxed);
                    g_enc_t_ms.store((int64_t)tms, memory_order_relaxed);
                    g_enc_recv_us.store(now_us(), memory_order_relaxed);
                }
            }
            line.clear();
        } else if (c != '\r') {
            if (line.size() < 63) line.push_back(c);
            else line.clear();   
        }
    }
}

// ============================================================
//  CAMERA
// ============================================================
void capture_thread_func() {
    VideoCapture cap(0, CAP_ANY);
    if (!cap.isOpened()) { cerr << "[CAM] Không mở được camera!\n"; app_exit = true; return; }
    cap.set(CAP_PROP_FRAME_WIDTH, 640); cap.set(CAP_PROP_FRAME_HEIGHT, 480); cap.set(CAP_PROP_FPS, 30);
    Mat tmp;
    int cnt = 0; auto t0 = chrono::steady_clock::now();
    while (!app_exit) {
        if (!cap.read(tmp) || tmp.empty()) { this_thread::sleep_for(chrono::milliseconds(5)); continue; }
        if (tmp.cols != CAM_W || tmp.rows != CAM_H) resize(tmp, tmp, Size(CAM_W, CAM_H), 0, 0, INTER_LINEAR);
        int64_t t_us = now_us();
        { lock_guard<mutex> lk(cap_mtx); tmp.copyTo(cap_frame); cap_frame_us = t_us; cap_ready = true; }
        cnt++;
        auto now = chrono::steady_clock::now();
        chrono::duration<float> dt = now - t0;
        if (dt.count() >= 1.0f) { g_fps_capture.store(cnt / dt.count(), memory_order_relaxed); cnt = 0; t0 = now; }
    }
    cap.release();
}

// ============================================================
//  MẶT NẠ VẠCH 
// ============================================================
Mat compute_mask(const Mat& bgr) {
    Mat gray, blur, white, floor, mask;
    cvtColor(bgr, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blur, Size(5, 5), 0);
    adaptiveThreshold(blur, white, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, ADAPT_BLOCK, ADAPT_C);
    threshold(blur, floor, FLOOR_MAX, 255, THRESH_BINARY_INV);
    morphologyEx(floor, floor, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(FLOOR_DILATE, FLOOR_DILATE)));
    bitwise_and(white, floor, mask);
    morphologyEx(mask, mask, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(3, 3)));
    Mat labels, stats, cents;
    int ncomp = connectedComponentsWithStats(mask, labels, stats, cents, 8, CV_32S);
    Mat clean = Mat::zeros(mask.size(), CV_8UC1);
    for (int i = 1; i < ncomp; i++)
        if (stats.at<int>(i, CC_STAT_AREA) >= MIN_BLOB) clean.setTo(255, labels == i);
    return clean;
}

// ============================================================
//  TIỆN ÍCH + PHÁT HIỆN TÂM LÀN 
// ============================================================
static int hist_peak(const Mat& mask, int x0, int x1, int y0, int y1, int& peak_cnt) {
    x0 = max(0, x0); x1 = min(mask.cols, x1);
    int W = mask.cols; vector<int> hist(W, 0);
    for (int y = y0; y < y1; y++) { const uchar* row = mask.ptr<uchar>(y); for (int x = x0; x < x1; x++) if (row[x]) hist[x]++; }
    int best = (x0 + x1) / 2, bestv = 0;
    for (int x = x0; x < x1; x++) if (hist[x] > bestv) { bestv = hist[x]; best = x; }
    peak_cnt = bestv; return best;
}
static bool window_centroid(const Mat& mask, int xc, int margin, int y0, int y1, float& out_x) {
    int x0 = max(0, xc - margin), x1 = min(mask.cols, xc + margin);
    double sum = 0; int cnt = 0;
    for (int y = y0; y < y1; y++) { const uchar* row = mask.ptr<uchar>(y); for (int x = x0; x < x1; x++) if (row[x]) { sum += x; cnt++; } }
    if (cnt < MIN_PIX) return false;
    out_x = float(sum / cnt); return true;
}

struct LaneResult { bool found = false; float offset = 0.0f; int nbands = 0; float curve = 0.0f; };

LaneResult detect_lane(const Mat& bgr, const Mat& mask, Mat& dbg) {
    LaneResult res;
    int W = mask.cols, H = mask.rows;
    float cx = W / 2.0f;
    int y0 = int(H * ROI_TOP_Y), roi_h = H - y0;
    int band_h = max(1, roi_h / N_BANDS);
    int margin = int(W * SEARCH_MARGIN);
    float half_lane_px = HALF_LANE * W;

    dbg = bgr.clone();
    line(dbg, Point(0, y0), Point(W, y0), Scalar(120, 120, 120), 1);

    int base_y0 = max(y0, H - 2 * band_h);
    int lcnt, rcnt;
    int lpeak = hist_peak(mask, 0, int(cx), base_y0, H, lcnt);
    int rpeak = hist_peak(mask, int(cx), W, base_y0, H, rcnt);
    bool lActive = lcnt >= COL_MIN, rActive = rcnt >= COL_MIN;
    float lx = lpeak, rx = rpeak;

    vector<float> centers, weights, lefts, rights;
    for (int b = N_BANDS - 1; b >= 0; b--) {
        int by0 = y0 + b * band_h;
        int by1 = (b == N_BANDS - 1) ? H : by0 + band_h;
        int bcy = (by0 + by1) / 2;
        float nx; bool lf = false, rf = false;
        if (lActive && window_centroid(mask, int(lx), margin, by0, by1, nx)) { lx = nx; lf = true; }
        if (rActive && window_centroid(mask, int(rx), margin, by0, by1, nx)) { rx = nx; rf = true; }
        float center;
        if      (lf && rf) center = (lx + rx) / 2.0f;
        else if (lf)       center = lx + half_lane_px;
        else if (rf)       center = rx - half_lane_px;
        else               continue;
        float w = 1.0f + 0.4f * (float(N_BANDS - 1 - b) / (N_BANDS - 1));
        centers.push_back(center); weights.push_back(w);
        if (lf) lefts.push_back(lx);
        if (rf) rights.push_back(rx);
        if (lf) circle(dbg, Point(int(lx), bcy), 3, Scalar(0, 255, 0), -1);
        if (rf) circle(dbg, Point(int(rx), bcy), 3, Scalar(0, 150, 255), -1);
        circle(dbg, Point(int(center), bcy), 4, Scalar(0, 255, 255), -1);
    }

    res.nbands = (int)centers.size();
    if (res.nbands < MIN_BANDS) {
        char tag[40]; snprintf(tag, sizeof(tag), "NO LANE bands=%d", res.nbands);
        putText(dbg, tag, Point(5, H - 6), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2);
        return res;
    }
    float sx = 0, sw = 0;
    for (size_t i = 0; i < centers.size(); i++) { sx += centers[i] * weights[i]; sw += weights[i]; }
    float lane_center = sx / sw;
    float curve = (centers.back() - centers.front()) / cx;
    if (fabsf(curve) > CURVE_TH) {
        float inner_x = 0; int n = 0;
        if (curve < 0 && !lefts.empty())  { for (float v : lefts)  inner_x += v; n = lefts.size(); }
        if (curve > 0 && !rights.empty()) { for (float v : rights) inner_x += v; n = rights.size(); }
        if (n > 0) { inner_x /= n; lane_center = lane_center + INNER_BIAS * (inner_x - lane_center); }
    }

    int nL = (int)lefts.size(), nR = (int)rights.size();
    if ((nL >= MIN_BANDS && nR == 0) || (nR >= MIN_BANDS && nL == 0)) {
        const vector<float>& ln = (nL > 0) ? lefts : rights;
        float near_x = ln.front();             
        float far_x  = ln.back();              
        float lean   = far_x - near_x;          
        float line_mean = 0; for (float v : ln) line_mean += v; line_mean /= ln.size();
        float dir, dist;
        if (fabsf(lean) > LEAN_TH) {            
            dir  = (lean < 0) ? +1.0f : -1.0f; 
            dist = HUG_LANE * W;                
        } else {                                
            dir  = (nL > 0) ? +1.0f : -1.0f;
            dist = half_lane_px;
        }
        lane_center = line_mean + dir * dist;
    }

    float offset = (lane_center - cx) / cx;
    offset = max(-1.0f, min(1.0f, offset));
    res.found = true; res.offset = offset; res.curve = curve;
    line(dbg, Point(int(lane_center), y0), Point(int(lane_center), H - 1), Scalar(0, 255, 255), 2);
    line(dbg, Point(int(cx), y0), Point(int(cx), H - 1), Scalar(180, 180, 180), 1);
    char tag[48]; snprintf(tag, sizeof(tag), "off=%+.3f bands=%d", offset, res.nbands);
    putText(dbg, tag, Point(5, H - 6), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 1);
    return res;
}

int offset_to_steer(float offset) {
    if (fabsf(offset) < DEADBAND) return 0;
    int sign = (offset > 0) ? 1 : -1;
    float x = min((fabsf(offset) - DEADBAND) / (1.0f - DEADBAND), 1.0f);
    float mag = 5.0f + STEER_GAIN * powf(x, STEER_POW);
#if STEER_INVERT
    sign = -sign;
#endif
    return (int)roundf(sign * min(mag, 100.0f));
}

// ============================================================
//  PHÁT HIỆN VẠCH NGƯỜI ĐI BỘ 
// ============================================================
bool detect_zebra(const Mat& mask) {
    int W = mask.cols, H = mask.rows;
    int x0 = int(W * ZEBRA_ROI_X0), x1 = int(W * ZEBRA_ROI_X1);
    int y0 = int(H * ZEBRA_ROI_Y0), y1 = int(H * ZEBRA_ROI_Y1);
    int rw = x1 - x0, rh = y1 - y0;
    if (rw <= 0 || rh <= 0) return false;

    Mat roi = mask(Rect(x0, y0, rw, rh)), roi_clean;
    morphologyEx(roi, roi_clean, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(9, 1)));
    morphologyEx(roi_clean, roi_clean, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(1, 3)));

    vector<vector<Point>> contours;
    findContours(roi_clean, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    vector<float> stripe_cy;
    for (const auto& cnt : contours) {
        if (contourArea(cnt) < ZEBRA_MIN_AREA) continue;
        Rect bb = boundingRect(cnt);
        if ((float)bb.width / max(bb.height, 1) < ZEBRA_STRIPE_ASPECT) continue;  
        if (bb.height < 5) continue;
        if (bb.x <= 2 || (bb.x + bb.width) >= rw - 2) continue;                   
        stripe_cy.push_back(bb.y + bb.height / 2.0f);
    }
    if ((int)stripe_cy.size() < ZEBRA_MIN_STRIPES) return false;

    sort(stripe_cy.begin(), stripe_cy.end());
    vector<float> gaps;
    for (size_t i = 1; i < stripe_cy.size(); i++) gaps.push_back(stripe_cy[i] - stripe_cy[i-1]);
    float mg = 0; for (float g : gaps) mg += g; mg /= gaps.size();
    if (mg < 4.0f) return false;
    float var = 0; for (float g : gaps) var += (g - mg) * (g - mg);
    if (sqrtf(var / gaps.size()) > ZEBRA_GAP_STD_MAX) return false;  
    return true;
}

// ============================================================
//  YOLO ONNX INFERENCE
// ============================================================
static dnn::Net g_yolo_net;
static bool     g_yolo_ok = false;

bool init_yolo(const string& model_path) {
    try {
        g_yolo_net = dnn::readNetFromONNX(model_path);
        g_yolo_net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
        g_yolo_net.setPreferableTarget(dnn::DNN_TARGET_CPU);
        g_yolo_ok = true;
        Mat dummy(YOLO_INPUT_H, YOLO_INPUT_W, CV_8UC3, Scalar(128,128,128)), tb;
        dnn::blobFromImage(dummy, tb, 1.0/255.0, Size(YOLO_INPUT_W, YOLO_INPUT_H), Scalar(), true, false, CV_32F);
        g_yolo_net.setInput(tb);
        Mat to = g_yolo_net.forward();
        printf("[YOLO] Output shape: [%d, %d, %d]\n", to.size[0], to.size[1], to.size[2]);
        cout << "[YOLO] Loaded: " << model_path << "\n";
    } catch (const Exception& e) {
        cerr << "[YOLO] Load failed: " << e.what() << "\n  -> CHẠY CHẾ ĐỘ CHỈ BÁM LÀN (không YOLO)\n";
        g_yolo_ok = false;
    }
    return g_yolo_ok;
}

vector<YoloDetection> run_yolo(const Mat& bgr) {
    vector<YoloDetection> results;
    if (!g_yolo_ok || bgr.empty()) return results;
    Mat square_img(YOLO_INPUT_H, YOLO_INPUT_W, CV_8UC3, Scalar(114, 114, 114));
    int pad_y = (YOLO_INPUT_H - bgr.rows) / 2;
    bgr.copyTo(square_img(Rect(0, pad_y, bgr.cols, bgr.rows)));
    Mat blob; dnn::blobFromImage(square_img, blob, 1.0/255.0, Size(YOLO_INPUT_W, YOLO_INPUT_H), Scalar(), true, false, CV_32F);
    g_yolo_net.setInput(blob);
    Mat raw_out = g_yolo_net.forward();
    int dim1 = raw_out.size[1], dim2 = raw_out.size[2];
    Mat out_tensor(dim1, dim2, CV_32F, raw_out.ptr<float>());
    if (dim1 < dim2) out_tensor = out_tensor.t();

    vector<int> class_ids; vector<float> confidences; vector<Rect> boxes;
    for (int i = 0; i < out_tensor.rows; ++i) {
        float* row = out_tensor.ptr<float>(i);
        float max_conf = 0.0f; int best_cls = -1;
        for (int c = 0; c < YOLO_NUM_CLASS; ++c)
            if (row[4 + c] > max_conf) { max_conf = row[4 + c]; best_cls = c; }
        if (max_conf < YOLO_CONF_THR || best_cls < 0) continue;
        float cx = row[0], cy = row[1] - pad_y, w = row[2], h = row[3];
        int x1 = max(0, (int)(cx - w/2)), y1 = max(0, (int)(cy - h/2));
        int bw = min((int)w, bgr.cols - x1), bh = min((int)h, bgr.rows - y1);
        if (bw <= 0 || bh <= 0) continue;
        class_ids.push_back(best_cls); confidences.push_back(max_conf); boxes.push_back(Rect(x1, y1, bw, bh));
    }
    vector<int> keep;
    dnn::NMSBoxes(boxes, confidences, YOLO_CONF_THR, YOLO_NMS_THR, keep);
    for (int idx : keep) results.push_back({class_ids[idx], confidences[idx], boxes[idx]});
    return results;
}

void draw_detections(Mat& dbg, const vector<YoloDetection>& dets) {
    for (const auto& d : dets) {
        if (d.class_id < 0 || d.class_id >= (int)CLASS_NAMES.size()) continue;
        const Scalar& col = CLASS_COLORS[d.class_id];
        rectangle(dbg, d.box, col, 2);
        char label[64]; snprintf(label, sizeof(label), "%s %.2f a=%d", CLASS_NAMES[d.class_id].c_str(), d.conf, d.box.area());
        int base; Size ts = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.45, 1, &base);
        Point tl(d.box.x, max(0, d.box.y - ts.height - 3));
        rectangle(dbg, tl, Point(tl.x + ts.width + 2, tl.y + ts.height + base + 3), col, FILLED);
        putText(dbg, label, Point(tl.x + 1, tl.y + ts.height + 1), FONT_HERSHEY_SIMPLEX, 0.45, Scalar(255,255,255), 1);
    }
}

Scenario determine_scenario(const vector<YoloDetection>& dets) {
    bool no_entry=false, stop=false, red=false, yellow=false, green=false, person=false;
    for (const auto& d : dets) {
        if (d.box.area() < MIN_BOX_AREA) continue;     
        switch (d.class_id) {
            case 0: stop=true; break; case 1: no_entry=true; break; case 2: red=true; break;
            case 3: yellow=true; break; case 4: green=true; break; case 5: person=true; break;
        }
    }
    if (no_entry) return Scenario::NO_ENTRY;
    if (stop)     return Scenario::STOP_SIGN;
    if (red)      return Scenario::RED_LIGHT;
    if (person)   return Scenario::PERSON_AHEAD;
    if (yellow)   return Scenario::YELLOW_LIGHT;
    if (green)    return Scenario::GREEN_LIGHT;
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

// YOLO THREAD: 
void yolo_thread_func() {
    int cnt = 0; auto t0 = chrono::steady_clock::now();
    while (!app_exit) {
        Mat frame;
        {
            unique_lock<mutex> lk(yolo_in_mtx);
            yolo_in_cv.wait_for(lk, chrono::milliseconds(50), []{ return yolo_in_ready || app_exit.load(); });
            if (app_exit) break;
            if (!yolo_in_ready) continue;
            frame = yolo_input_frame.clone();
            yolo_in_ready = false;
        }
        if (frame.empty()) continue;
        auto dets = run_yolo(frame);
        Scenario sc = determine_scenario(dets);
       
        bool is_stop  = (sc == Scenario::RED_LIGHT || sc == Scenario::NO_ENTRY ||
                         sc == Scenario::PERSON_AHEAD || sc == Scenario::STOP_SIGN);
        bool is_green = (sc == Scenario::GREEN_LIGHT);
        static int stop_votes = 0, green_votes = 0; static Scenario last_stop = Scenario::NORMAL;
        if (is_stop)  { stop_votes  = min(stop_votes + 1, 4); last_stop = sc; }
        else          { stop_votes  = max(stop_votes - 1, 0); }
        if (is_green) { green_votes = min(green_votes + 1, 4); }
        else          { green_votes = max(green_votes - 1, 0); }
        Scenario stable_sc;
        if      (stop_votes  >= YOLO_STOP_VOTES) stable_sc = last_stop;            
        else if (green_votes >= YOLO_STOP_VOTES) stable_sc = Scenario::GREEN_LIGHT; 
        else                                     stable_sc = Scenario::NORMAL;      
        { lock_guard<mutex> lk(yolo_out_mtx); yolo_last_dets = dets; yolo_last_sc = stable_sc; }

        cnt++;
        auto now = chrono::steady_clock::now();
        chrono::duration<float> dt = now - t0;
        if (dt.count() >= 1.0f) { g_fps_yolo.store(cnt / dt.count(), memory_order_relaxed); cnt = 0; t0 = now; }

        this_thread::sleep_for(chrono::milliseconds(120));
    }
}

// ============================================================
//  MAIN
// ============================================================
int main(int argc, char** argv) {
    if (argc > 1) YOLO_MODEL_PATH = argv[1];
    string log_path = (argc > 2) ? argv[2] : default_log_name();
    cout << "=============================================\n"
         << "  LANE FOLLOW + YOLO                          \n"
         << "=============================================\n"
         << "  a=chạy  s=dừng  c=chụp  q=thoát            \n"
         << "=============================================\n";

    signal(SIGPIPE, SIG_IGN);

    const char* ports[] = {"/dev/serial0", "/dev/ttyAMA0", "/dev/ttyS0"};
    bool uart_ok = false;
    for (auto p : ports) if (init_uart(p)) { cout << "[UART] " << p << " @115200\n"; uart_ok = true; break; }
    if (!uart_ok) { cerr << "[UART] Lỗi cổng!\n"; return 1; }

    setNumThreads(2);             
    init_yolo(YOLO_MODEL_PATH);  

    g_program_start = chrono::steady_clock::now();
    g_log.open(log_path, ios::out);
    if (g_log.is_open()) {
        g_log << "frame,t_ms,fps_vision,fps_capture,fps_yolo,latency_compute_ms,"
                 "lane_found,off,nbands,steer,spd,no_lane,scenario,force_stop,on_zebra,"
                 "enc_left,enc_right,enc_t_ms\n";
        g_log.flush();
        cout << "[LOG] Ghi số liệu vào: " << log_path << "\n";
    } else {
        cerr << "[LOG] Không mở được file log (" << log_path << ") — vẫn chạy bình thường, chỉ không lưu số liệu.\n";
    }

    termios oldt; tcgetattr(STDIN_FILENO, &oldt);
    { termios n = oldt; n.c_lflag &= ~(ICANON | ECHO); n.c_cc[VMIN] = 0; n.c_cc[VTIME] = 0;
      tcsetattr(STDIN_FILENO, TCSANOW, &n); }

    thread t_cap(capture_thread_func);
    thread t_uart(uart_thread_func);
    thread t_uart_rx(uart_rx_thread_func);
    thread t_mjpeg(mjpeg_server_thread);
    thread t_yolo(yolo_thread_func);

    float smooth_steer = 0.0f;
    int   no_lane = 0, fps_cnt = 0;
    float fps = 0.0f;
    auto  fps_t0 = chrono::steady_clock::now();

    bool  red_latched = false;
    bool  stopped_at_line = false;   
    auto  red_release_at = chrono::steady_clock::now();

    while (!app_exit) {
        char ch = 0; read(STDIN_FILENO, &ch, 1);
        if      (ch == 'a') { shared_running = true;  smooth_steer = 0; red_latched = false; printf("\n>>> CHẠY\n"); }
        else if (ch == 's') { shared_running = false; shared_steer = 0; shared_speed = 0; smooth_steer = 0; printf("\n>>> DỪNG\n"); }
        else if (ch == 'q') { app_exit = true; break; }

        Mat frame; int64_t frame_cap_us = 0;
        { lock_guard<mutex> lk(cap_mtx); if (cap_ready) { cap_frame.copyTo(frame); frame_cap_us = cap_frame_us; } }
        if (frame.empty()) { this_thread::sleep_for(chrono::milliseconds(5)); continue; }

        { lock_guard<mutex> lk(yolo_in_mtx); frame.copyTo(yolo_input_frame); yolo_in_ready = true; }
        yolo_in_cv.notify_one();

        Mat mask = compute_mask(frame);
        Mat dbg;
        LaneResult lane = detect_lane(frame, mask, dbg);

        static int last_good_steer = 0;
        int raw_steer;
        if (lane.found) { raw_steer = offset_to_steer(lane.offset); last_good_steer = raw_steer; }
        else            { raw_steer = last_good_steer; }
        smooth_steer = SMOOTH_ALPHA * raw_steer + (1.0f - SMOOTH_ALPHA) * smooth_steer;
        int steer_val = (int)roundf(smooth_steer);

        if (lane.found) no_lane = 0; else no_lane++;
        int spd;
        if (!lane.found)                                            spd = SLOW_SPEED;
        else if (ENABLE_CURVE_SPEED && fabsf(lane.curve) > CURVE_TH) spd = CURVE_SPEED;  
        else                                                        spd = DRIVE_SPEED;
        if (no_lane > NO_LANE_STOP) { spd = 0; last_good_steer = 0; }

        bool zebra_raw = detect_zebra(mask);
        static int zebra_confirm = 0, zebra_hold = 0;
        if (zebra_raw) zebra_confirm++; else zebra_confirm = 0;
        if (zebra_confirm >= ZEBRA_CONFIRM) zebra_hold = ZEBRA_HOLD_FRAMES;
        bool on_zebra = ENABLE_ZEBRA && zebra_hold > 0;
        if (zebra_hold > 0) zebra_hold--;
        if (on_zebra) {
            steer_val = 0;         
            last_good_steer = 0;
            spd = ZEBRA_SPEED;      
        }

        Scenario sc; vector<YoloDetection> dets;
        { lock_guard<mutex> lk(yolo_out_mtx); sc = yolo_last_sc; dets = yolo_last_dets; }

        auto now2 = chrono::steady_clock::now();
        bool force_stop = false;
        switch (sc) {
            case Scenario::NO_ENTRY:
            case Scenario::PERSON_AHEAD:
            case Scenario::STOP_SIGN:
                force_stop = true; break;                
            case Scenario::RED_LIGHT:
                if (!red_latched) {                       
                    red_latched = true;
                    red_release_at = now2 + chrono::milliseconds(RED_LATCH_TIMEOUT_MS);
                }
                break;
            default: break;
        }
        
        if (red_latched) {
            force_stop = true;                         
            if (sc == Scenario::GREEN_LIGHT || now2 >= red_release_at) {
                red_latched = false; stopped_at_line = false;
            }
        }
        if (force_stop) spd = 0;

        shared_steer.store(steer_val, memory_order_relaxed);
        shared_speed.store(shared_running.load() ? spd : 0, memory_order_relaxed);

        float latency_compute_ms = (frame_cap_us > 0) ? (now_us() - frame_cap_us) / 1000.0f : 0.0f;

        draw_detections(dbg, dets);
        char sctag[48]; snprintf(sctag, sizeof(sctag), "%s%s", scenario_name(sc), force_stop ? " [STOP]" : "");
        putText(dbg, sctag, Point(5, 18), FONT_HERSHEY_SIMPLEX, 0.55,
                force_stop ? Scalar(0,0,255) : Scalar(0,200,0), 2);
        if (on_zebra) putText(dbg, "ZEBRA -> di thang", Point(5, 38),
                              FONT_HERSHEY_SIMPLEX, 0.55, Scalar(0,140,255), 2);
        if (zebra_raw)
            putText(dbg, "ZEBRA", Point(220, 18), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(0,140,255), 2);

        if (ch == 'c' && !frame.empty()) {
            time_t rt; time(&rt); char tb[20]; strftime(tb, sizeof(tb), "%H%M%S", localtime(&rt));
            string ts(tb);
            imwrite("ly_" + ts + "_raw.jpg", frame);
            imwrite("ly_" + ts + "_dbg.jpg", dbg);
            printf("\n>>> [SNAP] ly_%s_*.jpg\n", ts.c_str());
        }

        if (!dbg.empty()) {
            Mat combined;
            if (!mask.empty()) { Mat m3; cvtColor(mask, m3, COLOR_GRAY2BGR); hconcat(dbg, m3, combined); }
            else combined = dbg;
            lock_guard<mutex> lk(stream_mtx); combined.copyTo(stream_frame);
        }

#if SHOW_WINDOW
        if (!dbg.empty())  imshow("LANE", dbg);
        if (waitKey(1) == 'q') { app_exit = true; break; }
#endif

        fps_cnt++;
        auto now = chrono::steady_clock::now();
        chrono::duration<float> dt = now - fps_t0;
        if (dt.count() >= 1.0f) { fps = fps_cnt / dt.count(); fps_cnt = 0; fps_t0 = now; }
        printf("\r[%s] %s fps=%4.1f off=%+.3f str=%+4d spd=%2d noln=%3d sc=%s zb=%d   ",
               shared_running.load() ? "RUN " : "STOP",
               lane.found ? "LANE" : "NOLN", fps,
               lane.found ? lane.offset : 0.0f, steer_val, spd, no_lane, scenario_name(sc), on_zebra ? 1 : 0);
        fflush(stdout);

        if (g_log.is_open()) {
            g_log_frame_idx++;
            double t_ms = chrono::duration<double, milli>(chrono::steady_clock::now() - g_program_start).count();
            string sc_name = scenario_name(sc);
            while (!sc_name.empty() && sc_name.back() == ' ') sc_name.pop_back();  
            g_log << g_log_frame_idx << ',' << fixed << setprecision(1) << t_ms << ','
                  << setprecision(2) << fps << ',' << g_fps_capture.load(memory_order_relaxed) << ','
                  << g_fps_yolo.load(memory_order_relaxed) << ',' << setprecision(2) << latency_compute_ms << ','
                  << (lane.found ? 1 : 0) << ',' << setprecision(4) << (lane.found ? lane.offset : 0.0f) << ','
                  << lane.nbands << ',' << steer_val << ',' << spd << ',' << no_lane << ','
                  << sc_name << ',' << (force_stop ? 1 : 0) << ',' << (on_zebra ? 1 : 0) << ','
                  << g_enc_left.load(memory_order_relaxed) << ',' << g_enc_right.load(memory_order_relaxed) << ','
                  << g_enc_t_ms.load(memory_order_relaxed) << "\n";
            if (g_log_frame_idx % 50 == 0) g_log.flush();   
        }
    }

    app_exit = true;
    yolo_in_cv.notify_all();
    t_mjpeg.detach();
    if (t_cap.joinable())     t_cap.join();
    if (t_uart.joinable())    t_uart.join();
    if (t_uart_rx.joinable()) t_uart_rx.join();
    if (t_yolo.joinable())    t_yolo.join();
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    if (uart_fd >= 0) { write(uart_fd, "S,0,0\n", 6); close(uart_fd); }
    if (g_log.is_open()) { g_log.flush(); g_log.close(); cout << "\n[LOG] Đã ghi " << g_log_frame_idx << " dòng.\n"; }
    printf("\nĐã dừng an toàn.\n");
    return 0;
}