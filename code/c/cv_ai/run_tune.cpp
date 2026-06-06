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
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// =============================================
// THÔNG SỐ CẤU HÌNH CAMERA & ÁNH SÁNG
// =============================================
constexpr int   CAM_W              = 320;
constexpr int   CAM_H              = 240;
constexpr int   CAM_FIXED_EXPOSURE = 30000;
constexpr float CAM_ANALOGUE_GAIN  = 4.0f;
constexpr float SW_BRIGHTNESS      = 1.2f;  // Tăng sáng bằng phần mềm (1.2 - 1.5)

// HSV Setup - Hạ V_MIN để bắt vạch trong môi trường tối
constexpr int HSV_H_MIN = 0,   HSV_H_MAX = 180;
constexpr int HSV_S_MIN = 0,   HSV_S_MAX = 85;
constexpr int HSV_V_MIN = 80,  HSV_V_MAX = 255;

// Thông số xe
constexpr int   DRIVE_SPEED      = 12;
constexpr int   SLOW_SPEED       = 6;
constexpr int   ZEBRA_SPEED      = 8;
constexpr int   ZEBRA_HOLD_FRAMES = 60;
constexpr float SMOOTH_ALPHA     = 0.75f;
constexpr float DEADBAND         = 0.04f;
constexpr float MAX_OFFSET_JUMP  = 0.65f;

// ROI - Mở rộng góc nhìn bên phải (0.80 -> 0.85) để bắt góc cua gắt tốt hơn
constexpr float ROI_TOP_LEFT_X  = 0.28f;
constexpr float ROI_TOP_RIGHT_X = 0.85f; 
constexpr float ROI_TOP_Y       = 0.42f;

constexpr float LANE_SLOPE_MIN  = 0.47f;
constexpr float LANE_SLOPE_MAX  = 5.67f;

// CV Params - Điều chỉnh Hough để bắt các đoạn đứt khúc ở góc cua
constexpr int CANNY_LOW  = 40,  CANNY_HIGH     = 100;
constexpr int HOUGH_THR  = 30,  HOUGH_MIN_LEN  = 20, HOUGH_MAX_GAP = 50;

bool is_zebra = false;

// =============================================
// SHARED STATE
// =============================================
atomic<int>  shared_steer(0);
atomic<int>  shared_speed(0);
atomic<bool> shared_running(false);
atomic<bool> app_exit(false);

Mat        cap_frame;           
mutex      cap_mtx;
condition_variable cap_cv;
bool       cap_ready = false;

struct VisionResult {
    bool  lane_found  = false;
    float lane_offset = 0.0f;
    bool  is_zebra    = false;
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

int uart_fd = -1;

// =============================================
// UART THREAD
// =============================================
bool init_uart(const char* port) {
    uart_fd = open(port, O_RDWR | O_NOCTTY | O_NDELAY);
    if (uart_fd < 0) return false;
    struct termios opt;
    tcgetattr(uart_fd, &opt);
    cfsetispeed(&opt, B115200);
    cfsetospeed(&opt, B115200);
    opt.c_cflag  |= (CLOCAL | CREAD);
    opt.c_cflag  &= ~(PARENB | CSTOPB | CSIZE); opt.c_cflag |= CS8;
    opt.c_lflag  &= ~(ICANON | ECHO | ECHOE | ISIG);
    opt.c_oflag  &= ~OPOST;
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

// =============================================
// CAMERA CAPTURE THREAD
// =============================================
void capture_thread_func() {
    VideoCapture cap(0, CAP_V4L2);
    if (!cap.isOpened()) {
        cerr << "[CAM] Không mở được camera!\n";
        app_exit = true; return;
    }
    cap.set(CAP_PROP_FRAME_WIDTH,  CAM_W);
    cap.set(CAP_PROP_FRAME_HEIGHT, CAM_H);
    cap.set(CAP_PROP_FPS, 30);
    cap.set(CAP_PROP_BUFFERSIZE, 1);  

    if (CAM_FIXED_EXPOSURE > 0) {
        cap.set(CAP_PROP_AUTO_EXPOSURE, 1); // 1 = Manual Mode
        cap.set(CAP_PROP_EXPOSURE, CAM_FIXED_EXPOSURE / 100.0);
        cap.set(CAP_PROP_GAIN, CAM_ANALOGUE_GAIN);
        system("v4l2-ctl -c auto_exposure=1,exposure_time_absolute=30000,analogue_gain=4000 > /dev/null 2>&1");
    }

    Mat tmp;
    while (!app_exit) {
        if (!cap.read(tmp) || tmp.empty()) continue;
        
        if (tmp.cols != CAM_W || tmp.rows != CAM_H) {
            resize(tmp, tmp, Size(CAM_W, CAM_H), 0, 0, INTER_LINEAR);
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

// =============================================
// VISION PROCESSING LOGIC
// =============================================
static Ptr<CLAHE> g_clahe;
void init_clahe() {
    g_clahe = createCLAHE(4.0, Size(4, 4));
}

Mat compute_white_mask(const Mat& bgr) {
    Mat hsv, blur_img, mask;
    cvtColor(bgr, hsv, COLOR_BGR2HSV);
    vector<Mat> ch;
    split(hsv, ch);
    g_clahe->apply(ch[2], ch[2]);
    merge(ch, hsv);
    GaussianBlur(hsv, blur_img, Size(7, 7), 0);
    inRange(blur_img,
            Scalar(HSV_H_MIN, HSV_S_MIN, HSV_V_MIN),
            Scalar(HSV_H_MAX, HSV_S_MAX, HSV_V_MAX),
            mask);
    return mask;
}

bool detect_zebra_from_mask(const Mat& white_mask, int W, int H) {
    int y0 = int(H * 0.45), y1 = int(H * 0.90);
    int x0 = int(W * 0.20), x1 = int(W * 0.80);
    Mat roi = white_mask(Rect(x0, y0, x1 - x0, y1 - y0));
    Mat roi_blur;
    GaussianBlur(roi, roi_blur, Size(5, 5), 0);
    Mat roi_thresh;
    threshold(roi_blur, roi_thresh, 127, 255, THRESH_BINARY);
    vector<vector<Point>> contours;
    findContours(roi_thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    int stripe_count = 0;
    for (const auto& cnt : contours) {
        double area = contourArea(cnt);
        Rect bb = boundingRect(cnt);
        if (area > 100 && bb.height > 15 && bb.width > 10)
            stripe_count++;
    }
    return stripe_count >= 3;
}

struct LaneResult { bool found; float offset; };

LaneResult detect_lane_from_mask(const Mat& bgr, const Mat& white_mask, Mat& dbg, int W, int H) {
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
        float dx = l[2] - l[0];
        float dy = l[3] - l[1];
        float len = hypotf(dx, dy);
        if (len < 1.0f || fabsf(dx) < 0.5f) continue;
        float sa = fabsf(dy / dx);
        if (sa < LANE_SLOPE_MIN || sa > LANE_SLOPE_MAX) continue;
        float xn = l[0] + (ly_near - l[1]) * dx / dy;
        float xf = l[0] + (ly_far  - l[1]) * dx / dy;
        float x_look = 0.6f * xn + 0.4f * xf;
        if (is_zebra) {
            if (fabsf(x_look - cx) < (ew_val * 0.35f)) {
                continue; 
            }
        }
        if (x_look < cx) { left_xs.push_back({x_look, len}); left_lines.push_back(l); } 
        else { right_xs.push_back({x_look, len}); right_lines.push_back(l); }
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
    float lane_center;
    bool  single = false;

    if (hl && hr) {
        float cw = xr - xl;
        {
            lock_guard<mutex> lk(algo_mtx); 
            if (cw > 80 && cw < 280) _ema_lane_width = 0.1f * cw + 0.9f * _ema_lane_width;
        }
        lane_center = (xl + xr) / 2.0f;
    } else {
        float ew;
        { lock_guard<mutex> lk(algo_mtx); ew = _ema_lane_width; }
        lane_center = hl ? xl + ew / 2.0f : xr - ew / 2.0f;
        single = true;
    }

    float offset = (lane_center - cx) / cx;
    offset = max(-1.0f, min(1.0f, offset));
    if (single) offset = max(-1.0f, min(1.0f, offset * 1.3f));

    res.found  = true;
    res.offset = offset;

    for (const auto& l : left_lines) line(dbg, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 255, 0), 2);
    for (const auto& l : right_lines) line(dbg, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 100, 255), 2);
    if (hl) circle(dbg, Point(int(xl), lookahead_y), 4, Scalar(255, 255, 255), -1);
    if (hr) circle(dbg, Point(int(xr), lookahead_y), 4, Scalar(255, 255, 255), -1);
    
    line(dbg, Point(int(lane_center), H - 10), Point(int(lane_center), int(H * ROI_TOP_Y)), Scalar(0, 255, 255), 2);
    line(dbg, Point(int(cx), H - 10), Point(int(cx), int(H * ROI_TOP_Y)), Scalar(128, 128, 128), 1);
    
    char tag[50]; int z_cnt;
    { lock_guard<mutex> lk(algo_mtx); ew_val = _ema_lane_width; z_cnt = _zebra_hold_cnt; }
    snprintf(tag, sizeof(tag), "off=%+5.3f W=%d", offset, int(ew_val));
    putText(dbg, tag, Point(5, H - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 1);
    
    if (z_cnt > 0) {
        char z_tag[30]; snprintf(z_tag, sizeof(z_tag), "ZEBRA hold=%d", z_cnt);
        putText(dbg, z_tag, Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(0, 0, 255), 2);
    }
    return res;
}

void vision_thread_func() {
    Mat frame;
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

        if (SW_BRIGHTNESS > 1.0f) {
            frame.convertTo(frame, -1, SW_BRIGHTNESS, 0);
        }

        Mat wmask = compute_white_mask(frame);
        Mat dbg;
        bool zebra = detect_zebra_from_mask(wmask, frame.cols, frame.rows);            
        LaneResult lane = detect_lane_from_mask(frame, wmask, dbg, frame.cols, frame.rows);

        {
            lock_guard<mutex> lk(vis_mtx);
            vis_result.lane_found  = lane.found;
            vis_result.lane_offset = lane.offset;
            vis_result.is_zebra    = zebra;
            frame.copyTo(vis_result.raw_frame);
            dbg.copyTo(vis_result.debug_frame);
            wmask.copyTo(vis_result.mask_frame);
            vis_ready = true;
        }
        vis_cv.notify_one();
    }
}

int offset_to_steer(float offset) {
    if (fabsf(offset) < DEADBAND) return 0;
    int   sign = (offset > 0) ? 1 : -1;
    float x    = min((fabsf(offset) - DEADBAND) / (1.0f - DEADBAND), 1.0f);
    float mag  = 5.0f + 120.0f * powf(x, 1.2f);  
    return (int)roundf(sign * min(mag, 100.0f)); 
}

// =============================================
// MAIN THREAD
// =============================================
int main() {
    cout << "=========================================\n"
         << "  C++ LANE DRIVER — MOBAXTERM STREAM     \n"
         << "=========================================\n"
         << "  a = Bắt đầu  |  s = Dừng\n"
         << "  c = Chụp ảnh |  q = Thoát hệ thống\n"
         << "=========================================\n";

    init_clahe();

    const char* ports[] = {"/dev/serial0", "/dev/ttyAMA0", "/dev/ttyS0"};
    bool uart_ok = false;
    for (auto p : ports) {
        if (init_uart(p)) { cout << "[UART] " << p << " @ 115200\n"; uart_ok = true; break; }
    }
    if (!uart_ok) { cerr << "[UART] Lỗi cổng!\n"; return 1; }

    struct termios oldt;
    tcgetattr(STDIN_FILENO, &oldt);
    {
        struct termios newt = oldt;
        newt.c_lflag     &= ~(ICANON | ECHO);
        newt.c_cc[VMIN]   = 0;
        newt.c_cc[VTIME]  = 0;
        tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    }

    thread t_capture(capture_thread_func);
    thread t_vision(vision_thread_func);
    thread t_uart(uart_thread_func);

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
            shared_running.store(true); smooth_steer = 0.0f; printf("\n>>> TỰ LÁI BẮT ĐẦU\n");
        } else if (ch == 's') {
            shared_running.store(false); shared_steer.store(0); shared_speed.store(0);
            smooth_steer = 0.0f; printf("\n>>> DỪNG LÁI\n");
        } else if (ch == 'q') {
            app_exit = true; break;
        } else if (ch == 'c') {
            time_t rawtime; struct tm * timeinfo; char time_buf[20];
            time(&rawtime); timeinfo = localtime(&rawtime);
            strftime(time_buf, sizeof(time_buf), "%H%M%S", timeinfo);
            string timestamp(time_buf);

            VisionResult vr;
            { lock_guard<mutex> lk(vis_mtx); vr = vis_result; }
            if (!vr.raw_frame.empty()) {
                imwrite("snap_" + timestamp + "_raw.jpg", vr.raw_frame);
                imwrite("snap_" + timestamp + "_debug.jpg", vr.debug_frame);
                imwrite("snap_" + timestamp + "_mask.jpg", vr.mask_frame);
                printf("\n>>> [SCREENSHOT] Đã lưu thành công.\n");
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

        float offset = vr.lane_offset;
        bool  found  = vr.lane_found;
        if (found) {
            lock_guard<mutex> lk(algo_mtx);
            if (!in_zebra && fabsf(offset - _prev_valid_offset) > MAX_OFFSET_JUMP) {
                found = false; 
            } else {
                _prev_valid_offset = offset;
            }
        }                
        int raw_steer = offset_to_steer(found ? offset : 0.0f);
        smooth_steer  = SMOOTH_ALPHA * raw_steer + (1.0f - SMOOTH_ALPHA) * smooth_steer;
        int steer_val = (int)roundf(smooth_steer);

        if (found) no_lane_cnt = 0; else no_lane_cnt++;

        int spd = DRIVE_SPEED;
        if (!found) spd = SLOW_SPEED;
        
        if (no_lane_cnt > 30) { 
            spd = 0; 
        }
        if (in_zebra) spd = ZEBRA_SPEED;

        shared_steer.store(steer_val, memory_order_relaxed);
        shared_speed.store(shared_running.load() ? spd : 0, memory_order_relaxed);

        fps_cnt++;
        auto now = chrono::steady_clock::now();
        chrono::duration<float> dt = now - fps_t0;
        if (dt.count() >= 1.0f) { fps = fps_cnt / dt.count(); fps_cnt = 0; fps_t0 = now; }

        printf("\r[%s] %-5s fps=%5.1f off=%+5.3f str=%+4d spd=%2d noln=%3d zbr=%2d",
               shared_running.load() ? "RUN " : "STOP",
               in_zebra ? "ZEBRA" : (found ? "LANE " : "NOLN "),
               fps, found ? offset : 0.0f, steer_val, spd, no_lane_cnt, zebra_hold);
        fflush(stdout);
    }

    app_exit = true; cap_cv.notify_all(); vis_cv.notify_all();
    if (t_capture.joinable()) t_capture.join();
    if (t_vision.joinable())  t_vision.join();
    if (t_uart.joinable())    t_uart.join();

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    if (uart_fd >= 0) { write(uart_fd, "S,0,0\n", 6); close(uart_fd); }
    printf("\nĐã đóng luồng an toàn.\n");
    return 0;
}