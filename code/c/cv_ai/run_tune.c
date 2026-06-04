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
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// ==========================================
// THÔNG SỐ CẤU HÌNH (Dựa trên bản Python)
// ==========================================
const int CAM_W = 320;
const int CAM_H = 240;

const int DRIVE_SPEED = 12;
const int SLOW_SPEED = 6;
const int ZEBRA_SPEED = 5;
const float SMOOTH_ALPHA = 0.75f;
const float DEADBAND = 0.04f;

const float ROI_BOT_LEFT_X = 0.00f;
const float ROI_BOT_RIGHT_X = 1.00f;
const float ROI_BOT_Y = 1.00f;
const float ROI_TOP_LEFT_X = 0.28f;
const float ROI_TOP_RIGHT_X = 0.80f;
const float ROI_TOP_Y = 0.42f;

const float LANE_SLOPE_MIN = 0.47f;
const float LANE_SLOPE_MAX = 5.67f;
const float MAX_OFFSET_JUMP = 0.65f;

const int HSV_H_MIN = 0,   HSV_H_MAX = 180;
const int HSV_S_MIN = 0,   HSV_S_MAX = 85;
const int HSV_V_MIN = 140, HSV_V_MAX = 255;

const int CANNY_LOW = 40, CANNY_HIGH = 100;
const int HOUGH_THRESHOLD = 30, HOUGH_MIN_LENGTH = 25, HOUGH_MAX_GAP = 40;

// ==========================================
// BIẾN ĐA LUỒNG (SHARED STATE)
// ==========================================
atomic<int> shared_steer(0);
atomic<int> shared_speed(0);
atomic<bool> shared_running(false);
atomic<bool> app_exit(false);

Mat shared_frame;
mutex frame_mutex;
condition_variable frame_cv;
bool new_frame_ready = false;

// Trạng thái thuật toán
float _ema_lane_width = 160.0f;
float _prev_valid_offset = 0.0f;
int _zebra_hold_cnt = 0;

// ==========================================
// CẤU HÌNH UART (LINUX POSIX)
// ==========================================
int uart_fd = -1;

void init_uart(const char* port) {
    uart_fd = open(port, O_RDWR | O_NOCTTY | O_NDELAY);
    if (uart_fd == -1) {
        cerr << "[LỖI] Không thể mở UART: " << port << endl;
        exit(1);
    }
    struct termios options;
    tcgetattr(uart_fd, &options);
    cfsetispeed(&options, B115200);
    cfsetospeed(&options, B115200);
    options.c_cflag |= (CLOCAL | CREAD);
    options.c_cflag &= ~PARENB;
    options.c_cflag &= ~CSTOPB;
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;
    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    options.c_oflag &= ~OPOST;
    tcsetattr(uart_fd, TCSANOW, &options);
}

// ==========================================
// THREAD: UART SENDER (10Hz)
// ==========================================
void uart_thread_func() {
    auto interval = chrono::milliseconds(100);
    while (!app_exit) {
        auto start_time = chrono::steady_clock::now();
        
        int st = shared_steer.load();
        int sp = shared_speed.load();
        bool run = shared_running.load();

        if (uart_fd != -1) {
            char buffer[32];
            if (run) sprintf(buffer, "F,%d,%d\n", st, sp);
            else     sprintf(buffer, "S,0,0\n");
            write(uart_fd, buffer, strlen(buffer));
        }

        auto end_time = chrono::steady_clock::now();
        auto elapsed = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
        if (elapsed < interval) {
            this_thread::sleep_for(interval - elapsed);
        }
    }
}

// ==========================================
// THREAD: CAMERA CAPTURE
// ==========================================
void capture_thread_func() {
    VideoCapture cap(0, CAP_V4L2); // Sử dụng V4L2 backend
    if (!cap.isOpened()) {
        cerr << "[LỖI] Không mở được Camera!" << endl;
        app_exit = true;
        return;
    }
    cap.set(CAP_PROP_FRAME_WIDTH, CAM_W);
    cap.set(CAP_PROP_FRAME_HEIGHT, CAM_H);
    cap.set(CAP_PROP_FPS, 60);

    Mat temp_frame;
    while (!app_exit) {
        cap >> temp_frame;
        if (temp_frame.empty()) continue;

        {
            lock_guard<mutex> lock(frame_mutex);
            temp_frame.copyTo(shared_frame);
            new_frame_ready = true;
        }
        frame_cv.notify_one(); // Đánh thức luồng xử lý ảnh
    }
    cap.release();
}

// ==========================================
// THUẬT TOÁN: NHẬN DIỆN ZEBRA (CONTOUR)
// ==========================================
bool detect_zebra_crossing(const Mat& frame) {
    int H = frame.rows;
    int W = frame.cols;

    int y0 = int(H * 0.45), y1_crop = int(H * 0.90);
    int x0 = int(W * 0.20), x1_crop = int(W * 0.80);
    Rect roi_rect(x0, y0, x1_crop - x0, y1_crop - y0);
    Mat roi = frame(roi_rect);

    Mat hsv, blur_img, mask;
    cvtColor(roi, hsv, COLOR_BGR2HSV);
    
    // CLAHE
    vector<Mat> channels;
    split(hsv, channels);
    Ptr<CLAHE> clahe = createCLAHE(4.0, Size(4, 4));
    clahe->apply(channels[2], channels[2]);
    merge(channels, hsv);

    GaussianBlur(hsv, blur_img, Size(5, 5), 0);
    inRange(blur_img, Scalar(HSV_H_MIN, HSV_S_MIN, HSV_V_MIN), Scalar(HSV_H_MAX, HSV_S_MAX, HSV_V_MAX), mask);

    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    int stripe_count = 0;
    for (const auto& cnt : contours) {
        double area = contourArea(cnt);
        Rect bBox = boundingRect(cnt);
        if (area > 100 && bBox.height > 15 && bBox.width > 10) {
            stripe_count++;
        }
    }

    return stripe_count >= 3;
}

// ==========================================
// THUẬT TOÁN: NHẬN DIỆN LANE & OFFSET
// ==========================================
struct DetectResult {
    bool found;
    float offset;
};

DetectResult detect_lane_offset(const Mat& frame, Mat& dbg) {
    DetectResult res = {false, 0.0f};
    int H = frame.rows;
    int W = frame.cols;

    Mat hsv, blur_img, white_mask, edges;
    cvtColor(frame, hsv, COLOR_BGR2HSV);
    
    vector<Mat> channels;
    split(hsv, channels);
    Ptr<CLAHE> clahe = createCLAHE(4.0, Size(4, 4));
    clahe->apply(channels[2], channels[2]);
    merge(channels, hsv);

    GaussianBlur(hsv, blur_img, Size(7, 7), 0);
    inRange(blur_img, Scalar(HSV_H_MIN, HSV_S_MIN, HSV_V_MIN), Scalar(HSV_H_MAX, HSV_S_MAX, HSV_V_MAX), white_mask);
    Canny(white_mask, edges, CANNY_LOW, CANNY_HIGH);

    Point tl(int(W * ROI_TOP_LEFT_X), int(H * ROI_TOP_Y));
    Point tr(int(W * ROI_TOP_RIGHT_X), int(H * ROI_TOP_Y));
    Point bl(int(W * ROI_BOT_LEFT_X), int(H * ROI_BOT_Y));
    Point br(int(W * ROI_BOT_RIGHT_X), int(H * ROI_BOT_Y));
    
    Mat roi_mask = Mat::zeros(edges.size(), CV_8UC1);
    Point pts[4] = {bl, tl, tr, br};
    const Point* ppt[1] = {pts};
    int npt[] = {4};
    fillPoly(roi_mask, ppt, npt, 1, Scalar(255));

    Mat masked_edges;
    bitwise_and(edges, roi_mask, masked_edges);

    vector<Vec4i> lines;
    HoughLinesP(masked_edges, lines, 1, CV_PI / 180.0, HOUGH_THRESHOLD, HOUGH_MIN_LENGTH, HOUGH_MAX_GAP);

    int lookahead_y = int(H * 0.75);
    float cx = W / 2.0f;

    vector<pair<float, float>> left_xs, right_xs; // {x_look, length}

    for (size_t i = 0; i < lines.size(); i++) {
        Vec4i l = lines[i];
        float dx = l[2] - l[0];
        float dy = l[3] - l[1];
        float length = hypot(dx, dy);
        
        if (length < 1.0f) continue;
        if (abs(dx) < 0.5f) continue;
        
        float slope_abs = abs(dy / dx);
        if (slope_abs < LANE_SLOPE_MIN || slope_abs > LANE_SLOPE_MAX) continue;

        float x_look = l[0] + (lookahead_y - l[1]) * dx / dy;

        if (x_look < cx) left_xs.push_back({x_look, length});
        else             right_xs.push_back({x_look, length});
    }

    auto weighted_mean = [](const vector<pair<float, float>>& pts_w) {
        float sum_x = 0, sum_w = 0;
        for (const auto& p : pts_w) {
            sum_x += p.first * p.second;
            sum_w += p.second;
        }
        return sum_w == 0 ? 0.0f : (sum_x / sum_w);
    };

    bool has_left = !left_xs.empty();
    bool has_right = !right_xs.empty();

    if (!has_left && !has_right) return res;

    float xl = has_left ? weighted_mean(left_xs) : 0.0f;
    float xr = has_right ? weighted_mean(right_xs) : 0.0f;
    float lane_center = 0.0f;
    bool single_lane = false;

    if (has_left && has_right) {
        float current_w = xr - xl;
        if (current_w > 80 && current_w < 280) {
            _ema_lane_width = 0.1f * current_w + 0.9f * _ema_lane_width;
        }
        lane_center = (xl + xr) / 2.0f;
    } else if (has_left) {
        lane_center = xl + (_ema_lane_width / 2.0f);
        single_lane = true;
    } else if (has_right) {
        lane_center = xr - (_ema_lane_width / 2.0f);
        single_lane = true;
    }

    float offset = (lane_center - cx) / cx;
    if (offset < -1.0f) offset = -1.0f; if (offset > 1.0f) offset = 1.0f;

    if (single_lane) {
        offset *= 1.3f;
        if (offset < -1.0f) offset = -1.0f; if (offset > 1.0f) offset = 1.0f;
    }

    res.found = true;
    res.offset = offset;
    return res;
}

// Chuyển Offset -> Steer PWM
int offset_to_steer(float offset) {
    if (abs(offset) < DEADBAND) return 0;
    int sign = (offset > 0) ? 1 : -1;
    float x = min((abs(offset) - DEADBAND) / (1.0f - DEADBAND), 1.0f);
    float mag = 5.0f + 75.0f * x;
    int steer = round(sign * min(mag, 80.0f));
    return steer;
}

// Keyboard Helper (Linux non-blocking)
int kbhit(void) {
    struct termios oldt, newt;
    int ch;
    int oldf;
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);
    ch = getchar();
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);
    if(ch != EOF) {
        ungetc(ch, stdin);
        return 1;
    }
    return 0;
}

// ==========================================
// MAIN LOOP
// ==========================================
int main() {
    cout << "========================================" << endl;
    cout << "  C++ HIGH-PERFORMANCE LANE DRIVER      " << endl;
    cout << "========================================" << endl;

    init_uart("/dev/serial0");

    thread capture_thread(capture_thread_func);
    thread uart_thread(uart_thread_func);

    Mat frame;
    float smooth_steer = 0.0f;
    int no_lane_cnt = 0;
    int fps_cnt = 0;
    auto fps_t0 = chrono::steady_clock::now();
    float fps = 0.0f;

    while (!app_exit) {
        // 1. Nhận input bàn phím
        if (kbhit()) {
            char ch = getchar();
            if (ch == 'a') {
                shared_running.store(true);
                smooth_steer = 0.0f;
                cout << "\n>>> TỰ LÁI BẮT ĐẦU" << endl;
            } else if (ch == 's') {
                shared_running.store(false);
                shared_steer.store(0); shared_speed.store(0);
                smooth_steer = 0.0f;
                cout << "\n>>> DỪNG" << endl;
            } else if (ch == 'q') {
                app_exit = true;
                break;
            }
        }

        // 2. Chờ ảnh mới từ Camera (Chống block CPU)
        {
            unique_lock<mutex> lock(frame_mutex);
            frame_cv.wait(lock, []{ return new_frame_ready || app_exit; });
            if (app_exit) break;
            shared_frame.copyTo(frame);
            new_frame_ready = false;
        }

        // 3. Xử lý ảnh OpenCV (Cực nhanh)
        Mat dbg_img; // Không debug ghi ảnh để max FPS
        DetectResult res = detect_lane_offset(frame, dbg_img);
        bool is_zebra = detect_zebra_crossing(frame);

        if (is_zebra) {
            _zebra_hold_cnt = ZEBRA_HOLD_FRAMES;
        } else if (_zebra_hold_cnt > 0) {
            _zebra_hold_cnt--;
        }
        bool in_zebra = _zebra_hold_cnt > 0;

        float offset = res.offset;
        if (!in_zebra) {
            if (res.found) {
                if (abs(offset - _prev_valid_offset) > MAX_OFFSET_JUMP) {
                    res.found = false;
                } else {
                    _prev_valid_offset = offset;
                }
            }
        } else {
            offset = _prev_valid_offset; // Giữ nguyên vô lăng khi qua Zebra
            res.found = true;
        }

        // 4. Tính toán điều khiển
        int raw_steer = offset_to_steer(res.found ? offset : 0.0f);
        smooth_steer = SMOOTH_ALPHA * raw_steer + (1 - SMOOTH_ALPHA) * smooth_steer;
        int steer_val = round(smooth_steer);

        if (res.found) no_lane_cnt = 0;
        else no_lane_cnt++;

        int spd = DRIVE_SPEED;
        if (!res.found) spd = SLOW_SPEED;
        if (no_lane_cnt > 15) {
            spd = 0; steer_val = 0; smooth_steer = 0.0f;
        }
        if (in_zebra) spd = ZEBRA_SPEED;

        // Cập nhật Atomic (Luồng UART tự lấy gửi đi)
        shared_steer.store(steer_val);
        if (shared_running.load()) shared_speed.store(spd);
        else shared_speed.store(0);

        // 5. Tính FPS
        fps_cnt++;
        auto now = chrono::steady_clock::now();
        chrono::duration<float> elapsed = now - fps_t0;
        if (elapsed.count() >= 1.0f) {
            fps = fps_cnt / elapsed.count();
            fps_cnt = 0;
            fps_t0 = now;
        }

        // Log ra Terminal
        printf("\r[%s] %-5s fps=%5.1f off=%+5.3f steer=%+4d spd=%2d noln=%3d",
               shared_running.load() ? "RUN " : "STOP",
               in_zebra ? "ZEBRA" : (res.found ? "LANE" : "NOLN"),
               fps,
               res.found ? offset : 0.0f,
               steer_val,
               spd,
               no_lane_cnt);
        fflush(stdout);
    }

    if (capture_thread.joinable()) capture_thread.join();
    if (uart_thread.joinable()) uart_thread.join();
    if (uart_fd != -1) close(uart_fd);

    cout << "\nĐã thoát an toàn." << endl;
    return 0;
}