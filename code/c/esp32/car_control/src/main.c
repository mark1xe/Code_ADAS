#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "driver/uart.h"
#include "driver/ledc.h"
#include "esp_timer.h"

// =============================================
// --- CẤU HÌNH PHẦN CỨNG ---
// =============================================
#define ENC_L_A 21
#define ENC_L_B 19
#define ENC_R_A 22
#define ENC_R_B 18

#define IN1 26
#define IN2 25
#define IN3 33
#define IN4 32
#define ENA 27
#define ENB 14

#define SERVO_PIN 13

#define UART_NUM UART_NUM_2
#define TX_PIN 17
#define RX_PIN 16
#define BUF_SIZE 256
#define PI_UART_BAUD 115200

// =============================================
// --- THÔNG SỐ XE ---
// =============================================
#define SERVO_CENTER 68
#define SERVO_LEFT 40
#define SERVO_RIGHT 120
#define SERVO_STEP 3

#define MAX_SPEED 30
#define FF_BASE 100
#define COMMAND_TIMEOUT_MS 1000

float Kp = 1.2;
float Ki = 0.08;

// =============================================
// --- BIẾN TOÀN CỤC ---
// =============================================
volatile long encLeftCount = 0;
volatile long encRightCount = 0;

int targetSpeedLeft = 0;
int targetSpeedRight = 0;
int pwmLeft = 0;
int pwmRight = 0;

float integralLeft = 0;
float integralRight = 0;
long prevEncLeft = 0;
long prevEncRight = 0;

int currentServoAngle = SERVO_CENTER;
int desiredServoAngle = SERVO_CENTER;

uint32_t lastCommandTime = 0;

// =============================================
// --- INTERRUPT ENCODER ---
// =============================================
static void IRAM_ATTR isr_enc_l_a(void* arg) {
    if (gpio_get_level(ENC_L_A) == gpio_get_level(ENC_L_B)) encLeftCount++; else encLeftCount--;
}
static void IRAM_ATTR isr_enc_l_b(void* arg) {
    if (gpio_get_level(ENC_L_A) != gpio_get_level(ENC_L_B)) encLeftCount++; else encLeftCount--;
}
static void IRAM_ATTR isr_enc_r_a(void* arg) {
    if (gpio_get_level(ENC_R_A) == gpio_get_level(ENC_R_B)) encRightCount++; else encRightCount--;
}
static void IRAM_ATTR isr_enc_r_b(void* arg) {
    if (gpio_get_level(ENC_R_A) != gpio_get_level(ENC_R_B)) encRightCount++; else encRightCount--;
}

// =============================================
// --- CÀI ĐẶT NGOẠI VI (HARDWARE INIT) ---
// =============================================
void hardware_init() {
    // 1. Khởi tạo GPIO Động cơ
    gpio_config_t io_conf = {
        .intr_type = GPIO_INTR_DISABLE,
        .mode = GPIO_MODE_OUTPUT,
        .pin_bit_mask = (1ULL<<IN1)|(1ULL<<IN2)|(1ULL<<IN3)|(1ULL<<IN4),
        .pull_down_en = 0,
        .pull_up_en = 0
    };
    gpio_config(&io_conf);
    gpio_set_level(IN1, 0); gpio_set_level(IN2, 0);
    gpio_set_level(IN3, 0); gpio_set_level(IN4, 0);

    // 2. Khởi tạo PWM Động cơ (LEDC 5000Hz, 8-bit)
    ledc_timer_config_t motor_timer = {
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .timer_num  = LEDC_TIMER_0,
        .duty_resolution = LEDC_TIMER_8_BIT,
        .freq_hz = 5000,
        .clk_cfg = LEDC_AUTO_CLK
    };
    ledc_timer_config(&motor_timer);

    ledc_channel_config_t ch_ena = {.channel = LEDC_CHANNEL_0, .duty = 0, .gpio_num = ENA, .speed_mode = LEDC_LOW_SPEED_MODE, .hpoint = 0, .timer_sel = LEDC_TIMER_0};
    ledc_channel_config_t ch_enb = {.channel = LEDC_CHANNEL_1, .duty = 0, .gpio_num = ENB, .speed_mode = LEDC_LOW_SPEED_MODE, .hpoint = 0, .timer_sel = LEDC_TIMER_0};
    ledc_channel_config(&ch_ena);
    ledc_channel_config(&ch_enb);

    // 3. Khởi tạo PWM Servo (LEDC 50Hz, 13-bit)
    ledc_timer_config_t servo_timer = {
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .timer_num  = LEDC_TIMER_1,
        .duty_resolution = LEDC_TIMER_13_BIT, // 8192 steps
        .freq_hz = 50,
        .clk_cfg = LEDC_AUTO_CLK
    };
    ledc_timer_config(&servo_timer);
    ledc_channel_config_t ch_servo = {.channel = LEDC_CHANNEL_2, .duty = 0, .gpio_num = SERVO_PIN, .speed_mode = LEDC_LOW_SPEED_MODE, .hpoint = 0, .timer_sel = LEDC_TIMER_1};
    ledc_channel_config(&ch_servo);

    // 4. Khởi tạo Ngắt Encoder
    gpio_config_t enc_conf = {
        .intr_type = GPIO_INTR_ANYEDGE,
        .mode = GPIO_MODE_INPUT,
        .pin_bit_mask = (1ULL<<ENC_L_A)|(1ULL<<ENC_L_B)|(1ULL<<ENC_R_A)|(1ULL<<ENC_R_B),
        .pull_down_en = 0,
        .pull_up_en = 1
    };
    gpio_config(&enc_conf);
    gpio_install_isr_service(0);
    gpio_isr_handler_add(ENC_L_A, isr_enc_l_a, NULL);
    gpio_isr_handler_add(ENC_L_B, isr_enc_l_b, NULL);
    gpio_isr_handler_add(ENC_R_A, isr_enc_r_a, NULL);
    gpio_isr_handler_add(ENC_R_B, isr_enc_r_b, NULL);

    // 5. Khởi tạo UART2
    uart_config_t uart_config = {
        .baud_rate = PI_UART_BAUD,
        .data_bits = UART_DATA_8_BITS,
        .parity    = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE
    };
    uart_param_config(UART_NUM, &uart_config);
    uart_set_pin(UART_NUM, TX_PIN, RX_PIN, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
    uart_driver_install(UART_NUM, BUF_SIZE * 2, 0, 0, NULL, 0);
}

// =============================================
// --- HÀM ĐIỀU KHIỂN ---
// =============================================
void write_servo_angle(int angle) {
    // Servo 0-180 độ tương ứng ~0.5ms đến 2.4ms pulse (50Hz = 20ms)
    // 13-bit = 8192. MIN = 8192 * 0.5/20 = 205. MAX = 8192 * 2.4/20 = 983
    int duty = 205 + (983 - 205) * angle / 180;
    ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_2, duty);
    ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_2);
}

void apply_motor_pwm(int pL, int pR) {
    gpio_set_level(IN1, pL > 0 ? 1 : 0);
    gpio_set_level(IN2, pL < 0 ? 1 : 0);
    ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0, abs(pL));
    ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0);

    gpio_set_level(IN3, pR > 0 ? 1 : 0);
    gpio_set_level(IN4, pR < 0 ? 1 : 0);
    ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_1, abs(pR));
    ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_1);
}

void hard_stop() {
    targetSpeedLeft = 0; targetSpeedRight = 0;
    pwmLeft = 0; pwmRight = 0;
    integralLeft = 0; integralRight = 0;
    prevEncLeft = encLeftCount; prevEncRight = encRightCount;
    apply_motor_pwm(0, 0);
}

// =============================================
// --- TASK 1: PID & SERVO CONTROL (CORE 1) ---
// =============================================
void control_task(void *pvParameters) {
    TickType_t xLastWakeTime = xTaskGetTickCount();
    const TickType_t xFrequency = pdMS_TO_TICKS(20); // 50Hz Loop

    while (1) {
        uint32_t now = esp_timer_get_time() / 1000; // ms

        // 1. An toàn: Timeout phanh khẩn cấp
        if ((targetSpeedLeft != 0 || targetSpeedRight != 0) && (now - lastCommandTime > COMMAND_TIMEOUT_MS)) {
            hard_stop();
            printf("[SAFE] UART Timeout -> Phanh khan cap!\n");
        }

        // 2. Tính PID
        long deltaL = encLeftCount - prevEncLeft;
        long deltaR = encRightCount - prevEncRight;
        prevEncLeft = encLeftCount; prevEncRight = encRightCount;

        if (targetSpeedLeft != 0 || targetSpeedRight != 0) {
            int dirL = (targetSpeedLeft > 0) ? 1 : (targetSpeedLeft < 0 ? -1 : 0);
            int dirR = (targetSpeedRight > 0) ? 1 : (targetSpeedRight < 0 ? -1 : 0);

            float errL = targetSpeedLeft - deltaL;
            float errR = targetSpeedRight - deltaR;
            integralLeft  = integralLeft + errL;
            integralRight = integralRight + errR;
            if (integralLeft > 200) integralLeft = 200; else if (integralLeft < -200) integralLeft = -200;
            if (integralRight > 200) integralRight = 200; else if (integralRight < -200) integralRight = -200;

            int basePwmLeft = (targetSpeedLeft != 0) ? (FF_BASE + abs(targetSpeedLeft) * (255 - FF_BASE) / MAX_SPEED) : 0;
            int basePwmRight = (targetSpeedRight != 0) ? (FF_BASE + abs(targetSpeedRight) * (255 - FF_BASE) / MAX_SPEED) : 0;

            pwmLeft = dirL * basePwmLeft + (int)(Kp * errL + Ki * integralLeft);
            pwmRight = dirR * basePwmRight + (int)(Kp * errR + Ki * integralRight);
            
            if (pwmLeft > 255) pwmLeft = 255; else if (pwmLeft < -255) pwmLeft = -255;
            if (pwmRight > 255) pwmRight = 255; else if (pwmRight < -255) pwmRight = -255;

            apply_motor_pwm(pwmLeft, pwmRight);
        }

        // 3. Servo Smoothing
        if (desiredServoAngle != currentServoAngle) {
            int diff = desiredServoAngle - currentServoAngle;
            if (abs(diff) <= SERVO_STEP) currentServoAngle = desiredServoAngle;
            else currentServoAngle += (diff > 0) ? SERVO_STEP : -SERVO_STEP;
            write_servo_angle(currentServoAngle);
        }

        vTaskDelayUntil(&xLastWakeTime, xFrequency); // Đảm bảo đúng 20ms quay lại
    }
}

// =============================================
// --- TASK 2: ĐỌC UART TỪ RASPI (CORE 0) ---
// =============================================
void uart_task(void *pvParameters) {
    uint8_t data[BUF_SIZE];
    char line[128];
    int line_len = 0;

    while (1) {
        int len = uart_read_bytes(UART_NUM, data, BUF_SIZE - 1, pdMS_TO_TICKS(10));
        if (len > 0) {
            for (int i = 0; i < len; i++) {
                char c = data[i];
                if (c == '\n') {
                    line[line_len] = '\0';
                    
                    // Xử lý chuỗi (Ví dụ: F,50,15)
                    if (line_len >= 5) {
                        char drive; int steer, spd;
                        if (sscanf(line, "%c,%d,%d", &drive, &steer, &spd) == 3) {
                            lastCommandTime = esp_timer_get_time() / 1000;
                            
                            if (spd == 0) drive = 'S';
                            if (steer > 100) steer = 100; else if (steer < -100) steer = -100;
                            
                            // Tính góc Servo
                            int targetAngle = SERVO_CENTER;
                            if (steer <= 0) targetAngle = SERVO_CENTER + steer * (SERVO_CENTER - SERVO_LEFT) / 100;
                            else            targetAngle = SERVO_CENTER + steer * (SERVO_RIGHT - SERVO_CENTER) / 100;
                            if (targetAngle < SERVO_LEFT) targetAngle = SERVO_LEFT;
                            if (targetAngle > SERVO_RIGHT) targetAngle = SERVO_RIGHT;
                            desiredServoAngle = targetAngle;

                            // Vi sai điện tử
                            float diffFactor = 0.55;
                            int spdL = spd; int spdR = spd;
                            if (steer < 0) spdL = spd - (spd * (abs(steer) / 100.0) * diffFactor);
                            else if (steer > 0) spdR = spd - (spd * (steer / 100.0) * diffFactor);
                            
                            if (spdL > MAX_SPEED) spdL = MAX_SPEED;
                            if (spdR > MAX_SPEED) spdR = MAX_SPEED;

                            if (drive == 'F') {
                                if (targetSpeedLeft <= 0) { integralLeft = 0; integralRight = 0; }
                                targetSpeedLeft = spdL; targetSpeedRight = spdR;
                            } else if (drive == 'B') {
                                if (targetSpeedLeft >= 0) { integralLeft = 0; integralRight = 0; }
                                targetSpeedLeft = -spdL; targetSpeedRight = -spdR;
                            } else {
                                hard_stop();
                            }
                        }
                    }
                    line_len = 0;
                } else if (c != '\r' && line_len < 127) {
                    line[line_len++] = c;
                }
            }
        }
    }
}

// =============================================
// --- MAIN ENTRY ---
// =============================================
void app_main() {
    printf("\n==================================\n");
    printf("ESP-IDF BARE-METAL CONTROL READY\n");
    printf("RX=GPIO16 | TX=GPIO17\n");
    printf("==================================\n");

    hardware_init();
    write_servo_angle(SERVO_CENTER);

    // Tách Task để tận dụng 2 nhân CPU
    // Core 0 chuyên lo nhận UART blocking để không ảnh hưởng PID
    xTaskCreatePinnedToCore(uart_task, "uart_task", 4096, NULL, 5, NULL, 0);
    
    // Core 1 chạy vòng lặp PID cứng 50Hz ưu tiên cao nhất
    xTaskCreatePinnedToCore(control_task, "control_task", 4096, NULL, 10, NULL, 1);
}