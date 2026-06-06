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
// PHẦN CỨNG
// =============================================
#define ENC_L_A  21
#define ENC_L_B  19
#define ENC_R_A  22
#define ENC_R_B  18

#define IN1 26
#define IN2 25
#define IN3 32
#define IN4 33
#define ENA 27
#define ENB 14

#define SERVO_PIN 13

#define UART_NUM        UART_NUM_2
#define TX_PIN          17
#define RX_PIN          16
#define BUF_SIZE        256
#define PI_UART_BAUD    115200

// =============================================
// THÔNG SỐ XE
// =============================================
#define SERVO_CENTER  68
#define SERVO_LEFT     8
#define SERVO_RIGHT  128
#define SERVO_STEP     3

#define MAX_SPEED          30
#define FF_BASE           100
#define COMMAND_TIMEOUT_MS 10000   

static float Kp = 1.2f;
static float Ki = 0.08f;

// =============================================
// ENCODER — portMUX cho dual-core safe 
// =============================================
static portMUX_TYPE enc_mux = portMUX_INITIALIZER_UNLOCKED;

volatile int32_t encLeftCount  = 0;   
volatile int32_t encRightCount = 0;

static void IRAM_ATTR isr_enc_l_a(void* arg) {
    portENTER_CRITICAL_ISR(&enc_mux);
    if (gpio_get_level(ENC_L_A) == gpio_get_level(ENC_L_B)) encLeftCount++;
    else encLeftCount--;
    portEXIT_CRITICAL_ISR(&enc_mux);
}
static void IRAM_ATTR isr_enc_l_b(void* arg) {
    portENTER_CRITICAL_ISR(&enc_mux);
    if (gpio_get_level(ENC_L_A) != gpio_get_level(ENC_L_B)) encLeftCount++;
    else encLeftCount--;
    portEXIT_CRITICAL_ISR(&enc_mux);
}
static void IRAM_ATTR isr_enc_r_a(void* arg) {
    portENTER_CRITICAL_ISR(&enc_mux);
    if (gpio_get_level(ENC_R_A) == gpio_get_level(ENC_R_B)) encRightCount++;
    else encRightCount--;
    portEXIT_CRITICAL_ISR(&enc_mux);
}
static void IRAM_ATTR isr_enc_r_b(void* arg) {
    portENTER_CRITICAL_ISR(&enc_mux);
    if (gpio_get_level(ENC_R_A) != gpio_get_level(ENC_R_B)) encRightCount++;
    else encRightCount--;
    portEXIT_CRITICAL_ISR(&enc_mux);
}

// =============================================
// BIẾN ĐIỀU KHIỂN
// =============================================
static int   targetSpeedLeft  = 0;
static int   targetSpeedRight = 0;
static int   pwmLeft          = 0;
static int   pwmRight         = 0;
static float integralLeft     = 0.0f;
static float integralRight    = 0.0f;
static int32_t prevEncLeft    = 0;
static int32_t prevEncRight   = 0;

static int currentServoAngle  = SERVO_CENTER;
static int desiredServoAngle  = SERVO_CENTER;

static uint32_t lastCommandTime = 0;

// =============================================
// HARDWARE INIT
// =============================================
void hardware_init(void) {
    // GPIO động cơ
    gpio_config_t io_conf = {
        .intr_type    = GPIO_INTR_DISABLE,
        .mode         = GPIO_MODE_OUTPUT,
        .pin_bit_mask = (1ULL<<IN1)|(1ULL<<IN2)|(1ULL<<IN3)|(1ULL<<IN4),
        .pull_down_en = 0,
        .pull_up_en   = 0
    };
    gpio_config(&io_conf);
    gpio_set_level(IN1,0); gpio_set_level(IN2,0);
    gpio_set_level(IN3,0); gpio_set_level(IN4,0);

    // PWM động cơ — LEDC 5kHz 8-bit
    ledc_timer_config_t mt = {
        .speed_mode      = LEDC_LOW_SPEED_MODE,
        .timer_num       = LEDC_TIMER_0,
        .duty_resolution = LEDC_TIMER_8_BIT,
        .freq_hz         = 5000,
        .clk_cfg         = LEDC_AUTO_CLK
    };
    ledc_timer_config(&mt);

    ledc_channel_config_t ena = {
        .channel   = LEDC_CHANNEL_0, .duty = 0,
        .gpio_num  = ENA, .speed_mode = LEDC_LOW_SPEED_MODE,
        .hpoint    = 0,   .timer_sel  = LEDC_TIMER_0
    };
    ledc_channel_config_t enb = {
        .channel   = LEDC_CHANNEL_1, .duty = 0,
        .gpio_num  = ENB, .speed_mode = LEDC_LOW_SPEED_MODE,
        .hpoint    = 0,   .timer_sel  = LEDC_TIMER_0
    };
    ledc_channel_config(&ena);
    ledc_channel_config(&enb);

    // PWM servo — LEDC 50Hz 13-bit
    ledc_timer_config_t st = {
        .speed_mode      = LEDC_LOW_SPEED_MODE,
        .timer_num       = LEDC_TIMER_1,
        .duty_resolution = LEDC_TIMER_13_BIT,
        .freq_hz         = 50,
        .clk_cfg         = LEDC_AUTO_CLK
    };
    ledc_timer_config(&st);

    ledc_channel_config_t servo_ch = {
        .channel   = LEDC_CHANNEL_2, .duty = 0,
        .gpio_num  = SERVO_PIN, .speed_mode = LEDC_LOW_SPEED_MODE,
        .hpoint    = 0,         .timer_sel  = LEDC_TIMER_1
    };
    ledc_channel_config(&servo_ch);

    // Encoder GPIO + ISR
    gpio_config_t ec = {
        .intr_type    = GPIO_INTR_ANYEDGE,
        .mode         = GPIO_MODE_INPUT,
        .pin_bit_mask = (1ULL<<ENC_L_A)|(1ULL<<ENC_L_B)|(1ULL<<ENC_R_A)|(1ULL<<ENC_R_B),
        .pull_down_en = 0,
        .pull_up_en   = 1
    };
    gpio_config(&ec);
    gpio_install_isr_service(0);
    gpio_isr_handler_add(ENC_L_A, isr_enc_l_a, NULL);
    gpio_isr_handler_add(ENC_L_B, isr_enc_l_b, NULL);
    gpio_isr_handler_add(ENC_R_A, isr_enc_r_a, NULL);
    gpio_isr_handler_add(ENC_R_B, isr_enc_r_b, NULL);

    // UART2 → Raspberry Pi
    uart_config_t uc = {
        .baud_rate  = PI_UART_BAUD,
        .data_bits  = UART_DATA_8_BITS,
        .parity     = UART_PARITY_DISABLE,
        .stop_bits  = UART_STOP_BITS_1,
        .flow_ctrl  = UART_HW_FLOWCTRL_DISABLE
    };
    uart_param_config(UART_NUM, &uc);
    uart_set_pin(UART_NUM, TX_PIN, RX_PIN,
                 UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
    uart_driver_install(UART_NUM, BUF_SIZE * 2, 0, 0, NULL, 0);
}

// =============================================
// SERVO
// =============================================
void write_servo_angle(int angle) {
    // 13-bit @ 50Hz: 0.5ms=205, 2.4ms=983
    int duty = 205 + (983 - 205) * angle / 180;
    ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_2, duty);
    ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_2);
}

// =============================================
// MOTOR PWM
// =============================================
void apply_motor_pwm(int pL, int pR) {
    gpio_set_level(IN1, pL > 0 ? 1 : 0);
    gpio_set_level(IN2, pL < 0 ? 1 : 0);
    ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0, (uint32_t)abs(pL));
    ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0);

    gpio_set_level(IN3, pR > 0 ? 1 : 0);
    gpio_set_level(IN4, pR < 0 ? 1 : 0);
    ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_1, (uint32_t)abs(pR));
    ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_1);
}

void hard_stop(void) {
    targetSpeedLeft = targetSpeedRight = 0;
    pwmLeft = pwmRight = 0;
    integralLeft = integralRight = 0.0f;
    portENTER_CRITICAL(&enc_mux);
    prevEncLeft  = encLeftCount;
    prevEncRight = encRightCount;
    portEXIT_CRITICAL(&enc_mux);
    apply_motor_pwm(0, 0);
}

// =============================================
// car_forward / car_backward
// =============================================
void car_forward(int spdL, int spdR) {
    portENTER_CRITICAL(&enc_mux);
    if (targetSpeedLeft <= 0) {   
        encLeftCount  = encRightCount  = 0;
        prevEncLeft   = prevEncRight   = 0;
        integralLeft  = integralRight  = 0.0f;
    }
    targetSpeedLeft  = spdL;
    targetSpeedRight = spdR;
    portEXIT_CRITICAL(&enc_mux);
    lastCommandTime = esp_timer_get_time() / 1000;
}

void car_backward(int spdL, int spdR) {
    portENTER_CRITICAL(&enc_mux);
    if (targetSpeedLeft >= 0) {   
        encLeftCount  = encRightCount  = 0;
        prevEncLeft   = prevEncRight   = 0;
        integralLeft  = integralRight  = 0.0f;
    }
    targetSpeedLeft  = -spdL;
    targetSpeedRight = -spdR;
    portEXIT_CRITICAL(&enc_mux);
    lastCommandTime = esp_timer_get_time() / 1000;
}

// =============================================
// TASK 1: CONTROL — Core 1, 50Hz 
// =============================================
void control_task(void *pv) {
    TickType_t xLast = xTaskGetTickCount();
    const TickType_t xFreq = pdMS_TO_TICKS(20);   // 50 Hz

    while (1) {
        uint32_t now_ms = (uint32_t)(esp_timer_get_time() / 1000);

        // Watchdog: timeout → phanh khẩn cấp
        if ((targetSpeedLeft != 0 || targetSpeedRight != 0) &&
            (now_ms - lastCommandTime > COMMAND_TIMEOUT_MS)) {
            hard_stop();
            printf("[SAFE] UART timeout → phanh!\n");
        }

        // Đọc encoder an toàn 
        int32_t rawL, rawR;
        portENTER_CRITICAL(&enc_mux);
        rawL = encLeftCount;  rawR = encRightCount;
        portEXIT_CRITICAL(&enc_mux);
        int32_t deltaL = rawL - prevEncLeft;
        int32_t deltaR = rawR - prevEncRight;
        prevEncLeft = rawL; prevEncRight = rawR;

        if (targetSpeedLeft != 0 || targetSpeedRight != 0) {
            int dirL = (targetSpeedLeft  > 0) ? 1 : -1;
            int dirR = (targetSpeedRight > 0) ? 1 : -1;

            bool encDead = (deltaL == 0 && deltaR == 0);
            if (encDead) {
                int rawL_pwm = FF_BASE + abs(targetSpeedLeft)  * (255 - FF_BASE) / MAX_SPEED;
                int rawR_pwm = FF_BASE + abs(targetSpeedRight) * (255 - FF_BASE) / MAX_SPEED;
                pwmLeft  = (targetSpeedLeft  == 0) ? 0 : dirL * rawL_pwm;
                pwmRight = (targetSpeedRight == 0) ? 0 : dirR * rawR_pwm;
            } else {
                // PID bình thường
                float errL = (float)targetSpeedLeft  - (float)deltaL;
                float errR = (float)targetSpeedRight - (float)deltaR;
                integralLeft  += errL;
                integralRight += errR;
                // Anti-windup
                if (integralLeft  >  200.0f) integralLeft  =  200.0f;
                if (integralLeft  < -200.0f) integralLeft  = -200.0f;
                if (integralRight >  200.0f) integralRight =  200.0f;
                if (integralRight < -200.0f) integralRight = -200.0f;

                int baseL = (targetSpeedLeft  != 0) ?
                    (FF_BASE + abs(targetSpeedLeft)  * (255 - FF_BASE) / MAX_SPEED) : 0;
                int baseR = (targetSpeedRight != 0) ?
                    (FF_BASE + abs(targetSpeedRight) * (255 - FF_BASE) / MAX_SPEED) : 0;

                pwmLeft  = dirL * baseL + (int)(Kp * errL + Ki * integralLeft);
                pwmRight = dirR * baseR + (int)(Kp * errR + Ki * integralRight);
            }

            // Clamp
            if (pwmLeft  >  255) pwmLeft  =  255;
            if (pwmLeft  < -255) pwmLeft  = -255;
            if (pwmRight >  255) pwmRight =  255;
            if (pwmRight < -255) pwmRight = -255;

            apply_motor_pwm(pwmLeft, pwmRight);
        }

        // Servo smoothing
        if (desiredServoAngle != currentServoAngle) {
            int diff = desiredServoAngle - currentServoAngle;
            if (abs(diff) <= SERVO_STEP) currentServoAngle = desiredServoAngle;
            else currentServoAngle += (diff > 0) ? SERVO_STEP : -SERVO_STEP;
            write_servo_angle(currentServoAngle);
        }

        vTaskDelayUntil(&xLast, xFreq);
    }
}

// =============================================
// TASK 2: UART RX — Core 0
// =============================================
void uart_task(void *pv) {
    uint8_t data[BUF_SIZE];
    char    line[128];
    int     line_len = 0;

    while (1) {
        int len = uart_read_bytes(UART_NUM, data, BUF_SIZE - 1, pdMS_TO_TICKS(10));
        for (int i = 0; i < len; i++) {
            char c = (char)data[i];
            if (c == '\n') {
                line[line_len] = '\0';
                if (line_len >= 5) {
                    char   drive;
                    int    steer, spd;
                    if (sscanf(line, "%c,%d,%d", &drive, &steer, &spd) == 3) {
                        lastCommandTime = (uint32_t)(esp_timer_get_time() / 1000);

                        if (spd == 0) drive = 'S';
                        if (steer >  100) steer =  100;
                        if (steer < -100) steer = -100;

                        // Tính góc servo
                        int tgt = SERVO_CENTER;
                        if (steer <= 0)
                            tgt = SERVO_CENTER + steer * (SERVO_CENTER - SERVO_LEFT)  / 100;
                        else
                            tgt = SERVO_CENTER + steer * (SERVO_RIGHT  - SERVO_CENTER) / 100;
                        if (tgt < SERVO_LEFT)  tgt = SERVO_LEFT;
                        if (tgt > SERVO_RIGHT) tgt = SERVO_RIGHT;
                        desiredServoAngle = tgt;

                        // Vi sai điện tử
                        float diff = 0.55f;
                        int   spdL = spd, spdR = spd;
                        if (steer < 0)
                            spdL = spd - (int)(spd * (abs(steer) / 100.0f) * diff);
                        else if (steer > 0)
                            spdR = spd - (int)(spd * (steer       / 100.0f) * diff);
                        if (spdL > MAX_SPEED) spdL = MAX_SPEED;
                        if (spdR > MAX_SPEED) spdR = MAX_SPEED;
                        if (spdL < 1) spdL = 1;
                        if (spdR < 1) spdR = 1;

                        if (drive == 'F') {
                            car_forward(spdL, spdR);
                        } else if (drive == 'B') {
                            car_backward(spdL, spdR);
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

// =============================================
// ENTRY POINT
// =============================================
void app_main(void) {
    printf("\n==============================\n");
    printf("  ESP-IDF LANE CAR CONTROLLER \n");
    printf("  RX=GPIO16 | TX=GPIO17       \n");
    printf("==============================\n");

    hardware_init();
    write_servo_angle(SERVO_CENTER);

    // Core 0: UART (blocking-safe, không ảnh hưởng PID)
    xTaskCreatePinnedToCore(uart_task,    "uart_task",    4096, NULL,  5, NULL, 0);
    // Core 1: PID + Servo 50Hz — ưu tiên cao nhất
    xTaskCreatePinnedToCore(control_task, "control_task", 4096, NULL, 10, NULL, 1);
}