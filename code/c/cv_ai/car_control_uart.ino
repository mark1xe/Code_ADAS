#include <Arduino.h>
#include <ESP32Servo.h>

// =============================================
// --- CHÂN ENCODER ---
// =============================================
const int ENC_L_A = 19;
const int ENC_L_B = 21;
const int ENC_R_A = 18;
const int ENC_R_B = 22;

// =============================================
// --- CHÂN ĐỘNG CƠ ---
// =============================================
const int IN1 = 26, IN2 = 25;
const int IN3 = 33, IN4 = 32;
const int ENA = 27, ENB = 14;

// =============================================
// --- UART2 ---
// RX2 = GPIO16, TX2 = GPIO17
// Pi TX (GPIO14) -> ESP32 RX2 (GPIO16)
// Pi RX (GPIO15) -> ESP32 TX2 (GPIO17)
// Protocol: "F,<steer>,<speed>\n" | "B,..,.." | "S,0,0\n"
// =============================================
#define PI_UART_BAUD 115200

// Bài tự test động cơ lúc khởi động: 1 = bật, 0 = tắt.
// Khi bật, ESP32 tự quay 2 bánh ~1.5s ngay sau khi boot để kiểm tra
#define BOOT_MOTOR_TEST 0

// =============================================
// --- SERVO LÁI (Timer 3 riêng, LEDC motor dùng timer 0/1) ---
// =============================================
const int SERVO_PIN    = 13;
const int SERVO_CENTER = 68;   // thẳng (giữa 40-120)
const int SERVO_LEFT   = 40;   // rẽ trái tối đa
const int SERVO_RIGHT  = 96;   // rẽ phải tối đa (96=68+28: cân với trái 68-40=28 -> hết quá đà, lái đối xứng)
Servo steerServo;
int currentServoAngle = SERVO_CENTER;
int desiredServoAngle = SERVO_CENTER;  // chỉ write khi thực sự thay đổi

// =============================================
// --- ENCODER ---
// =============================================
volatile long encLeftCount  = 0;
volatile long encRightCount = 0;

void IRAM_ATTR isrLeft() {
  bool A = digitalRead(ENC_L_A);
  bool B = digitalRead(ENC_L_B);
  if (A == B) encLeftCount++;
  else        encLeftCount--;
}
void IRAM_ATTR isrLeftB() {
  bool A = digitalRead(ENC_L_A);
  bool B = digitalRead(ENC_L_B);
  if (A != B) encLeftCount++;
  else        encLeftCount--;
}
void IRAM_ATTR isrRight() {
  bool A = digitalRead(ENC_R_A);
  bool B = digitalRead(ENC_R_B);
  if (A == B) encRightCount++;
  else        encRightCount--;
}
void IRAM_ATTR isrRightB() {
  bool A = digitalRead(ENC_R_A);
  bool B = digitalRead(ENC_R_B);
  if (A != B) encRightCount++;
  else        encRightCount--;
}

// =============================================
// --- CHẾ ĐỘ ĐIỀU KHIỂN TỐC ĐỘ ---
//   USE_ENCODER = 1 : PID vòng kín dùng encoder 
//   USE_ENCODER = 0 : open-loop thuần — chạy ổn định, phù hợp xe lái bằng servo. ĐANG BẬT.
// =============================================
#define USE_ENCODER 0

// =============================================
// --- PID + FEEDFORWARD ---
// =============================================
float Kp = 1.2;            
float Ki = 0.08;           
const int FF_BASE = 150;   // PWM cơ sở
const int MAX_SPEED = 30;  // encoder ticks/50ms

int  targetSpeedLeft  = 0;
int  targetSpeedRight = 0;
int  pwmLeft  = 0;
int  pwmRight = 0;
float integralLeft  = 0;
float integralRight = 0;
long  prevEncLeft  = 0;
long  prevEncRight = 0;
unsigned long prevTime    = 0;
unsigned long lastReport  = 0;
unsigned long lastCommandTime = 0;
const unsigned long COMMAND_TIMEOUT_MS = 10000; // 10s

// =============================================
// --- ĐIỀU KHIỂN MOTOR ---
// =============================================
void applyMotorPWM(int pL, int pR) {
  if (pL > 0)      { digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);  }
  else if (pL < 0) { digitalWrite(IN1, LOW);  digitalWrite(IN2, HIGH); }
  else             { digitalWrite(IN1, LOW);  digitalWrite(IN2, LOW);  }
  ledcWrite(ENA, constrain(abs(pL), 0, 255));

  if (pR > 0)      { digitalWrite(IN3, LOW);  digitalWrite(IN4, HIGH); }
  else if (pR < 0) { digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);  }
  else             { digitalWrite(IN3, LOW);  digitalWrite(IN4, LOW);  }
  ledcWrite(ENB, constrain(abs(pR), 0, 255));
}

void hardStop() {
  targetSpeedLeft  = 0;
  targetSpeedRight = 0;
  pwmLeft  = 0;
  pwmRight = 0;
  integralLeft  = 0;
  integralRight = 0;
  prevEncLeft  = encLeftCount;
  prevEncRight = encRightCount;
  applyMotorPWM(0, 0);
}

void markCommandSeen() { lastCommandTime = millis(); }

void computePID() {
  unsigned long now = millis();
  if (now - prevTime < 50) return;

  long deltaL = encLeftCount  - prevEncLeft;
  long deltaR = encRightCount - prevEncRight;
  prevEncLeft  = encLeftCount;
  prevEncRight = encRightCount;

  if (targetSpeedLeft == 0 && targetSpeedRight == 0) {
    hardStop();
    prevTime = now;
    return;
  }

  int dir = (targetSpeedLeft > 0) ? 1 : -1;

  // Encoder không có tín hiệu -> open-loop thuần
  bool encDead = (deltaL == 0 && deltaR == 0);

  if (encDead) {
    int rawPWM = map(abs(targetSpeedLeft), 0, MAX_SPEED, FF_BASE, 255);
    rawPWM     = constrain(rawPWM, FF_BASE, 255);
    pwmLeft    = dir * rawPWM;
    pwmRight   = dir * rawPWM;
  } else {
    float errL = targetSpeedLeft  - deltaL;
    float errR = targetSpeedRight - deltaR;
    integralLeft  = constrain(integralLeft  + errL, -200, 200);
    integralRight = constrain(integralRight + errR, -200, 200);
    pwmLeft  = constrain(dir * FF_BASE + (int)(Kp * errL + Ki * integralLeft),  -255, 255);
    pwmRight = constrain(dir * FF_BASE + (int)(Kp * errR + Ki * integralRight), -255, 255);
  }

  applyMotorPWM(pwmLeft, pwmRight);
  prevTime = now;
}

// =============================================
// --- SERVO ---
// =============================================
void updateServo() {
  if (desiredServoAngle != currentServoAngle) {
    currentServoAngle = desiredServoAngle;
    steerServo.write(currentServoAngle);
  }
}

// =============================================
// --- HÀM ĐIỀU KHIỂN XE ---
// =============================================
void carForward(int spd) {
  if (targetSpeedLeft <= 0) {
    encLeftCount = encRightCount = 0;
    prevEncLeft = prevEncRight = 0;
    integralLeft = integralRight = 0;
  }
  targetSpeedLeft  =  spd;
  targetSpeedRight =  spd;
  markCommandSeen();
}
void carBackward(int spd) {
  if (targetSpeedLeft >= 0) {
    encLeftCount = encRightCount = 0;
    prevEncLeft = prevEncRight = 0;
    integralLeft = integralRight = 0;
  }
  targetSpeedLeft  = -spd;
  targetSpeedRight = -spd;
  markCommandSeen();
}
void carStop() {
  markCommandSeen();
  hardStop();
}

// =============================================
// --- UART: nhận lệnh từ Raspberry Pi ---
// Format: "<DRIVE>,<steer_val>,<SPEED>\n"
//   DRIVE    : F | B | S
//   steer_val: -100 (full trái) ... 0 (thẳng) ... +100 (full phải)
//   SPEED    : 1..MAX_SPEED
// Ánh xạ steer_val -> góc servo bằng nội suy 2 vế bất đối xứng.
// =============================================
void handleUART() {
  if (!Serial2.available()) return;

  String line = Serial2.readStringUntil('\n');
  line.trim();
  if (line.length() < 5) return;

  int c1 = line.indexOf(',');
  int c2 = line.lastIndexOf(',');
  if (c1 < 0 || c2 == c1) return;  // format sai

  char drive   = (char)line.charAt(0);
  int  steerVal = line.substring(c1 + 1, c2).toInt();  // -100..+100
  int  spd      = line.substring(c2 + 1).toInt();

  // Ánh xạ steer -> góc servo
  steerVal = constrain(steerVal, -100, 100);
  int targetAngle;
  if (steerVal <= 0)
    targetAngle = SERVO_CENTER + steerVal * (SERVO_CENTER - SERVO_LEFT) / 100;
  else
    targetAngle = SERVO_CENTER + steerVal * (SERVO_RIGHT - SERVO_CENTER) / 100;
  desiredServoAngle = constrain(targetAngle, SERVO_LEFT, SERVO_RIGHT);
  markCommandSeen();

  switch (drive) {
    // speed <= 0 -> DỪNG HẲN (không kẹp lên 1). Đây là lệnh dừng
    // khi gặp biển STOP / đèn đỏ / NO_ENTRY (RPi gửi "F,steer,0").
    case 'F': if (spd <= 0) carStop(); else carForward(constrain(spd, 1, MAX_SPEED));  break;
    case 'B': if (spd <= 0) carStop(); else carBackward(constrain(spd, 1, MAX_SPEED)); break;
    case 'S': carStop();                                 break;
    default: break;
  }
}

// =============================================
// --- SETUP ---
// =============================================
void setup() {
  Serial.begin(115200);

  // Motor pins
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  // LEDC cho motor (timer 0/1) — KHÔNG đụng timer 3 của servo
  ledcAttach(ENA, 5000, 8);
  ledcAttach(ENB, 5000, 8);
  applyMotorPWM(0, 0);

  // Encoder
  pinMode(ENC_L_A, INPUT_PULLUP);
  pinMode(ENC_L_B, INPUT_PULLUP);
  pinMode(ENC_R_A, INPUT_PULLUP);
  pinMode(ENC_R_B, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENC_L_A), isrLeft,  CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_L_B), isrLeftB, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_R_A), isrRight, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_R_B), isrRightB, CHANGE);

  // Servo — allocate timer 3 TRƯỚC khi ledcAttach motor đã chạy ở trên
  ESP32PWM::allocateTimer(3);
  steerServo.setPeriodHertz(50);
  steerServo.attach(SERVO_PIN);
  delay(200);
  steerServo.write(SERVO_CENTER);
  delay(300);
  lastCommandTime = millis();

  // UART2: nhận lệnh từ Raspberry Pi
  Serial2.begin(PI_UART_BAUD, SERIAL_8N1, 16, 17);  // RX=GPIO16, TX=GPIO17
  Serial.println("UART2 ready (RX=GPIO16, TX=GPIO17) - cho lenh tu Pi...");
  Serial.println("[MODE] UART-only, KHONG WiFi/Web.");

#if BOOT_MOTOR_TEST
  // --- Test TỪNG KÊNH riêng, cả 2 chiều, lặp 3 vòng ---
  // Áp PWM trực tiếp, BỎ QUA PID/UART/encoder. Chỉ cần NHÌN bánh xe.
  // TRÁI:  IN1=GPIO26, IN2=GPIO25, ENA=GPIO27
  // PHẢI:  IN3=GPIO33, IN4=GPIO32, ENB=GPIO14
  Serial.println("\n===== MOTOR CHANNEL TEST (3 vong) =====");
  for (int k = 1; k <= 3; k++) {
    Serial.printf("--- Vong %d ---\n", k);
    Serial.println("[1] TRAI  - tien (IN1=H,IN2=L,ENA=200)");
    applyMotorPWM(200, 0);   delay(1500);  applyMotorPWM(0, 0);  delay(500);
    Serial.println("[2] TRAI  - lui  (IN1=L,IN2=H,ENA=200)");
    applyMotorPWM(-200, 0);  delay(1500);  applyMotorPWM(0, 0);  delay(500);
    Serial.println("[3] PHAI  - tien (IN4=H,IN3=L,ENB=200)");
    applyMotorPWM(0, 200);   delay(1500);  applyMotorPWM(0, 0);  delay(500);
    Serial.println("[4] PHAI  - lui  (IN4=L,IN3=H,ENB=200)");
    applyMotorPWM(0, -200);  delay(1500);  applyMotorPWM(0, 0);  delay(500);
  }
  applyMotorPWM(0, 0);
  Serial.println("===== TEST XONG — chuyen sang che do UART =====\n");
  lastCommandTime = millis();  // reset timeout sau test
#endif
}

// =============================================
// --- LOOP ---
// =============================================
void loop() {
  handleUART();   // nhận lệnh từ Raspberry Pi qua Serial2
  computePID();   // PID tốc độ, xuất PWM motor (mỗi 50 ms)
  updateServo();  // write servo nếu góc thay đổi

  // Dừng khẩn cấp nếu mất lệnh > 10s
  if ((targetSpeedLeft != 0 || targetSpeedRight != 0) &&
      millis() - lastCommandTime > COMMAND_TIMEOUT_MS) {
    hardStop();
    Serial.println("[SAFE] Command timeout -> hard stop");
  }

  // Log encoder mỗi 2s (debug; có thể bỏ nếu muốn tối giản hơn)
  if (millis() - lastReport >= 2000) {
    lastReport = millis();
    Serial.printf("[ENC] L:%ld R:%ld | PWM L:%d R:%d | Servo:%d | Target:%d\n",
                  encLeftCount, encRightCount, pwmLeft, pwmRight,
                  currentServoAngle, targetSpeedLeft);
  }
}
