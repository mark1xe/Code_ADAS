#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <ESP32Servo.h>

// =============================================
// --- WIFI ---
// =============================================
const char* ssid = "Thanh Thuy";
const char* password = "thanhthuy";
WebServer server(80);

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
// --- UART2 (giao tiếp với Raspberry Pi) ---
// RX2 = GPIO16, TX2 = GPIO17  ← KHÔNG xung đột bootloader
// Pi TX (GPIO14) → ESP32 RX2 (GPIO16)
// Pi RX (GPIO15) → ESP32 TX2 (GPIO17)
// Protocol: "F,C,5\n" | "F,L,5\n" | "F,R,5\n" | "S,C,0\n"
// =============================================
#define PI_UART_BAUD 115200

// =============================================
// --- SERVO LÁI ---
// Servo dùng Timer 3 riêng, LEDC motor dùng Timer 0,1
// Tránh xung đột hoàn toàn
// =============================================

const int SERVO_PIN = 13;
const int SERVO_CENTER = 68;  // thẳng (giữa 40-120)
const int SERVO_LEFT = 40;    // rẽ trái tối đa
const int SERVO_RIGHT = 120;  // rẽ phải tối đa
Servo steerServo;
int currentServoAngle = SERVO_CENTER;
int desiredServoAngle = SERVO_CENTER;  // chỉ write khi thực sự thay đổi

// =============================================
// --- ENCODER ---
// =============================================
volatile long encLeftCount = 0;
volatile long encRightCount = 0;

void IRAM_ATTR isrLeft() {
  bool A = digitalRead(ENC_L_A);
  bool B = digitalRead(ENC_L_B);
  if (A == B) encLeftCount++;
  else encLeftCount--;
}
void IRAM_ATTR isrLeftB() {
  bool A = digitalRead(ENC_L_A);
  bool B = digitalRead(ENC_L_B);
  if (A != B) encLeftCount++;
  else encLeftCount--;
}
void IRAM_ATTR isrRight() {
  bool A = digitalRead(ENC_R_A);
  bool B = digitalRead(ENC_R_B);
  if (A == B) encRightCount++;
  else encRightCount--;  // đảo dấu: tiến = dương
}
void IRAM_ATTR isrRightB() {
  bool A = digitalRead(ENC_R_A);
  bool B = digitalRead(ENC_R_B);
  if (A != B) encRightCount++;
  else encRightCount--;  // đảo dấu
}

// =============================================
// --- PID + FEEDFORWARD ---
// Kp/Ki tăng để bù lệch nhanh hơn
// FF_BASE giảm để xe chạy chậm, dễ kiểm soát
// =============================================
float Kp = 1.2;            // Bù lệch nhanh
float Ki = 0.08;           // Tích lũy bù lệch
const int FF_BASE = 150;   // PWM cơ sở đủ mạnh để xe chạy
const int MAX_SPEED = 30;  // encoder ticks/50ms

int targetSpeedLeft = 0;
int targetSpeedRight = 0;
int pwmLeft = 0;
int pwmRight = 0;
float integralLeft = 0;
float integralRight = 0;
long prevEncLeft = 0;
long prevEncRight = 0;
unsigned long prevTime = 0;
unsigned long lastReport = 0;
unsigned long lastCommandTime = 0;
const unsigned long COMMAND_TIMEOUT_MS = 10000; // 10s

// =============================================
// --- ĐIỀU KHIỂN MOTOR ---
// Chỉ dùng ledcWrite cho ENA/ENB
// KHÔNG đụng vào bất kỳ hàm servo nào ở đây
// =============================================
void applyMotorPWM(int pL, int pR) {
  if (pL > 0) {
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);
  } else if (pL < 0) {
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, HIGH);
  } else {
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, LOW);
  }
  ledcWrite(ENA, constrain(abs(pL), 0, 255));

  if (pR > 0) {
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, HIGH);
  } else if (pR < 0) {
    digitalWrite(IN3, HIGH);
    digitalWrite(IN4, LOW);
  } else {
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, LOW);
  }
  ledcWrite(ENB, constrain(abs(pR), 0, 255));
}

void hardStop() {
  targetSpeedLeft = 0;
  targetSpeedRight = 0;
  pwmLeft = 0;
  pwmRight = 0;
  integralLeft = 0;
  integralRight = 0;
  prevEncLeft = encLeftCount;
  prevEncRight = encRightCount;
  applyMotorPWM(0, 0);
}

void markCommandSeen() {
  lastCommandTime = millis();
}

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

  // Nếu encoder không có tín hiệu (chưa kết nối) → open-loop thuần
  // nhân abs(target) vào [FF_BASE, 255]
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

    // Ánh xạ tốc độ mục tiêu thành PWM cơ sở động (Dynamic Feedforward)
    // Giả sử PWM = 60 là mức điện áp tối thiểu để bánh bắt đầu lăn
    int basePwmLeft = map(abs(targetSpeedLeft), 0, MAX_SPEED, 60, 255);
    int basePwmRight = map(abs(targetSpeedRight), 0, MAX_SPEED, 60, 255);
    if (targetSpeedLeft == 0) basePwmLeft = 0;
    if (targetSpeedRight == 0) basePwmRight = 0;

    pwmLeft  = constrain(dir * basePwmLeft + (int)(Kp * errL + Ki * integralLeft),  -255, 255);
    pwmRight = constrain(dir * basePwmRight + (int)(Kp * errR + Ki * integralRight), -255, 255);
  }

// =============================================
// --- SERVO: chỉ write khi góc thực sự thay đổi
// Tách hoàn toàn khỏi vòng PID motor
// =============================================
void updateServo() {
  if (desiredServoAngle != currentServoAngle) {
    currentServoAngle = desiredServoAngle;
    steerServo.write(currentServoAngle);
  }
}

// =============================================
// --- HÀM ĐIỀU KHIỂN XE ---
// KHÔNG gọi servo ở đây — tách biệt hoàn toàn
// =============================================
void carForward(int spdL, int spdR) {
  if (targetSpeedLeft <= 0) {
    encLeftCount = encRightCount = 0;
    prevEncLeft = prevEncRight = 0;
    integralLeft = integralRight = 0;
  }
  targetSpeedLeft  = spdL;
  targetSpeedRight = spdR;
  markCommandSeen();
}

void carBackward(int spdL, int spdR) {
  if (targetSpeedLeft >= 0) {
    encLeftCount = encRightCount = 0;
    prevEncLeft = prevEncRight = 0;
    integralLeft = integralRight = 0;
  }
  targetSpeedLeft  = -spdL;
  targetSpeedRight = -spdR;
  markCommandSeen();
}

// Servo chỉ đặt desired — updateServo() trong loop mới write
void steerCenter() {
  desiredServoAngle = SERVO_CENTER;
  markCommandSeen();
}
void steerLeft() {
  desiredServoAngle = SERVO_LEFT;
  markCommandSeen();
}
void steerRight() {
  desiredServoAngle = SERVO_RIGHT;
  markCommandSeen();
}

// =============================================
// --- UART: nhận lệnh từ Raspberry Pi ---
// Format: "<DRIVE>,<steer_val>,<SPEED>\n"
//   DRIVE    : F(forward) | B(backward) | S(stop)
//   steer_val: số nguyên -100 (full trái) … 0 (thẳng) … +100 (full phải)
//   SPEED    : số nguyên 1-8
// Ví dụ: "F,-45,5\n"  "F,0,5\n"  "F,72,4\n"  "S,0,0\n"
//
// Ánh xạ steer_val → góc servo:
//   steer_val = -100 → SERVO_LEFT  (góc nhỏ nhất)
//   steer_val =    0 → SERVO_CENTER
//   steer_val = +100 → SERVO_RIGHT (góc lớn nhất)
//   Dùng nội suy tuyến tính 2 vế vì servo không đối xứng (62-50=12, 90-62=28)
// =============================================
void handleUART() {
  if (!Serial2.available()) return;

  String line = Serial2.readStringUntil('\n');
  line.trim();
  if (line.length() < 5) return;

  int c1 = line.indexOf(',');
  int c2 = line.lastIndexOf(',');
  if (c1 < 0 || c2 == c1) return;  // format sai

  char drive = (char)line.charAt(0);
  int steerVal = line.substring(c1 + 1, c2).toInt();  // -100 … +100
  int spd = line.substring(c2 + 1).toInt();

  // --- Ánh xạ steer_val → góc servo bằng nội suy tuyến tính ---
  steerVal = constrain(steerVal, -100, 100);
  int targetAngle;
  if (steerVal <= 0) {
    // [-100, 0] → [SERVO_LEFT, SERVO_CENTER]
    targetAngle = SERVO_CENTER + steerVal * (SERVO_CENTER - SERVO_LEFT) / 100;
  } else {
    // [0, +100] → [SERVO_CENTER, SERVO_RIGHT]
    targetAngle = SERVO_CENTER + steerVal * (SERVO_RIGHT - SERVO_CENTER) / 100;
  }
  desiredServoAngle = constrain(targetAngle, SERVO_LEFT, SERVO_RIGHT);
  markCommandSeen();

  // --- Vi sai điện tử (Electronic Differential) ---
  // Giảm tốc bánh bên trong góc cua để xe không bị ghì
  float diffFactor = 0.55; // Độ giảm (0.55 = giảm tối đa 55% tốc độ bánh trong)
  int spdL = spd;
  int spdR = spd;

  if (steerVal < 0) {
      // Cua trái -> Bánh trái nằm bên trong
      spdL = spd - (spd * (abs(steerVal) / 100.0) * diffFactor);
  } else if (steerVal > 0) {
      // Cua phải -> Bánh phải nằm bên trong
      spdR = spd - (spd * (steerVal / 100.0) * diffFactor);
  }

  // --- Áp dụng drive ---
  switch (drive) {
    case 'F': carForward(constrain(spdL, 1, MAX_SPEED), constrain(spdR, 1, MAX_SPEED)); break;
    case 'B': carBackward(constrain(spdL, 1, MAX_SPEED), constrain(spdR, 1, MAX_SPEED)); break;
    case 'S': carStop(); break;
    default: break;
  }
}

// =============================================
// --- WEB HANDLERS ---
// =============================================

void handleForward() {
  int spd = server.hasArg("spd") ? server.arg("spd").toInt() : 15;
  carForward(constrain(spd, 1, MAX_SPEED));
  server.send(200, "text/plain", "OK");
}
void handleBackward() {
  int spd = server.hasArg("spd") ? server.arg("spd").toInt() : 15;
  carBackward(constrain(spd, 1, MAX_SPEED));
  server.send(200, "text/plain", "OK");
}
void handleKeepalive() { markCommandSeen(); server.send(200, "text/plain", "OK"); }
void handleStop() {
  carStop();
  server.send(200, "text/plain", "OK");
}
void handleSteerLeft() {
  steerLeft();
  server.send(200, "text/plain", "OK");
}
void handleSteerRight() {
  steerRight();
  server.send(200, "text/plain", "OK");
}
void handleSteerCenter() {
  steerCenter();
  server.send(200, "text/plain", "OK");
}
void handleSteerAngle() {
  if (server.hasArg("v")) {
    int angle = constrain(server.arg("v").toInt(), SERVO_LEFT, SERVO_RIGHT);
    desiredServoAngle = angle;
    markCommandSeen();
  }
  server.send(200, "text/plain", "OK");
}

void handleTelemetry() {
  char json[200];
  snprintf(json, sizeof(json),
           "{\"encL\":%ld,\"encR\":%ld,\"pwmL\":%d,\"pwmR\":%d,\"servo\":%d,\"targetL\":%d,\"targetR\":%d}",
           encLeftCount, encRightCount, pwmLeft, pwmRight,
           currentServoAngle, targetSpeedLeft, targetSpeedRight);
  server.send(200, "application/json", json);
}

// =============================================
// --- TRANG WEB ---
// =============================================
const char INDEX_HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<title>RC Car</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');
  :root {
    --bg:#0a0a0f; --panel:#111118; --border:#1e1e2e;
    --green:#00f5c4; --pink:#f500a0; --yellow:#ffcc00;
    --text:#c8d0e0; --dim:#3a3a5a;
  }
  *{margin:0;padding:0;box-sizing:border-box;-webkit-tap-highlight-color:transparent;}
  body{
    background:var(--bg);color:var(--text);
    font-family:'Share Tech Mono',monospace;
    min-height:100vh;display:flex;flex-direction:column;
    align-items:center;padding:16px;
    background-image:
      radial-gradient(ellipse at 20% 20%,rgba(0,245,196,.05) 0%,transparent 50%),
      radial-gradient(ellipse at 80% 80%,rgba(245,0,160,.05) 0%,transparent 50%);
  }
  h1{font-family:'Orbitron',monospace;font-size:1.4rem;font-weight:900;
    letter-spacing:.15em;
    background:linear-gradient(90deg,var(--green),var(--pink));
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    text-align:center;margin-bottom:4px;}
  .sub{font-size:.7rem;color:var(--dim);text-align:center;letter-spacing:.1em;margin-bottom:16px;}
  .tele{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;width:100%;max-width:500px;margin-bottom:14px;}
  .tbox{background:var(--panel);border:1px solid var(--border);border-radius:8px;
    padding:10px 6px;text-align:center;border-top:2px solid var(--green);}
  .tbox.wide{grid-column:span 2;}
  .tlabel{font-size:.55rem;color:var(--dim);letter-spacing:.08em;text-transform:uppercase;margin-bottom:3px;}
  .tval{font-family:'Orbitron',monospace;font-size:1rem;font-weight:700;color:var(--green);}
  .tval.neg{color:var(--pink);} .tval.mid{color:var(--yellow);}
  .card{width:100%;max-width:500px;background:var(--panel);border:1px solid var(--border);
    border-radius:10px;padding:14px;margin-bottom:12px;}
  .card-header{display:flex;justify-content:space-between;margin-bottom:10px;}
  .clabel{font-size:.6rem;color:var(--dim);letter-spacing:.12em;text-transform:uppercase;}
  .cval{font-family:'Orbitron',monospace;font-size:1rem;color:var(--green);}
  input[type=range]{-webkit-appearance:none;width:100%;height:5px;
    background:var(--border);border-radius:3px;}
  input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:24px;height:24px;
    border-radius:50%;background:var(--green);
    box-shadow:0 0 10px rgba(0,245,196,.5);cursor:pointer;}
  .btn{background:var(--panel);border:1px solid var(--border);border-radius:10px;
    color:var(--text);font-family:'Orbitron',monospace;font-weight:700;
    cursor:pointer;display:flex;align-items:center;justify-content:center;
    transition:all .1s;user-select:none;touch-action:manipulation;}
  .btn:active{transform:scale(.93);}
  .steer-row{display:flex;gap:8px;width:100%;max-width:500px;margin-bottom:12px;}
  .btn-steer{flex:1;height:54px;font-size:.75rem;letter-spacing:.08em;}
  .btn-steer.sl:active{border-color:var(--green);background:rgba(0,245,196,.1);color:var(--green);}
  .btn-steer.sc:active{border-color:var(--yellow);background:rgba(255,204,0,.1);color:var(--yellow);}
  .btn-steer.sr:active{border-color:var(--pink);background:rgba(245,0,160,.1);color:var(--pink);}
  .drive-grid{display:grid;grid-template-columns:1fr 1fr 1fr;
    grid-template-rows:70px 70px;gap:8px;width:100%;max-width:500px;}
  .btn-fwd{grid-column:2;grid-row:1;font-size:1.6rem;border-color:var(--green);color:var(--green);}
  .btn-fwd:active{background:rgba(0,245,196,.15);box-shadow:0 0 20px rgba(0,245,196,.3);}
  .btn-stp{grid-column:2;grid-row:2;font-size:.7rem;letter-spacing:.08em;border-color:var(--yellow);color:var(--yellow);}
  .btn-stp:active{background:rgba(255,204,0,.15);}
  .btn-bwd{grid-column:3;grid-row:2;font-size:1.6rem;border-color:var(--pink);color:var(--pink);}
  .btn-bwd:active{background:rgba(245,0,160,.15);}
  .status{display:flex;align-items:center;gap:8px;font-size:.65rem;color:var(--dim);margin-top:10px;}
  .dot{width:7px;height:7px;border-radius:50%;background:var(--green);
    box-shadow:0 0 6px var(--green);animation:blink 1.5s infinite;}
  @keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
</style>
</head>
<body>
<h1>&#9664;&#9654; RC CAR</h1>
<div class="sub">ESP32 WIFI CONTROL</div>
<div class="tele">
  <div class="tbox"><div class="tlabel">ENC L</div><div class="tval" id="eL">0</div></div>
  <div class="tbox"><div class="tlabel">ENC R</div><div class="tval" id="eR">0</div></div>
  <div class="tbox"><div class="tlabel">PWM L</div><div class="tval" id="pL">0</div></div>
  <div class="tbox"><div class="tlabel">PWM R</div><div class="tval" id="pR">0</div></div>
  <div class="tbox wide"><div class="tlabel">SERVO</div><div class="tval mid" id="sv">72</div></div>
  <div class="tbox wide"><div class="tlabel">TARGET</div><div class="tval" id="tg">0</div></div>
</div>
<div class="card">
  <div class="card-header">
    <span class="clabel">&#9654; SPEED</span>
    <span class="cval" id="spdDisp">15</span>
  </div>
  <input type="range" id="spdSlider" min="5" max="30" value="15"
    oninput="document.getElementById('spdDisp').innerText=this.value">
</div>
<div class="steer-row">
  <button class="btn btn-steer sl" onpointerdown="steerHold('L')" onpointerup="steerRelease()" onpointerleave="steerRelease()">&#9664; TRAI</button>
  <button class="btn btn-steer sc" onpointerdown="steerGoCenter()">&#9675; GIUA</button>
  <button class="btn btn-steer sr" onpointerdown="steerHold('R')" onpointerup="steerRelease()" onpointerleave="steerRelease()">PHAI &#9654;</button>
</div>
<div class="drive-grid">
  <button class="btn btn-fwd"
    onpointerdown="hold('fwd')" onpointerup="release()" onpointerleave="release()">&#9650;</button>
  <button class="btn btn-stp" onpointerdown="cmd('/stop')">&#9632; STOP</button>
  <button class="btn btn-bwd"
    onpointerdown="hold('bwd')" onpointerup="release()" onpointerleave="release()">&#9660;</button>
</div>
<div class="status"><div class="dot"></div><span id="st">CONNECTING...</span></div>
<script>
  let timer=null;
  let steerTimer=null;
  let steerAngle=80;
  let driveActive=false;
  const STEER_MIN=40, STEER_MAX=120, STEER_CTR=80, STEER_STEP=4;
  function spd(){return document.getElementById('spdSlider').value;}
  async function cmd(url){try{await fetch(url);}catch(e){}}
  function hold(dir){
    clearInterval(timer);
    driveActive=true;
    const driveUrl=(dir==='fwd')?'/forward?spd='+spd():'/backward?spd='+spd();
    cmd(driveUrl);
    // keepalive để giữ lệnh, KHÔNG reset encoder
    timer=setInterval(()=>{ if(driveActive) cmd('/keepalive'); },400);
  }
  function release(){
    if(!driveActive) return;  // bỏ qua nếu nút drive không đang giữ
    driveActive=false;
    clearInterval(timer);
    timer=null;
    cmd('/stop');
  }
  function steerHold(dir){
    clearInterval(steerTimer);
    function step(){
      if(dir==='L') steerAngle=Math.max(STEER_MIN,steerAngle-STEER_STEP);
      else          steerAngle=Math.min(STEER_MAX,steerAngle+STEER_STEP);
      cmd('/steer/angle?v='+steerAngle);
    }
    step();
    steerTimer=setInterval(step,80);
  }
  function steerRelease(){clearInterval(steerTimer);steerTimer=null;}
  function steerGoCenter(){steerAngle=STEER_CTR;cmd('/steer/center');}
  // Chỉ dừng khi rời tab/app, KHÔNG dừng khi thả nút lái
  window.addEventListener('blur', ()=>{release();steerRelease();});
  document.addEventListener('visibilitychange', ()=>{ if(document.hidden){release();steerRelease();} });
  async function poll(){
    try{
      const r=await fetch('/telemetry');
      const d=await r.json();
      const set=(id,v,cls)=>{const el=document.getElementById(id);el.innerText=v;el.className='tval'+(cls||'');};
      set('eL',d.encL,d.encL<0?' neg':'');
      set('eR',d.encR,d.encR<0?' neg':'');
      set('pL',d.pwmL,d.pwmL<0?' neg':'');
      set('pR',d.pwmR,d.pwmR<0?' neg':'');
      set('sv',d.servo,' mid');
      set('tg',d.targetL,d.targetL<0?' neg':'');
      document.getElementById('st').innerText='OK // '+new Date().toLocaleTimeString('vi-VN');
    }catch(e){document.getElementById('st').innerText='DISCONNECTED!';}
  }
  setInterval(poll,500);poll();
</script>
</body>
</html>
)rawliteral";

void handleRoot() {
  server.send_P(200, "text/html", INDEX_HTML);
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

  // LEDC cho motor — chỉ định rõ timer 0 và 1 để KHÔNG đụng timer 3 của servo
  ledcAttach(ENA, 5000, 8);
  ledcAttach(ENB, 5000, 8);
  applyMotorPWM(0, 0);

  // Encoder
  pinMode(ENC_L_A, INPUT_PULLUP);
  pinMode(ENC_L_B, INPUT_PULLUP);
  pinMode(ENC_R_A, INPUT_PULLUP);
  pinMode(ENC_R_B, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENC_L_A), isrLeft, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_L_B), isrLeftB, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_R_A), isrRight, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_R_B), isrRightB, CHANGE);

  // Servo — allocate timer 3 riêng biệt, KHÔNG dùng timer 0/1/2
  // Thứ tự quan trọng: allocate timer TRƯỚC KHI ledcAttach motor
  ESP32PWM::allocateTimer(3);  // ← Timer 3 riêng cho servo
  steerServo.setPeriodHertz(50);
  steerServo.attach(SERVO_PIN);  // default pulse range, đúng với góc đã test
  delay(200);
  steerServo.write(SERVO_CENTER);
  delay(300);  // chờ servo về vị trí trước khi tiếp tục
  lastCommandTime = millis();

  // WiFi
  Serial.printf("Ket noi: %s\n", ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.printf("\nIP: http://%s\n", WiFi.localIP().toString().c_str());

  // Web routes
  server.on("/", handleRoot);
  server.on("/forward", handleForward);
  server.on("/backward", handleBackward);
  server.on("/stop", handleStop);
  server.on("/steer/left", handleSteerLeft);
  server.on("/steer/right", handleSteerRight);
  server.on("/steer/center", handleSteerCenter);
  server.on("/steer/angle", handleSteerAngle);
  server.on("/telemetry", handleTelemetry);
  server.begin();
  Serial.println("Server started!");

  // UART2: nhận lệnh từ Raspberry Pi qua GPIO16(RX2)/GPIO17(TX2)
  // Dùng GPIO16/17 → KHÔNG đụng GPIO3/1 (bootloader) → nạp code an toàn
  Serial2.begin(PI_UART_BAUD, SERIAL_8N1, 16, 17);  // RX=GPIO16, TX=GPIO17
  Serial2.setTimeout(10);
  Serial.println("UART2 ready (RX=GPIO3/RX0, TX=GPIO1/TX0) - waiting for Pi commands...");
}

// =============================================
// --- LOOP ---
// =============================================
void loop() {
  server.handleClient();
  handleUART();  // nhận lệnh từ Raspberry Pi qua Serial2
  computePID();
  updateServo();  // ← servo chỉ write tại đây, tách khỏi PID và web handler

  if ((targetSpeedLeft != 0 || targetSpeedRight != 0) && millis() - lastCommandTime > COMMAND_TIMEOUT_MS) {
    hardStop();
    Serial.println("[SAFE] Command timeout -> hard stop");
  }

  if (millis() - lastReport >= 2000) {
    lastReport = millis();
    Serial.printf("[ENC] L:%ld R:%ld | PWM L:%d R:%d | Servo:%d | Target:%d\n",
                  encLeftCount, encRightCount, pwmLeft, pwmRight,
                  currentServoAngle, targetSpeedLeft);
  }
}
