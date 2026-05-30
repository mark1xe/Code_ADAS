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
// =============================================
#define PI_UART_BAUD 115200

// =============================================
// --- SERVO LÁI ---
// =============================================
const int SERVO_PIN = 13;
const int SERVO_CENTER = 68;  // thẳng
const int SERVO_LEFT = 40;    // rẽ trái tối đa
const int SERVO_RIGHT = 120;  // rẽ phải tối đa
Servo steerServo;
int currentServoAngle = SERVO_CENTER;
int desiredServoAngle = SERVO_CENTER;

// =============================================
// --- ENCODER ---
// =============================================
volatile long encLeftCount = 0;
volatile long encRightCount = 0;

void IRAM_ATTR isrLeft() {
  bool A = digitalRead(ENC_L_A);
  bool B = digitalRead(ENC_L_B);
  if (A == B) encLeftCount++; else encLeftCount--;
}
void IRAM_ATTR isrLeftB() {
  bool A = digitalRead(ENC_L_A);
  bool B = digitalRead(ENC_L_B);
  if (A != B) encLeftCount++; else encLeftCount--;
}
void IRAM_ATTR isrRight() {
  bool A = digitalRead(ENC_R_A);
  bool B = digitalRead(ENC_R_B);
  if (A == B) encRightCount++; else encRightCount--;
}
void IRAM_ATTR isrRightB() {
  bool A = digitalRead(ENC_R_A);
  bool B = digitalRead(ENC_R_B);
  if (A != B) encRightCount++; else encRightCount--;
}

// =============================================
// --- PID + FEEDFORWARD ---
// =============================================
float Kp = 1.2;
float Ki = 0.08;           
const int FF_BASE = 100;
const int MAX_SPEED = 30;

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
const unsigned long COMMAND_TIMEOUT_MS = 10000;

// =============================================
// --- ĐIỀU KHIỂN MOTOR ---
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

  // Tách biệt hướng quay 2 bánh để an toàn khi vô cua gắt
  int dirL = (targetSpeedLeft > 0) ? 1 : (targetSpeedLeft < 0 ? -1 : 0);
  int dirR = (targetSpeedRight > 0) ? 1 : (targetSpeedRight < 0 ? -1 : 0);
  
  bool encDead = (deltaL == 0 && deltaR == 0);
  
  if (encDead) {
    int rawPwmLeft = map(abs(targetSpeedLeft), 0, MAX_SPEED, FF_BASE, 255);
    int rawPwmRight = map(abs(targetSpeedRight), 0, MAX_SPEED, FF_BASE, 255);
    
    // An toàn tuyệt đối: Target = 0 thì ngắt điện PWM
    if (targetSpeedLeft == 0) rawPwmLeft = 0;
    if (targetSpeedRight == 0) rawPwmRight = 0;
    
    pwmLeft  = dirL * constrain(rawPwmLeft, 0, 255);
    pwmRight = dirR * constrain(rawPwmRight, 0, 255);
  } else {
    float errL = targetSpeedLeft  - deltaL;
    float errR = targetSpeedRight - deltaR;
    integralLeft  = constrain(integralLeft  + errL, -200, 200);
    integralRight = constrain(integralRight + errR, -200, 200);

    // Dynamic Feedforward kết hợp với FF_BASE mạnh mẽ
    int basePwmLeft = map(abs(targetSpeedLeft), 0, MAX_SPEED, FF_BASE, 255);
    int basePwmRight = map(abs(targetSpeedRight), 0, MAX_SPEED, FF_BASE, 255);
    
    if (targetSpeedLeft == 0) basePwmLeft = 0;
    if (targetSpeedRight == 0) basePwmRight = 0;

    pwmLeft  = constrain(dirL * basePwmLeft + (int)(Kp * errL + Ki * integralLeft),  -255, 255);
    pwmRight = constrain(dirR * basePwmRight + (int)(Kp * errR + Ki * integralRight), -255, 255);
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
// --- HÀM ĐIỀU KHIỂN XE TÁCH BIỆT TRÁI/PHẢI ---
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

void carStop() {
  markCommandSeen();
  hardStop();
}

void steerCenter() { desiredServoAngle = SERVO_CENTER; markCommandSeen(); }
void steerLeft() { desiredServoAngle = SERVO_LEFT; markCommandSeen(); }
void steerRight() { desiredServoAngle = SERVO_RIGHT; markCommandSeen(); }

// =============================================
// --- UART: nhận lệnh từ Raspberry Pi ---
// =============================================
void handleUART() {
  if (!Serial2.available()) return;
  String line = Serial2.readStringUntil('\n');
  line.trim();
  if (line.length() < 5) return;

  int c1 = line.indexOf(',');
  int c2 = line.lastIndexOf(',');
  if (c1 < 0 || c2 == c1) return;

  char drive = (char)line.charAt(0);
  int steerVal = line.substring(c1 + 1, c2).toInt();  
  int spd = line.substring(c2 + 1).toInt();
  
  // KIỂM TRA ĐẶC BIỆT: Nếu Pi muốn dừng (Tốc độ = 0)
  if (spd == 0 && (drive == 'F' || drive == 'B')) {
      drive = 'S';
  }

  steerVal = constrain(steerVal, -100, 100);
  int targetAngle;
  if (steerVal <= 0) {
    targetAngle = SERVO_CENTER + steerVal * (SERVO_CENTER - SERVO_LEFT) / 100;
  } else {
    targetAngle = SERVO_CENTER + steerVal * (SERVO_RIGHT - SERVO_CENTER) / 100;
  }
  desiredServoAngle = constrain(targetAngle, SERVO_LEFT, SERVO_RIGHT);
  markCommandSeen();

  // --- Vi sai điện tử (Electronic Differential) ---
  float diffFactor = 0.55; 
  int spdL = spd;
  int spdR = spd;

  if (steerVal < 0) {
      spdL = spd - (spd * (abs(steerVal) / 100.0) * diffFactor);
  } else if (steerVal > 0) {
      spdR = spd - (spd * (steerVal / 100.0) * diffFactor);
  }

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
  int s = constrain(spd, 1, MAX_SPEED);
  carForward(s, s);
  server.send(200, "text/plain", "OK");
}
void handleBackward() {
  int spd = server.hasArg("spd") ? server.arg("spd").toInt() : 15;
  int s = constrain(spd, 1, MAX_SPEED);
  carBackward(s, s);
  server.send(200, "text/plain", "OK");
}
void handleKeepalive() { markCommandSeen(); server.send(200, "text/plain", "OK"); }
void handleStop() { carStop(); server.send(200, "text/plain", "OK"); }
void handleSteerLeft() { steerLeft(); server.send(200, "text/plain", "OK"); }
void handleSteerRight() { steerRight(); server.send(200, "text/plain", "OK"); }
void handleSteerCenter() { steerCenter(); server.send(200, "text/plain", "OK"); }
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
// --- TRANG WEB HTML ---
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
    timer=setInterval(()=>{ if(driveActive) cmd('/keepalive'); },400);
  }
  function release(){
    if(!driveActive) return;
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

  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);

  ledcAttach(ENA, 5000, 8);
  ledcAttach(ENB, 5000, 8);
  applyMotorPWM(0, 0);

  pinMode(ENC_L_A, INPUT_PULLUP); pinMode(ENC_L_B, INPUT_PULLUP);
  pinMode(ENC_R_A, INPUT_PULLUP); pinMode(ENC_R_B, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENC_L_A), isrLeft, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_L_B), isrLeftB, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_R_A), isrRight, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_R_B), isrRightB, CHANGE);

  ESP32PWM::allocateTimer(3);
  steerServo.setPeriodHertz(50);
  steerServo.attach(SERVO_PIN);
  delay(200);
  steerServo.write(SERVO_CENTER);
  delay(300);
  
  lastCommandTime = millis();

  Serial.printf("Ket noi: %s\n", ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.printf("\nIP: http://%s\n", WiFi.localIP().toString().c_str());

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

  // UART2 với Timeout được giới hạn an toàn 20ms
  Serial2.begin(PI_UART_BAUD, SERIAL_8N1, 16, 17);
  Serial2.setTimeout(20);
  Serial.println("UART2 ready (RX=GPIO3/RX0, TX=GPIO1/TX0) - waiting for Pi commands...");
}

// =============================================
// --- LOOP ---
// =============================================
void loop() {
  server.handleClient();
  handleUART();
  computePID();
  updateServo();

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