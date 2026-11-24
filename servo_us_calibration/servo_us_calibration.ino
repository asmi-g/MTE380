// MG995 + PCA9685. Serial Monitor: 115200 baud, Newline.
// Measure horn angles with a protractor using one fixed zero and direction.
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <stdlib.h>
#include <math.h>

Adafruit_PWMServoDriver pwm(0x40);
const int neutralUs[] = {1000, 1000, 1200};
const int stepUs = 10;
const int travelUs = 50;  // Local calibration window on either side of neutral
int channel = 0, pulse = 1100, firstPulse = 0;
float firstAngle = 0;
bool haveFirst = true, ready = true;

void moveTo(int us) {
  pulse = constrain(us, neutralUs[channel]-travelUs, neutralUs[channel]+travelUs);
  pwm.writeMicroseconds(channel, pulse);
  Serial.print("CH "); Serial.print(channel);
  Serial.print("  commanded pulse (us): "); Serial.println(pulse);
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(100);
  if (!pwm.begin()) { Serial.println("PCA9685 missing; check wiring."); return; }
  pwm.setPWMFreq(50);
  for (int i=0; i<3; i++) pwm.writeMicroseconds(i, neutralUs[i]);
  ready = true;
  Serial.println("0/1/2 select servo | + or - step 10 us | n neutral | a ANGLE then b ANGLE");
  moveTo(neutralUs[channel]);
}

void loop() {
  if (!ready || !Serial.available()) return;
  String cmd = Serial.readStringUntil('\n');
  cmd.trim();
  if (!cmd.length()) return;
  if (cmd.length()==1 && cmd[0]>='0' && cmd[0]<='2') {
    moveTo(neutralUs[channel]);
    channel = cmd[0]-'0';
    haveFirst = false;
    moveTo(neutralUs[channel]);
  } else if (cmd=="+") moveTo(pulse+stepUs);
  else if (cmd=="-") moveTo(pulse-stepUs);
  else if (cmd=="n") { haveFirst=false; moveTo(neutralUs[channel]); }
  else if (cmd.startsWith("a ") || cmd.startsWith("b ")) {
    const char *start = cmd.c_str()+2;
    char *end;
    float angle = strtod(start, &end);
    if (end==start || *end || !isfinite(angle)) { Serial.println("Enter an angle, e.g. a 42.5"); return; }
    if (cmd[0]=='a') {
      firstAngle=angle; firstPulse=pulse; haveFirst=true;
      Serial.println("First point saved. Step to another pulse, then enter b ANGLE.");
    } else if (!haveFirst || firstPulse==pulse || fabs(angle-firstAngle)<0.01) {
      Serial.println("Need two distinct pulses and measured angles. Save first point with a ANGLE.");
    } else {
      Serial.print("CH "); Serial.print(channel);
      Serial.print("  US_PER_DEGREE = ");
      Serial.println(fabs((pulse-firstPulse)/(angle-firstAngle)), 2);
    }
  } else Serial.println("Use 0/1/2, +, -, n, a ANGLE, or b ANGLE.");
}
