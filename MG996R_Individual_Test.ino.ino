
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

#define SERVO_FREQ 50

// Convert pulse width in microseconds to PCA9685 ticks
uint16_t usToTicks(int us) {
  return (uint32_t)us * 4096 / 20000;
}

void moveServo(int channel, int pulse) {
  pwm.setPWM(channel, 0, usToTicks(pulse));
}

void testServo(int channel) {
  Serial.print("Testing servo channel ");
  Serial.println(channel);

  // Start at neutral
  moveServo(channel, 1500);
  delay(1000);

  // Small movement in one direction
  moveServo(channel, 1450);
  delay(1000);

  // Small movement in the other direction
  moveServo(channel, 1550);
  delay(1000);

  // Return to neutral
  moveServo(channel, 1500);
  delay(1000);

  Serial.println("Test completed.");
}

void setup() {
  Serial.begin(115200);

  pwm.begin();
  pwm.setPWMFreq(SERVO_FREQ);

  delay(500);

  // Start all channels with no commanded pulse
  for (int i = 0; i < 16; i++) {
    pwm.setPWM(i, 0, 0);
  }

  Serial.println("MG996R Individual Servo Test");
  Serial.println("Enter 1 to test channel 0");
  Serial.println("Enter 2 to test channel 1");
  Serial.println("Enter 3 to test channel 2");
}

void loop() {
  if (Serial.available()) {
    char command = Serial.read();

    if (command == '1') {
      testServo(0);
    }
    else if (command == '2') {
      testServo(1);
    }
    else if (command == '3') {
      testServo(2);
    }
  }
}
