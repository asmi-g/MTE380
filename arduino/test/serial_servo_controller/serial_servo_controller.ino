
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// Initialize PWM driver
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// Original servo pulse settings
#define SERVOMIN 150
#define SERVOMAX 600

// Servo channels
#define SERVO1 0
#define SERVO2 1
#define SERVO3 2

#define BAUD_RATE 9600

// Convert angle to PWM pulse
int angleToPulse(int angle) {

    angle = constrain(angle, 0, 180);

    return map(angle, 0, 180, SERVOMIN, SERVOMAX);
}

// Set servo positions
void setServoAngle(int channel, int angle) {

    int pulse = angleToPulse(angle);

    pwm.setPWM(channel, 0, pulse);
}

void setup() {

    Serial.begin(BAUD_RATE);

    pwm.begin();
    pwm.setPWMFreq(50);

    delay(500);

    // Initialize all servos at 90 degrees
    setServoAngle(SERVO1, 90);
    setServoAngle(SERVO2, 90);
    setServoAngle(SERVO3, 90);

    Serial.println("Servo controller ready.");
}

void loop() {

    if (Serial.available()) {

        String input = Serial.readStringUntil('\n');

        int angle1, angle2, angle3;

        int result = sscanf(
            input.c_str(),
            "%d,%d,%d",
            &angle1,
            &angle2,
            &angle3
        );

        if (result == 3) {

            setServoAngle(SERVO1, angle1);
            setServoAngle(SERVO2, angle2);
            setServoAngle(SERVO3, angle3);

        }
    }
}
