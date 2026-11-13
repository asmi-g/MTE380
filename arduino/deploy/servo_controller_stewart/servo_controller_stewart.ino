#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Servo configuration
Adafruit_PWMServoDriver pwm(0x40);
const int servoChannels[3] = {0, 1, 2};
const float SERVO_FREQUENCY = 50.0;

// Example calibration values: replace with measured values for each servo
const bool CALIBRATION_READY = true; //needs to be true
const float MIN_ANGLE[3] = {80.0, 80.0, 80.0};
const float MAX_ANGLE[3] = {100.0, 100.0, 100.0};
const float NEUTRAL_ANGLE[3] = {90.0, 90.0, 90.0};
const float NEUTRAL_US[3] = {1500.0, 1500.0, 1500.0};
const float US_PER_DEGREE[3] = {10.0, 10.0, 10.0};  // Negative for reversed direction
const float MIN_US[3] = {1400.0, 1400.0, 1400.0};
const float MAX_US[3] = {1600.0, 1600.0, 1600.0};

// Communication variables
const unsigned long BAUD_RATE = 115200;
const unsigned long COMMAND_TIMEOUT_MS = 500;
float targetAngle[3];
bool newCommand = false;
bool neutralCommand = false;
bool enabled = false;
bool atLevel = true;
unsigned long lastCommandTime = 0;
char commandBuffer[80];
int commandLength = 0;
bool invalidCommand = false;

bool readCommand(char *command);
bool updateServos(const float angles[3]);

void setup() {
  // Initialize serial communication
  Serial.begin(BAUD_RATE);

  // Initialize servo shield
  if (!pwm.begin()) {
    Serial.println("ERROR: PCA9685 not found");
    return;
  }
  pwm.setPWMFreq(SERVO_FREQUENCY);
  for (int i = 0; i < 3; i++) {
    pwm.setPWM(servoChannels[i], 0, 4096);
  }

  // Check servo calibration
  if (!CALIBRATION_READY) {
    Serial.println("SET CALIBRATION: outputs disabled");
    return;
  }
  for (int i = 0; i < 3; i++) {
    if (!isfinite(US_PER_DEGREE[i]) || US_PER_DEGREE[i] == 0 ||
        !isfinite(MIN_ANGLE[i]) || !isfinite(MAX_ANGLE[i]) ||
        !isfinite(MIN_US[i]) || !isfinite(MAX_US[i]) ||
        MIN_ANGLE[i] > MAX_ANGLE[i] || MIN_US[i] > MAX_US[i] ||
        MIN_US[i] <= 0 || MAX_US[i] >= 1000000.0 / SERVO_FREQUENCY) {
      Serial.println("ERROR: invalid calibration");
      return;
    }
  }

  // Move to neutral position
  enabled = updateServos(NEUTRAL_ANGLE);

  // Initialize variables
  for (int i = 0; i < 3; i++) {
    targetAngle[i] = NEUTRAL_ANGLE[i];
  }
  lastCommandTime = millis();

  // Send ready message
  Serial.println(enabled ? "READY" : "ERROR: neutral outside limits");
}

void loop() {
  // Check for incoming serial data
  for (int count = 0; count < 64 && Serial.available() > 0; count++) {
    char receivedCharacter = Serial.read();

    // Read a complete command ending with a newline
    if (receivedCharacter == '\n') {
      if (!invalidCommand && enabled) {
        commandBuffer[commandLength] = '\0';
        newCommand = readCommand(commandBuffer);
      }
      commandLength = 0;
      invalidCommand = false;
      break;
    }

    // Store incoming characters
    if (!invalidCommand && receivedCharacter != '\r') {
      if (receivedCharacter < 32 || receivedCharacter > 126 || commandLength >= 79) {
        invalidCommand = true;
      } else {
        commandBuffer[commandLength++] = receivedCharacter;
      }
    }
  }

  // Update servos if new command received
  if (newCommand) {
    if (updateServos(targetAngle)) {
      lastCommandTime = millis();
      atLevel = neutralCommand;
    }
    newCommand = false;
  }

  // Return to neutral if commands stop
  if (enabled && !atLevel && millis() - lastCommandTime >= COMMAND_TIMEOUT_MS) {
    updateServos(NEUTRAL_ANGLE);
    atLevel = true;
  }
}

// Read A,angle1,angle2,angle3 or N for neutral
bool readCommand(char *command) {
  neutralCommand = strcmp(command, "N") == 0;
  if (neutralCommand) {
    for (int i = 0; i < 3; i++) {
      targetAngle[i] = NEUTRAL_ANGLE[i];
    }
    return true;
  }
  if (strncmp(command, "A,", 2) != 0) {
    return false;
  }

  // Read three absolute joint angles in degrees
  float receivedAngle[3];
  char *nextValue = command + 2;
  for (int i = 0; i < 3; i++) {
    char *end;
    receivedAngle[i] = strtod(nextValue, &end);
    if (end == nextValue || !isfinite(receivedAngle[i])) {
      return false;
    }
    if (i < 2) {
      if (*end != ',') return false;
      nextValue = end + 1;
    } else if (*end != '\0') {
      return false;
    }
  }

  // Store target angles
  for (int i = 0; i < 3; i++) {
    targetAngle[i] = receivedAngle[i];
  }
  return true;
}

bool updateServos(const float angles[3]) {
  float pulseUs[3];

  // Validate all angles and pulse widths before moving any servo
  for (int i = 0; i < 3; i++) {
    if (!isfinite(angles[i]) || angles[i] < MIN_ANGLE[i] || angles[i] > MAX_ANGLE[i]) {
      return false;
    }
    pulseUs[i] = NEUTRAL_US[i] + (angles[i] - NEUTRAL_ANGLE[i]) * US_PER_DEGREE[i];
    if (!isfinite(pulseUs[i]) || pulseUs[i] < MIN_US[i] || pulseUs[i] > MAX_US[i]) {
      return false;
    }
  }

  // Update servo positions
  for (int i = 0; i < 3; i++) {
    pwm.writeMicroseconds(servoChannels[i], (uint16_t)lroundf(pulseUs[i]));
  }
  return true;
}