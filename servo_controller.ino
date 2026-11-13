#include <Servo.h>

// Servo configuration
Servo beamServo;
const int servoPin = 9;
const int MIN_ANGLE = 0;
const int MAX_ANGLE = 30;
const int NEUTRAL_ANGLE = 15;

// Communication variables
int targetAngle = NEUTRAL_ANGLE;
bool newCommand = false;

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  
  // Attach servo
  beamServo.attach(servoPin);
  
  // Move to neutral position
  beamServo.write(NEUTRAL_ANGLE);
  
  // Initialize variables
  targetAngle = NEUTRAL_ANGLE;
  
  // Optional: Send ready message
  Serial.println("Arduino servo controller ready");
}

void loop() {
  // Check for incoming serial data
  if (Serial.available() > 0) {
    // Read the incoming byte
    int receivedAngle = Serial.read();
    
    // Validate angle range
    if (receivedAngle >= MIN_ANGLE && receivedAngle <= MAX_ANGLE) {
      targetAngle = receivedAngle;
      newCommand = true;
    }
  }
  
  // Update servo if new command received
  if (newCommand) {
    beamServo.write(targetAngle);
    newCommand = false;
    
    // Optional: Echo back the angle for debugging
    // Serial.print("Angle set to: ");
    // Serial.println(targetAngle);
  }
  
  // Small delay for stability
  delay(10);
}