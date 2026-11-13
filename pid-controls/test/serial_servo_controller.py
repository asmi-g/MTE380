
import serial
import time

# Serial configuration
SERIAL_PORT = "COM5"
BAUD_RATE = 9600

# Sweep configuration
MIN_ANGLE = 0
MAX_ANGLE = 180
STEP_DELAY = 0.015  # 15 ms

def send_angles(arduino, angle):

    command = f"{angle},{angle},{angle}\n"

    arduino.write(command.encode("utf-8"))


def main():

    try:
        arduino = serial.Serial(
            SERIAL_PORT,
            BAUD_RATE,
            timeout=1
        )

        time.sleep(2)

        print("Arduino connected.")

        # Zero all three servos
        send_angles(arduino, 90)
        time.sleep(2)

        print("Starting sweep...")

        while True:

            # Sweep from 0 to 180
            for angle in range(MIN_ANGLE, MAX_ANGLE + 1):

                send_angles(arduino, angle)
                time.sleep(STEP_DELAY)

            # Sweep from 180 back to 0
            for angle in range(MAX_ANGLE, MIN_ANGLE - 1, -1):

                send_angles(arduino, angle)
                time.sleep(STEP_DELAY)

    except KeyboardInterrupt:
        print("Sweep stopped.")

    finally:
        if 'arduino' in locals() and arduino.is_open:
            arduino.close()


if __name__ == "__main__":
    main()
