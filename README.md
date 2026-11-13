# MTE380 Stewart Platform Code

To deploy the code,

1. Change line 144 in `pid-controls/deploy/stewart_controller_3servo.py` to wherever your arduino com port is:
```
    'port': 'COM5', 
```

2. Flash the arduino from the IDE, with the file located at `arduino/deploy/servo_controller_stewart/servo_controller_stewart.ino`

3. Run the python file within your VS Code IDE, with the file located at `pid-controls/deploy/stewart_controller_3servo.py`