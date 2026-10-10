# Gesture Subway Runner

Gesture Subway Runner is a browser game controlled by webcam hand gestures. A Python server uses OpenCV and MediaPipe to track one hand, then sends the detected hand position and gestures to the game over a local WebSocket connection.

## Requirements

- Windows, macOS, or Linux
- Python 3.10 or newer
- A working webcam
- A modern browser such as Chrome, Edge, or Firefox
- Good lighting so the camera can see your hand

The first dependency installation may take a few minutes because MediaPipe and OpenCV are included.

## Start the project on Windows

1. Download or clone this project and open its folder.
2. Install Python 3.10 or newer from [python.org](https://www.python.org/downloads/). During installation, enable **Add Python to PATH**.
3. Connect a webcam and close other applications that may be using it.
4. Double-click `start_game.bat`.
5. On the first run, the launcher installs the packages in `requirements.txt` automatically.
6. Allow camera access if Windows or your browser asks for permission.
7. The game opens automatically. Keep the command window open while playing; it runs the gesture server.
8. To stop the project, return to the command window and press `Ctrl+C`.

## Manual setup

Open PowerShell or a terminal in the project directory, then run:

```powershell
python -m venv .venv
```

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run the following once in a PowerShell window opened for your user account, then retry activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Install the dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start the server and game:

```powershell
python gesture_server.py
```

The server opens `gesture game.html` automatically and listens on `127.0.0.1:8765`. If it does not open, open `gesture game.html` manually in a browser while the Python server is still running.

On macOS or Linux, the equivalent commands are:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 gesture_server.py
```

## How to play

- Move your hand left or right to change lanes.
- Swipe upward to jump.
- Swipe downward to duck.
- Hold an open palm to start or restart.
- Hold a fist to pause or resume.
- Collect coins and avoid trains and barriers.

Keyboard controls are available as a fallback:

- `Left Arrow` / `Right Arrow`: change lanes
- `Up Arrow`: jump
- `Down Arrow`: duck
- `Space`: start, pause, or resume
- `C`: show or hide the camera preview

## Optional server flags

```text
--camera INDEX       Select a camera (default: 0)
--debug              Show an annotated OpenCV camera window
--no-mirror          Disable the mirrored camera image
--no-preview         Do not send the camera preview to the browser
--no-browser         Do not open the game automatically
--port PORT          Change the WebSocket port
```

Examples:

```powershell
python gesture_server.py --camera 1
python gesture_server.py --no-preview --no-browser
python gesture_server.py --debug
```

The game currently connects to port `8765`, so changing `--port` requires updating the WebSocket port in `gesture game.html` as well.

## Troubleshooting

### `Python not found on PATH`

Install Python, enable **Add Python to PATH**, reopen the terminal, and verify:

```powershell
python --version
```

### Dependencies fail to install

Make sure Python is 3.10 or newer, activate the virtual environment, and run:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### The camera cannot be opened

Close Zoom, Teams, camera apps, and other programs using the webcam. Check the operating system camera privacy settings, then try `--camera 1` if you have more than one camera.

### The game says the server is offline

Keep `python gesture_server.py` or `start_game.bat` running. The browser game expects the local WebSocket server at `127.0.0.1:8765`.

### The image is too dark or gestures are unreliable

Face a light source, keep your hand within the camera frame, use a plain background, and keep the webcam unobstructed. You can run `python gesture_server.py --debug` to inspect tracking.

### The browser does not open automatically

Start the server, then open `gesture game.html` from this folder manually. Do not close the server window while playing.

## Project files

- `gesture_server.py`: webcam capture, MediaPipe hand tracking, gesture detection, and WebSocket server
- `gesture game.html`: browser game and keyboard fallback controls
- `requirements.txt`: Python dependencies
- `start_game.bat`: Windows launcher that checks Python and installs dependencies when needed
