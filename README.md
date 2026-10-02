# RESTEMER

RESTEMER is a Python desktop project for live audio remixing. It listens to the user's default microphone/soundcard input, detects drum and percussion energy, adds synthetic percussion layers and processing, and plays the result back through the output device.

This is a first working version intended for experimentation and rapid iteration. It is not a full professional stem-separation engine, but it gives you a real-time audio remix pipeline you can build on.

## Features

- live input from the user's soundcard / microphone
- simple drum/percussion onset detection
- generated kick/snare/hi-hat layer based on detected beats
- output back through the default speaker or output device
- saves a WAV file option for recordings

## Install

1. Clone the repo
2. Create a virtual environment
3. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python -m restemer.app --duration 20 --save-output demo.wav
```

Optional flags:

- `--input-device` select a microphone/input device index
- `--output-device` select an output device index
- `--sample-rate 44100`
- `--block-size 2048`
- `--volume 1.0`
- `--debug`

## Notes

- You may need to install PortAudio on your system for `sounddevice` to work.
- On Linux: `sudo apt install portaudio19-dev`
- On macOS: `brew install portaudio`
- On Windows: the package usually works with the bundled PortAudio binaries.

## Current algorithm

The current implementation is intentionally simple:

1. Capture audio frames from the input device
2. Convert to mono
3. Compute a short-time energy envelope and onset signal
4. Detect likely drum hits using a threshold
5. Add synthetic kick/snare/hi-hat layers aligned with those hits
6. Mix the processed signal with the original audio
7. Output to the soundcard or save to disk

This is a solid base for venturing into better stem separation later with libraries like Demucs or Spleeter.
