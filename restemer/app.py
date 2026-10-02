import argparse
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

from .core import process_block


class RestemerStream:
    def __init__(
        self,
        sample_rate=44100,
        block_size=2048,
        input_device=None,
        output_device=None,
        volume=1.0,
        save_output=None,
        debug=False,
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.input_device = input_device
        self.output_device = output_device
        self.volume = volume
        self.save_output = save_output
        self.debug = debug
        self.output_frames = []

    def _safe_device(self, device):
        if device is not None:
            return device
        return sd.default.device

    def process_callback(self, indata, outdata, frames, time_info, status):
        if status:
            if self.debug:
                print(f"Audio status: {status}")

        if indata is None or indata.size == 0:
            outdata.fill(0)
            return

        try:
            block = np.asarray(indata)
            if block.ndim == 2 and block.shape[1] > 0:
                block = block[:, 0]
            if block.size == 0:
                outdata.fill(0)
                return

            processed = process_block(block, self.sample_rate)
            if processed.size == 0:
                outdata.fill(0)
                return

            length = min(frames, processed.shape[0])
            outdata[:length, 0] = processed[:length] * self.volume
            outdata[length:, 0] = 0.0

            if self.save_output is not None:
                self.output_frames.append(processed[:length].copy())
        except Exception as exc:
            if self.debug:
                print(f"Processing error: {exc}")
            outdata.fill(0)

    def run(self, duration_seconds=None):
        input_device = self._safe_device(self.input_device)
        output_device = self._safe_device(self.output_device)

        try:
            stream = sd.Stream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                device=(input_device, output_device),
                channels=(1, 1),
                dtype='float32',
                callback=self.process_callback,
            )

            with stream:
                if duration_seconds is not None:
                    sd.sleep(int(duration_seconds * 1000))
                else:
                    while True:
                        time.sleep(0.1)
        except Exception as exc:
            raise RuntimeError(f"Failed to open audio stream: {exc}") from exc

        if self.save_output is not None and self.output_frames:
            rendered = np.concatenate(self.output_frames)
            sf.write(self.save_output, rendered, self.sample_rate)
            print(f"Saved output to {self.save_output}")


def build_parser():
    parser = argparse.ArgumentParser(description="RESTEMER: real-time drum remixing from live audio input")
    parser.add_argument("--duration", type=float, default=10.0, help="How long to run the stream in seconds")
    parser.add_argument("--input-device", type=int, default=None, help="Input device index")
    parser.add_argument("--output-device", type=int, default=None, help="Output device index")
    parser.add_argument("--sample-rate", type=int, default=44100, help="Sample rate")
    parser.add_argument("--block-size", type=int, default=2048, help="Audio block size")
    parser.add_argument("--volume", type=float, default=1.0, help="Output volume")
    parser.add_argument("--save-output", type=str, default=None, help="Optional output WAV file path")
    parser.add_argument("--debug", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    print("RESTEMER starting...")
    print("If you hear nothing, check your input and output devices.")
    print("Use Python to list devices if needed.")

    try:
        stream = RestemerStream(
            sample_rate=args.sample_rate,
            block_size=args.block_size,
            input_device=args.input_device,
            output_device=args.output_device,
            volume=args.volume,
            save_output=args.save_output,
            debug=args.debug,
        )
        stream.run(duration_seconds=args.duration)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        print("Try listing devices with:")
        print("python -c \"import sounddevice as sd; print(sd.query_devices())\"")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
