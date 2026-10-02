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
        self.started_at = time.time()

    def process_callback(self, indata, outdata, frames, time_info, status):
        if status:
            if self.debug:
                print(status)

        block = indata[:, 0].astype(np.float32)
        processed = process_block(block, self.sample_rate)
        outdata[:, 0] = processed[:frames] * self.volume

        if self.save_output is not None:
            self.output_frames.append(processed[:frames].copy())

    def run(self, duration_seconds=None):
        if duration_seconds is not None:
            stop_after = duration_seconds
            stream = sd.Stream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                device=(self.input_device, self.output_device),
                channels=(1, 1),
                dtype='float32',
                callback=self.process_callback,
            )
            with stream:
                sd.sleep(int(duration_seconds * 1000))
        else:
            stream = sd.Stream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                device=(self.input_device, self.output_device),
                channels=(1, 1),
                dtype='float32',
                callback=self.process_callback,
            )
            with stream:
                while True:
                    time.sleep(0.1)

        if self.save_output is not None:
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
    print("Use Python to list audio devices if needed.")

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


if __name__ == "__main__":
    main()
