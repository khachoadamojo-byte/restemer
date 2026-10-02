import numpy as np


def mono_to_float32(audio):
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)
    if np.max(np.abs(audio)) > 1.0:
        audio = audio / np.max(np.abs(audio))
    return audio


def detect_onsets(signal, sample_rate, frame_size=2048, hop_size=512, threshold=0.12):
    if len(signal) < frame_size:
        return np.array([], dtype=int)

    energy = []
    for i in range(0, len(signal) - frame_size + 1, hop_size):
        frame = signal[i : i + frame_size]
        rms = np.sqrt(np.mean(np.square(frame)))
        energy.append(rms)

    if len(energy) < 2:
        return np.array([], dtype=int)

    energy = np.asarray(energy, dtype=np.float32)
    diff = np.diff(energy)
    onset_mask = diff > threshold
    onset_indices = np.where(onset_mask)[0] + 1
    onset_times = onset_indices * hop_size / sample_rate
    return onset_times


def synth_kick(sample_rate, duration=0.12, freq=55.0, volume=0.55):
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    env = np.exp(-8.0 * t / duration)
    tone = np.sin(2 * np.pi * freq * t)
    waveform = env * tone * volume
    return waveform.astype(np.float32)


def synth_snare(sample_rate, duration=0.12, volume=0.35):
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    noise = np.random.default_rng(0).standard_normal(len(t))
    env = np.exp(-18.0 * t / duration)
    waveform = env * noise * volume
    return waveform.astype(np.float32)


def synth_hat(sample_rate, duration=0.06, volume=0.18):
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    noise = np.random.default_rng(1).standard_normal(len(t))
    env = np.exp(-30.0 * t / duration)
    waveform = env * noise * volume
    return waveform.astype(np.float32)


def build_extra_percussion(signal, sample_rate, onset_times, mode="beat"):
    extra = np.zeros_like(signal, dtype=np.float32)
    if onset_times.size == 0:
        return extra

    for onset in onset_times:
        start_sample = int(onset * sample_rate)
        if start_sample < 0 or start_sample >= len(signal):
            continue

        if mode == "beat":
            kick = synth_kick(sample_rate)
            snare = synth_snare(sample_rate)
            hat = synth_hat(sample_rate)
            kick_end = min(len(signal) - start_sample, len(kick))
            snare_end = min(len(signal) - start_sample, len(snare))
            hat_end = min(len(signal) - start_sample, len(hat))
            extra[start_sample : start_sample + kick_end] += kick[:kick_end]
            extra[start_sample : start_sample + snare_end] += snare[:snare_end]
            if start_sample % 2 == 0:
                extra[start_sample : start_sample + hat_end] += hat[:hat_end]
        else:
            kick = synth_kick(sample_rate, duration=0.10, freq=65.0, volume=0.4)
            kick_end = min(len(signal) - start_sample, len(kick))
            extra[start_sample : start_sample + kick_end] += kick[:kick_end]

    return extra


def process_block(signal, sample_rate):
    signal = mono_to_float32(signal)
    onsets = detect_onsets(signal, sample_rate, threshold=0.1)
    extra = build_extra_percussion(signal, sample_rate, onsets, mode="beat")
    mixed = signal + (0.55 * extra)
    mixed = np.clip(mixed, -1.0, 1.0)
    return mixed
