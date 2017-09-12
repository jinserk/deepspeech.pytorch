import os
import wave
import librosa
from data_loader import load_audio, NoiseInjection

rfilename = "../test/conan1-8k.wav"
wfilename = "../test/conan1-8k-contaminated.wav"
nfilename = "../test/conan1-8k-noise.wav"

noise_dir = "./noise"
sample_rate = 8000
noise_levels = (0, 0.1)

noise_injector = NoiseInjection(noise_dir, sample_rate, noise_levels)

wav, _ = load_audio(rfilename)
contaminated, noise = noise_injector.inject_noise(wav)
librosa.output.write_wav(wfilename, contaminated, sample_rate)
librosa.output.write_wav(nfilename, noise, sample_rate)

