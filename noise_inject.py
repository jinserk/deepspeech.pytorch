import argparse

import torch
import torchaudio

from data.data_loader import load_audio, NoiseInjection

parser = argparse.ArgumentParser()
parser.add_argument('--noise_path', default='data/noise', help='The noise file to mix in')
parser.add_argument('--input_file', default='input.wav', help='The input audio to inject noise into')
parser.add_argument('--output_file', default='output.wav', help='The noise file to mix in')
parser.add_argument('--noise_file', default='noise.wav', help='The noise file to mix in')
parser.add_argument('--noise_min', default=0.0,
                    help='Minimum noise level to sample from. (0.0-1.0)', type=float)
parser.add_argument('--noise_max', default=0.5,
                    help='Maximum noise levels to sample from. (0.0-1.0)', type=float)
args = parser.parse_args()

data, sample_rate = load_audio(args.input_file)
noise_injector = NoiseInjection(args.noise_path, sample_rate, (args.noise_min, args.noise_max))
mixed_data, noise_data = noise_injector.inject_noise(data)
mixed_data = torch.FloatTensor(mixed_data).unsqueeze(1)  # Add channels dim
noise_data = torch.FloatTensor(noise_data).unsqueeze(1)

torchaudio.save(args.output_file, mixed_data, sample_rate)
torchaudio.save(args.noise_file, noise_data, sample_rate)
print('Saved mixed file to {}, noise file to {}'.format(args.output_file, args.noise_file))
