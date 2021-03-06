from __future__ import print_function

import fnmatch
import io
import os
import glob
from tqdm import tqdm

from data_loader import get_audio_length
import subprocess


def create_manifest(data_path, out_path, tag, ordered=True):
    manifest_path = os.path.join(out_path, '%s_manifest.csv' % tag)
    file_paths = []
    #wav_files = [os.path.join(dirpath, f)
    #             for dirpath, dirnames, files in os.walk(data_path)
    #             for f in fnmatch.filter(files, '*.wav')]
    print(data_path)
    wav_files = [filename for filename in glob.iglob('{}/**/*.wav'.format(data_path), recursive=True)]

    for file_path in tqdm(wav_files, total=len(wav_files)):
        file_paths.append(file_path.strip())
    print('\n')
    if ordered:
        _order_files(file_paths)
    with io.FileIO(manifest_path, "w") as file:
        for wav_path in tqdm(file_paths, total=len(file_paths)):
            duration = get_audio_length(wav_path)
            transcript_path = wav_path.replace('/wav/', '/txt/').replace('.wav', '.txt')
            sample = os.path.abspath(wav_path) + ',' + '{}'.format(duration) + ',' + os.path.abspath(transcript_path) + '\n'
            file.write(sample.encode('utf-8'))
    print('\n')


def _order_files(file_paths):
    print("Sorting files by length...")

    def func(element):
        output = subprocess.check_output(
            ['soxi -D \"%s\"' % element.strip()],
            shell=True
        )
        return float(output)

    file_paths.sort(key=func)
