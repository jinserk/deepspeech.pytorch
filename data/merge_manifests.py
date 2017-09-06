from __future__ import print_function

import argparse
import io
import os

import subprocess

from utils import update_progress

parser = argparse.ArgumentParser(description='Merges all manifest CSV files in specified folder.')
parser.add_argument('--merge_dir', default='manifests/', help='Path to all manifest files you want to merge')
parser.add_argument('--min_duration', default=-1, type=int,
                    help='Optionally prunes any samples shorter than the min duration (given in seconds, default off)')
parser.add_argument('--max_duration', default=-1, type=int,
                    help='Optionally prunes any samples longer than the max duration (given in seconds, default off)')
parser.add_argument('--output_path', default='merged_manifest.csv', help='Output path to merged manifest')

args = parser.parse_args()

files = []
for file in os.listdir(args.merge_dir):
    if file.endswith(".csv"):
        with open(os.path.join(args.merge_dir, file), 'r') as fh:
            files += fh.readlines()

prune_min = args.min_duration >= 0
prune_max = args.max_duration >= 0
if prune_min:
    print("Pruning files with minimum duration %d" % (args.min_duration))
if prune_max:
    print("Pruning files with  maximum duration of %d" % (args.max_duration))

new_files = []
size = len(files)
acc_size = 0
for x in range(size):
    file_path = files[x]
    file_path = file_path.split(',')
    output = subprocess.check_output(
        ['soxi -D \"%s\"' % file_path[0].strip()],
        shell=True
    )
    duration = float(output)
    if prune_min or prune_max:
        duration_fit = True
        if prune_min:
            if duration < args.min_duration:
                duration_fit = False
        if prune_max:
            if duration > args.max_duration:
                duration_fit = False
        if duration_fit:
            new_files.append((file_path[0], file_path[1], duration))
            acc_size += duration
    else:
        new_files.append((file_path[0], file_path[1], duration))
    update_progress(x / float(size))

def func(element):
    return element[2]

print("\nSorting files by length...")
new_files.sort(key=func)

print("Saving new manifest...")

with io.FileIO(args.output_path, 'w') as f:
    for utt in new_files:
        f.write(utt[0].strip().encode('utf-8'))
        f.write(','.encode('utf-8'))
        f.write(utt[1].strip().encode('utf-8'))
        f.write('\n'.encode('utf-8'))

print("total merged length: {} secs".format(acc_size))
