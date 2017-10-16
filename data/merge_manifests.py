from __future__ import print_function

import argparse
import io
import os
import glob

import subprocess
from tqdm import trange

parser = argparse.ArgumentParser(description='Merges all manifest CSV files in specified folder.')
parser.add_argument('--merge_dir', default='manifests/', help='Path to all manifest files you want to merge')
parser.add_argument('--min_duration', default=-1, type=float,
                    help='Optionally prunes any samples shorter than the min duration (given in seconds, default off)')
parser.add_argument('--max_duration', default=15, type=float,
                    help='Optionally prunes any samples longer than the max duration (given in seconds, default 15 secs)')
parser.add_argument('--output_path', default='merged_manifest.csv', help='Output path to merged manifest')

args = parser.parse_args()

files = []
for manifest in glob.iglob(os.path.join(args.merge_dir, "**/*.csv"), recursive=True):
    try:
        with open(manifest, 'r') as fh:
            files += fh.readlines()
    except:
        continue

prune_min = args.min_duration >= 0
prune_max = args.max_duration >= 0
if prune_min:
    print("Pruning files with minimum duration of %.3f secs" % (args.min_duration))
if prune_max:
    print("Pruning files with maximum duration of %.3f secs" % (args.max_duration))

new_files = []
size = len(files)
acc_size = 0.
    file_path = files[x].strip()
    file_path = file_path.split(',')
    #output = subprocess.check_output(
    #    ['soxi -D \"%s\"' % file_path[0].strip()],
    #    shell=True
    #)
    #duration = float(output)
    duration = float(file_path[1])
    if prune_min or prune_max:
        duration_fit = True
        if prune_min:
            if duration < args.min_duration:
                duration_fit = False
        if prune_max:
            if duration > args.max_duration:
                duration_fit = False
        if duration_fit:
            new_files.append((file_path[0], duration, file_path[2]))
            acc_size += duration
    else:
        new_files.append((file_path[0], duration, file_path[2]))
        acc_size += duration

def func(element):
    return element[1]

print("\nSorting files by length...")
new_files.sort(key=func)

print("Saving new manifest...")

with io.FileIO(args.output_path, 'w') as f:
    for utt in new_files:
        f.write(utt[0].strip().encode('utf-8'))
        f.write(','.encode('utf-8'))
        f.write('{}'.format(utt[1]).encode('utf-8'))
        f.write(','.encode('utf-8'))
        f.write(utt[2].strip().encode('utf-8'))
        f.write('\n'.encode('utf-8'))

print("total merged length: {:.3f} secs".format(acc_size))
