#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

# Prerequisite: Having the sph2pipe tool in your PATH:
# https://www.ldc.upenn.edu/language-resources/tools/sphere-conversion-tools

# Make sure we can import stuff from util/
# This script needs to be run from the root of the DeepSpeech repository
import sys
import os

import codecs
import os
import glob
import re
import pandas
import subprocess
import unicodedata
import wave
import audioop

sph2pipe_bin = "/d1/jbaik/kaldi/tools/sph2pipe_v2.5/sph2pipe"

def _download_and_preprocess_data(data_dir):
    # Assume data_dir contains extracted LDC2004S13, LDC2004T19, LDC2005S13, LDC2005T19

    # Conditionally convert Fisher sph data to wav
    _maybe_convert_wav(data_dir, "LDC2004S13", "fisher-2004-wav")
    _maybe_convert_wav(data_dir, "LDC2005S13", "fisher-2005-wav")

    # Conditionally split Fisher wav data
    all_2004 = _split_wav_and_sentences(data_dir,
                             original_data="fisher-2004-wav",
                             converted_data="fisher-2004-split-wav",
                             trans_data=os.path.join("LDC2004T19", "fe_03_p1_tran", "data", "trans"))
    all_2005 = _split_wav_and_sentences(data_dir,
                             original_data="fisher-2005-wav",
                             converted_data="fisher-2005-split-wav",
                             trans_data=os.path.join("LDC2005T19", "fe_03_p2_tran", "data", "trans"))

    # The following files have incorrect transcripts that are much longer than
    # their audio source. The result is that we end up with more labels than time
    # slices, which breaks CTC.
    #all_2004.loc[all_2004["wav_filename"].str.endswith("fe_03_00265-33.53-33.81.wav"), "transcript"] = "correct"
    #all_2004.loc[all_2004["wav_filename"].str.endswith("fe_03_00991-527.39-528.3.wav"), "transcript"] = "that's one of those"
    #all_2005.loc[all_2005["wav_filename"].str.endswith("fe_03_10282-344.42-344.84.wav"), "transcript"] = "they don't want"
    #all_2005.loc[all_2005["wav_filename"].str.endswith("fe_03_10677-101.04-106.41.wav"), "transcript"] = "uh my mine yeah the german shepherd pitbull mix he snores almost as loud as i do"
    _replace_transcript(all_2004, "fe_03_00265-33.53-33.81.wav", "correct")
    _replace_transcript(all_2004, "fe_03_00991-527.39-528.3.wav", "that's one of those")
    _replace_transcript(all_2005, "fe_03_10282-344.42-344.84.wav", "they don't want")
    _replace_transcript(all_2005, "fe_03_10677-101.04-106.41.wav", "uh my mine yeah the german shepherd pitbull mix he snores almost as loud as i do")

    # The following file is just a short sound and not at all transcribed like provided.
    # So we just exclude it.
    all_2004 = all_2004[~all_2004["wav_filename"].str.endswith("fe_03_00027-393.8-394.05.wav")]

    # The following file is far too long and would ruin our training batch size.
    # So we just exclude it.
    all_2005 = all_2005[~all_2005["wav_filename"].str.endswith("fe_03_11487-31.09-234.06.wav")]

    # Conditionally split Fisher data into train/validation/test sets
    train_2004, dev_2004, test_2004 = _split_sets(all_2004)
    train_2005, dev_2005, test_2005 = _split_sets(all_2005)

    # Join 2004 and 2005 data
    train_files = train_2004.append(train_2005)
    dev_files = dev_2004.append(dev_2005)
    test_files = test_2004.append(test_2005)

    # Write sets to disk as CSV files
    train_files.to_csv(os.path.join(data_dir, "fisher-train.csv"), header=False, index=False)
    dev_files.to_csv(os.path.join(data_dir, "fisher-dev.csv"), header=False, index=False)
    test_files.to_csv(os.path.join(data_dir, "fisher-test.csv"), header=False, index=False)

def _replace_transcript(files, wav_filename, transcript):
    txt_filename = files.loc[files["wav_filename"].str.endswith(wav_filename), "txt_filename"].values[0]
    transcript = validate_label(transcript)
    if transcript != None:
        with open(txt_filename, "w") as f:
            f.write(transcript)

def _maybe_convert_wav(data_dir, original_data, converted_data):
    source_dir = os.path.join(data_dir, original_data)
    target_dir = os.path.join(data_dir, converted_data)

    # Conditionally convert sph files to wav files
    if os.path.exists(target_dir):
        print("skipping maybe_convert_wav")
        return

    # Create target_dir
    os.makedirs(target_dir)

    # Loop over sph files in source_dir and convert each to 16-bit PCM wav
    for sph_file in glob.iglob(os.path.join(source_dir, "**/*.sph"), recursive=True):
       for channel in ["1", "2"]:
            wav_filename = os.path.splitext(os.path.basename(sph_file))[0] + "_c" + channel + ".wav"
            wav_file = os.path.join(target_dir, wav_filename)
            print("converting {} to {}".format(sph_file, wav_file))
            subprocess.check_call([sph2pipe_bin, "-c", channel, "-p", "-f", "rif", sph_file, wav_file])

def _parse_transcriptions(trans_file):
    segments = []
    with codecs.open(trans_file, "r", "utf-8") as fin:
        for line in fin:
            if line.startswith("#") or len(line) <= 1:
                continue

            tokens = line.split()
            start_time = float(tokens[0])
            stop_time = float(tokens[1])
            speaker = tokens[2]
            transcript = " ".join(tokens[3:])

            # We need to do the encode-decode dance here because encode
            # returns a bytes() object on Python 3, and text_to_char_array
            # expects a string.
            transcript = unicodedata.normalize("NFKD", transcript)  \
                                    .encode("ascii", "ignore")      \
                                    .decode("ascii", "ignore")

            segments.append({
                "start_time": start_time,
                "stop_time": stop_time,
                "speaker": speaker,
                "transcript": transcript,
            })
    return segments

def _cut_utterance(src_sph_file, target_wav_file, start_time, end_time, sample_rate=8000):
    subprocess.call(["sox -V1 --norm {} -r {} -b 16 -c 1 {} trim {} ={}".format(src_sph_file,
                                                                                str(sample_rate),
                                                                                target_wav_file,
                                                                                start_time,
                                                                                end_time)],
                    shell=True)

def _split_wav_and_sentences(data_dir, trans_data, original_data, converted_data):
    trans_dir = os.path.join(data_dir, trans_data)
    source_dir = os.path.join(data_dir, original_data)
    target_dir = os.path.join(data_dir, converted_data)
    os.makedirs(target_dir, exist_ok=True)

    target_wav_dir = os.path.join(target_dir, "wav")
    target_txt_dir = os.path.join(target_dir, "txt")
    os.makedirs(target_wav_dir, exist_ok=True)
    os.makedirs(target_txt_dir, exist_ok=True)

    files = []

    # Loop over transcription files and split corresponding wav
    for trans_file in glob.iglob(os.path.join(trans_dir, "**/*.txt"), recursive=True):
        segments = _parse_transcriptions(trans_file)

        # Open wav corresponding to transcription file
        wav_filenames = [os.path.splitext(os.path.basename(trans_file))[0] + "_c" + channel + ".wav" for channel in ["1", "2"]]
        wav_files = [os.path.join(source_dir, wav_filename) for wav_filename in wav_filenames]

        print("splitting {} according to {}".format(wav_files, trans_file))

        origAudios = [wave.open(wav_file, "r") for wav_file in wav_files]

        # Loop over segments and split wav_file for each segment
        for segment in segments:
            # Create wav segment filename
            start_time = segment["start_time"]
            stop_time = segment["stop_time"]
            base_filename = os.path.splitext(os.path.basename(trans_file))[0] + "-" + str(start_time) + "-" + str(stop_time)
            new_wav_filename = base_filename + ".wav"
            new_wav_file = os.path.join(target_wav_dir, new_wav_filename)

            channel = 0 if segment["speaker"] == "A:" else 1
            _split_and_resample_wav(origAudios[channel], start_time, stop_time, new_wav_file)

            new_txt_filename = base_filename + ".txt"
            new_txt_file = os.path.join(target_txt_dir, new_txt_filename)

            new_wav_filesize = os.path.getsize(new_wav_file)
            transcript = validate_label(segment["transcript"])
            if transcript != None:
                files.append((os.path.abspath(new_wav_file),
                              os.path.abspath(new_txt_file)))
                with open(new_txt_file, "w") as f:
                    f.write(transcript)

        # Close origAudios
        for origAudio in origAudios:
            origAudio.close()

    return pandas.DataFrame(data=files, columns=["wav_filename", "txt_filename"])

def _split_and_resample_wav(origAudio, start_time, stop_time, new_wav_file):
    nChannels = origAudio.getnchannels()
    sampleWidth = origAudio.getsampwidth()
    frameRate = origAudio.getframerate()
    origAudio.setpos(int(start_time * frameRate))
    chunkData = origAudio.readframes(int((stop_time - start_time) * frameRate))
    # by doubling the frame-rate we effectively go from 8 kHz to 16 kHz
    chunkData, _ = audioop.ratecv(chunkData, sampleWidth, nChannels, frameRate, 2 * frameRate, None)
    chunkAudio = wave.open(new_wav_file, "w")
    chunkAudio.setnchannels(nChannels)
    chunkAudio.setsampwidth(sampleWidth)
    chunkAudio.setframerate(2 * frameRate)
    chunkAudio.writeframes(chunkData)
    chunkAudio.close()

def _split_sets(filelist):
    # We initially split the entire set into 80% train and 20% test, then
    # split the train set into 80% train and 20% validation.
    train_beg = 0
    train_end = int(0.8 * len(filelist))

    dev_beg = int(0.8 * train_end)
    dev_end = train_end
    train_end = dev_beg

    test_beg = dev_end
    test_end = len(filelist)

    return (filelist[train_beg:train_end],
            filelist[dev_beg:dev_end],
            filelist[test_beg:test_end])

# Validate and normalize transcriptions. Returns a cleaned version of the label
# or None if it's invalid.
def validate_label(label):
    # For now we can only handle [a-z ']
    if "(" in label or \
                    "<" in label or \
                    "[" in label or \
                    "]" in label or \
                    "&" in label or \
                    "*" in label or \
                    "{" in label or \
            re.search(r"[0-9]", label) != None:
        return None

    label = label.replace("-", "")
    label = label.replace("_", "")
    label = label.replace(".", "")
    label = label.replace(",", "")
    label = label.replace("?", "")
    label = label.strip()

    return label.upper()


if __name__ == "__main__":
    _download_and_preprocess_data(sys.argv[1])
