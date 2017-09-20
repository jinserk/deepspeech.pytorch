import os
import glob
import argparse
import subprocess
from utils import create_manifest, update_progress
import shutil

parser = argparse.ArgumentParser(description='Processes and downloads LibriSpeech dataset.')
parser.add_argument("--target_dir", default='librispeech', type=str, help="Directory to store the dataset.")
parser.add_argument('--sample_rate', default=8000, type=int, help='Sample rate')
parser.add_argument('--files_to_use', default="train-clean-100.tar.gz,"
                                              "train-clean-360.tar.gz,train-other-500.tar.gz,"
                                              "dev-clean.tar.gz,dev-other.tar.gz,"
                                              "test-clean.tar.gz,test-other.tar.gz", type=str,
                    help='list of file names to download')
args = parser.parse_args()

#LIBRI_SPEECH_URLS = {
#    "train": ["http://www.openslr.org/resources/12/train-clean-100.tar.gz",
#              "http://www.openslr.org/resources/12/train-clean-360.tar.gz",
#              "http://www.openslr.org/resources/12/train-other-500.tar.gz"],
#
#    "val": ["http://www.openslr.org/resources/12/dev-clean.tar.gz",
#            "http://www.openslr.org/resources/12/dev-other.tar.gz"],
#
#    "test": ["http://www.openslr.org/resources/12/test-clean.tar.gz",
#             "http://www.openslr.org/resources/12/test-other.tar.gz"]
#}
LIBRISPEECH_DIRS = {
    "train": ["train-clean-100", "train-clean-360", "train-other-500"],
    "val": ["dev-clean", "dev-other"],
    "test": ["test-clean", "test-other"],
}


def _preprocess_transcript(phrase):
    return phrase.strip().lower()


def _process_file(wav_dir, txt_dir, filename):
    base_dir = os.path.dirname(filename)
    base_filename = os.path.basename(filename)
    wav_recording_path = os.path.join(wav_dir, base_filename.replace(".flac", ".wav"))
    subprocess.call(["sox -V1 --norm {} -r {} -b 16 -c 1 {}".format(filename, str(args.sample_rate),
                                                                    wav_recording_path)], shell=True)
    # process transcript
    txt_transcript_path = os.path.join(txt_dir, base_filename.replace(".flac", ".txt"))
    transcript_file = os.path.join(base_dir, "-".join(base_filename.split('-')[:-1]) + ".trans.txt")
    assert os.path.exists(transcript_file), "Transcript file {} does not exist.".format(transcript_file)
    transcriptions = open(transcript_file).read().strip().split("\n")
    transcriptions = {t.split()[0].split("-")[-1]: " ".join(t.split()[1:]) for t in transcriptions}
    with open(txt_transcript_path, "w") as f:
        key = base_filename.replace(".flac", "").split("-")[-1]
        assert key in transcriptions, "{} is not in the transcriptions".format(key)
        f.write(_preprocess_transcript(transcriptions[key]))
        f.flush()


def main():
    #target_dl_dir = args.target_dir
    #if not os.path.exists(target_dl_dir):
    #    os.makedirs(target_dl_dir)
    files_to_dl = args.files_to_use.strip().split(',')
    for split_type, lst_libri in LIBRISPEECH_DIRS.items():
        split_dir = os.path.join(args.target_dir, split_type)
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
        split_wav_dir = os.path.join(split_dir, "wav")
        if not os.path.exists(split_wav_dir):
            os.makedirs(split_wav_dir)
        split_txt_dir = os.path.join(split_dir, "txt")
        if not os.path.exists(split_txt_dir):
            os.makedirs(split_txt_dir)
        #extracted_dir = os.path.join(split_dir, "LibriSpeech")
        #if os.path.exists(extracted_dir):
        #    shutil.rmtree(extracted_dir)
        for subdir in lst_libri:
            # check if we want to dl this file
            #dl_flag = False
            #for f in files_to_dl:
            #    if url.find(f) != -1:
            #        dl_flag = True
            #if not dl_flag:
            #    print("Skipping url: {}".format(url))
            #    continue
            #filename = url.split("/")[-1]
            #target_filename = os.path.join(split_dir, filename)
            #if not os.path.exists(target_filename):
            #    wget.download(url, split_dir)
            #print("Unpacking {}...".format(filename))
            #tar = tarfile.open(target_filename)
            #tar.extractall(split_dir)
            #tar.close()
            #os.remove(target_filename)
            extracted_dir = os.path.join(args.target_dir, subdir)
            print("Converting {}...".format(extracted_dir))
            assert os.path.exists(extracted_dir), "Archive {} was not properly uncompressed.".format(filename)
            counter = 0
            entries = glob.glob(os.path.join(extracted_dir, "**/*.flac"), recursive=True)
            for f in entries:
                _process_file(wav_dir=split_wav_dir, txt_dir=split_txt_dir, filename=f)
                counter += 1
                update_progress(counter / float(len(entries)))
            print("\n")
            #shutil.rmtree(extracted_dir)
        create_manifest(split_dir, args.target_dir, 'libri_' + split_type)


if __name__ == "__main__":
    main()
