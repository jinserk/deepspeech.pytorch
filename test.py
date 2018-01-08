import argparse

import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

from decoder import BeamCTCDecoder, GreedyDecoder, LatticeDecoder

from data.data_loader import SpectrogramDataset, AudioDataLoader
from model import DeepSpeech

parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser.add_argument('--model_path', default='models/deepspeech_final.pth.tar',
                    help='Path to model file created by training')
parser.add_argument('--cuda', action="store_true", help='Use cuda to test model')
parser.add_argument('--test_manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/test_manifest.csv')
parser.add_argument('--batch_size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--decoder', default="greedy", choices=["greedy", "beam", "lattice", "none"], type=str, help="Decoder to use")
parser.add_argument('--verbose', action="store_true", help="print out decoded output and error of each sample")

no_decoder_args = parser.add_argument_group("No Decoder Options", "Configuration options for when no decoder is specified")
no_decoder_args.add_argument('--output_path', default=None, type=str, help="Where to save raw acoustic output")

beam_args = parser.add_argument_group("Beam Decode Options", "Configurations options for the CTC Beam Search decoder")
beam_args.add_argument('--top_paths', default=1, type=int, help='number of beams to return')
beam_args.add_argument('--beam_width', default=10, type=int, help='Beam width to use')
beam_args.add_argument('--lm_path', default=None, type=str,
                       help='Path to an (optional) kenlm language model for use with beam search (req\'d with trie)')
beam_args.add_argument('--alpha', default=0.8, type=float, help='Language model weight')
beam_args.add_argument('--beta', default=1, type=float, help='Language model word bonus (all words)')
beam_args.add_argument('--cutoff_top_n', default=40, type=int,
                       help='Cutoff number in pruning, only top cutoff_top_n characters with highest probs in '
                            'vocabulary will be used in beam search, default 40.')
beam_args.add_argument('--cutoff_prob', default=1.0, type=float,
                       help='Cutoff probability in pruning,default 1.0, no pruning.')
beam_args.add_argument('--lm_workers', default=1, type=int, help='Number of LM processes to use')

args = parser.parse_args()

if __name__ == '__main__':
    if not torch.cuda.is_available():
        args.cuda = False

    model = DeepSpeech.load_model(args.model_path, cuda=args.cuda)
    model.eval()

    labeler = DeepSpeech.get_labeler(model)
    audio_conf = DeepSpeech.get_audio_conf(model)

    if args.decoder == "beam":
        decoder = BeamCTCDecoder(labeler, lm_path=args.lm_path, alpha=args.alpha, beta=args.beta,
                                 cutoff_top_n=args.cutoff_top_n, cutoff_prob=args.cutoff_prob,
                                 beam_width=args.beam_width, num_processes=args.lm_workers)
    elif args.decoder == "greedy":
        decoder = GreedyDecoder(labeler, blank_index=labels.index('_'))
    elif args.decoder == "lattice":
        decoder = LatticeDecoder(labeler)
    else:
        decoder = None

    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.test_manifest,
                                      labeler=labeler, count_label=False, normalize=True, augment=False)
    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=args.cuda)

    total_cer, total_wer = 0, 0
    output_data = []
    for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs, targets, transcripts, input_percentages, target_sizes = data

        #inputs = Variable(inputs, volatile=True)
        inputs = Variable(inputs)

        with torch.no_grad():
            # unflatten targets
            split_targets = []
            offset = 0
            for size in target_sizes:
                split_targets.append(targets[offset:offset + size])
                offset += size

            if args.cuda:
                inputs = inputs.cuda(async=False)

            out = model(inputs)
            out = out.transpose(0, 1)  # TxNxH
            seq_length = out.size(0)
            sizes = input_percentages.mul_(int(seq_length)).int()

            if decoder is None:
                # add output to data array, and continue
                output_data.append((out.data.cpu().numpy(), sizes.numpy()))
                continue

            wer, cer = 0, 0

            if labeler.is_char():
                decoded_output, _, = decoder.decode(out.data, sizes)
                #target_strings = decoder.convert_to_strings(split_targets)
                for x in range(len(transcripts)):
                    transcript, reference = decoded_output[x][0], transcripts[x]
                    wer_inst = decoder.wer(transcript, reference) / float(len(reference.split()))
                    cer_inst = decoder.cer(transcript, reference) / float(len(reference))
                    wer += wer_inst
                    cer += cer_inst
                    if args.verbose:
                        print("Ref:", reference.lower())
                        print("Hyp:", transcript.lower())
                        print("WER:", wer_inst, "CER:", cer_inst, "\n")
            else: # if phone labeling, cer is used to count token error rate
                decoded_tokens, _ = decoder.decode_token(out.data, sizes, index_output=True)
                decoded_output, _ = decoder.decode(out.data, sizes)
                #target_tokens = decoder.greedy_check(split_targets, index_output=True)
                for x in range(len(transcripts)):
                    tokens, ref_tokens = decoded_tokens[x][0], split_targets[x]
                    transcript, reference = decoded_output[x][0], transcripts[x]
                    wer_inst = decoder.wer(transcript, reference) / float(len(reference.split()))
                    cer_inst = decoder.cer(tokens, ref_tokens) / float(len(ref_tokens))
                    wer += wer_inst
                    cer += cer_inst
                    if args.verbose:
                        print("Ref:", reference.lower())
                        print("Hyp:", transcript.lower())
                        print("WER:", wer_inst, "CER:", cer_inst, "\n")

            total_cer += cer
            total_wer += wer

    if decoder is not None:
        wer = total_wer / len(test_loader.dataset)
        cer = total_cer / len(test_loader.dataset)

        print('Test Summary \t'
              'Average WER {wer:.3f}\t'
              'Average CER {cer:.3f}\t'.format(wer=wer * 100, cer=cer * 100))
    else:
        np.save(args.output_path, output_data)
