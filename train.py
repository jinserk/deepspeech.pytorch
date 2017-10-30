import sys
import argparse
import errno
import json
import os
import time
import logging
import signal
from tqdm import tqdm

import torch
from torch.autograd import Variable
from torch.utils.data.sampler import BatchSampler
from warpctc_pytorch import CTCLoss
from data.data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler
from decoder import GreedyDecoder
from model import DeepSpeech, supported_rnns

sys.path.append('/home/jbaik/setup/pytorch/YellowFin_Pytorch/tuner_utils')
from yellowfin import YFOptimizer

parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--train_manifest', metavar='DIR',
                    help='path to train manifest csv', default='data/train_manifest.csv')
parser.add_argument('--val_manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/val_manifest.csv')
parser.add_argument('--sample_rate', default=8000, type=int, help='Sample rate')
parser.add_argument('--batch_size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in data-loading')
parser.add_argument('--labels_path', default='labels.json', help='Contains all characters for transcription')
parser.add_argument('--window_size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window_stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--hidden_size', default=800, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden_layers', default=5, type=int, help='Number of RNN layers')
parser.add_argument('--rnn_type', default='lstm', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--epochs', default=50, type=int, help='Number of training epochs')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--optim', default='sgd', type=str, help='Optimization method')
parser.add_argument('--optim_restart', dest='optim_restart', action='store_true', help='Optimization restart if continute_from exists')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
parser.add_argument('--alpha', default=0.99, type=float, help='RMSprop optimizer alpha')
parser.add_argument('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')
parser.add_argument('--epsilon', default=1e-8, type=float, help='Adam optimizer epsilon')
parser.add_argument('--max_norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--learning_anneal', default=1.1, type=float, help='Annealing applied to learning rate every epoch')
parser.add_argument('--silent', dest='silent', action='store_true', help='Turn off progress tracking per iteration')
parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Enables checkpoint saving of model')
parser.add_argument('--checkpoint_per_batch', default=0, type=int, help='Save checkpoint per batch. 0 means never save')
parser.add_argument('--visdom', dest='visdom', action='store_true', help='Turn on visdom graphing')
parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='Turn on tensorboard graphing')
parser.add_argument('--log_dir', default='visualize/deepspeech_final', help='Location of tensorboard log')
parser.add_argument('--log_params', dest='log_params', action='store_true', help='Log parameter values and gradients')
parser.add_argument('--id', default='Deepspeech training', help='Identifier for visdom/tensorboard run')
parser.add_argument('--save_folder', default='models/', help='Location to save epoch models')
parser.add_argument('--model_path', default='models/deepspeech_final.pth.tar',
                    help='Location to save best validation model')
parser.add_argument('--continue_from', default='', help='Continue from checkpoint model')
parser.add_argument('--augment', dest='augment', action='store_true', help='Use random tempo and gain perturbations.')
parser.add_argument('--noise_dir', default=None,
                    help='Directory to inject noise into audio. If default, noise Inject not added')
parser.add_argument('--noise_prob', default=0.4, help='Probability of noise being added per sample')
parser.add_argument('--noise_min', default=0.0,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise_max', default=0.5,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)
parser.add_argument('--sortagrad', dest='sortagrad', action='store_true',
                    help='Turn off sampling from dataset based on sequence length (smallest to largest)')

# create logger
log = logging.getLogger('deepspeech.pytorch')
log.setLevel(logging.DEBUG)
fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

# console handler
chdr = logging.StreamHandler()
chdr.setLevel(logging.DEBUG)
chdr.setFormatter(fmt)
log.addHandler(chdr)


def to_np(x):
    return x.data.cpu().numpy()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    args = parser.parse_args()
    save_folder = args.save_folder

    try:
        os.makedirs(save_folder, exist_ok=True)
        # log file handler
        fhdr = logging.FileHandler("{0}/train.log".format(save_folder))
        fhdr.setLevel(logging.DEBUG)
        fhdr.setFormatter(fmt)
        log.addHandler(fhdr)
    except OSError as e:
        raise

    log.critical('Training starts with command:')
    log.critical('{}'.format(' '.join(sys.argv)))

    loss_results = torch.Tensor(args.epochs)
    cer_results = torch.Tensor(args.epochs)
    wer_results = torch.Tensor(args.epochs)
    best_wer = None

    if args.visdom:
        from visdom import Visdom
        viz = Visdom()
        opts = dict(title=args.id, ylabel='', xlabel='Epoch', legend=['Loss', 'WER', 'CER'])
        viz_window = None
        epochs = torch.arange(1, args.epochs + 1)

    if args.tensorboard:
        try:
            os.makedirs(args.log_dir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                log.warn('Tensorboard log directory already exists: {}'.format(args.log_dir))
                for file in os.listdir(args.log_dir):
                    file_path = os.path.join(args.log_dir, file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        raise
            else:
                raise
        from tensorboardX import SummaryWriter
        tensorboard_writer = SummaryWriter(args.log_dir)

    criterion = CTCLoss()

    with open(args.labels_path) as label_file:
        labels = str(''.join(json.load(label_file)))

    audio_conf = dict(sample_rate=args.sample_rate,
                      window_size=args.window_size,
                      window_stride=args.window_stride,
                      window=args.window,
                      noise_dir=args.noise_dir,
                      noise_prob=args.noise_prob,
                      noise_levels=(args.noise_min, args.noise_max))

    train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, labels=labels,
                                       normalize=True, augment=args.augment)

    if args.sortagrad:
        train_sampler = BucketingSampler(train_dataset, batch_size=args.batch_size)
        train_loader = AudioDataLoader(train_dataset, batch_sampler=train_sampler,
                                       num_workers=args.num_workers, pin_memory=True)
    else:
        train_loader = AudioDataLoader(train_dataset, shuffle=True,
                                       num_workers=args.num_workers, pin_memory=True)
        train_sampler = train_loader.batch_sampler

    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.val_manifest, labels=labels,
                                      normalize=True, augment=False)
    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    rnn_type = args.rnn_type.lower()
    assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"

    model = DeepSpeech(rnn_hidden_size=args.hidden_size,
                       nb_layers=args.hidden_layers,
                       labels=labels,
                       rnn_type=supported_rnns[rnn_type],
                       audio_conf=audio_conf,
                       bidirectional=True)

    parameters = model.parameters()

    if args.optim == "rmsprop":
        log.info(f"optimization: RMSprop (lr={args.lr}, alpha={args.alpha}, eps={args.epsilon}, weight_decay=0, mementum=0, centered=False)")
        optimizer = torch.optim.RMSprop(parameters, lr=args.lr, alpha=args.alpha, eps=args.epsilon, weight_decay=0, momentum=0, centered=False)
    elif args.optim == "adam":
        log.info(f"optimization: Adam (lr={args.lr}, betas=({args.beta1}, {args.beta2}), eps={args.epsilon}, weight_decay=0)")
        optimizer = torch.optim.Adam(parameters, lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon, weight_decay=0)
        args.learning_anneal = 1.
    elif args.optim == "yellowfin":
        log.info(f"optimization: YFOptimizer (lr={args.lr}, mu=0.0)")
        optimizer = YFOptimizer(parameters, lr=args.lr, mu=0.0)
        args.learning_anneal = 1.
    else: #args.optim == "sgd":
        log.info(f"optimization: SGD (lr={args.lr}, momentum={args.momentum}, nestrov=True)")
        optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=args.momentum, nesterov=True)

    decoder = GreedyDecoder(labels)

    if args.continue_from:
        log.info("Loading checkpoint model %s" % args.continue_from)
        package = torch.load(args.continue_from)
        model.load_state_dict(package['state_dict'])
        if not args.optim_restart:
            #optimizer.load_state_dict(package['optim_dict'])
            # replace lr
            optim_state = package['optim_dict']
            #optim_state['param_groups'][0]['lr'] = args.lr / (args.learning_anneal ** 10)
            optimizer.load_state_dict(optim_state)

            start_epoch = int(package.get('epoch', 1)) - 1  # Python index start at 0 for training
            start_iter = package.get('iteration', None)
            if start_iter is None:
                start_epoch += 1  # Assume that we saved a model after an epoch finished, so start at the next epoch.
                start_iter = 0
            else:
                start_iter += 1

            log.info('Learning rate resetting to: {lr:.6f}'.format(lr=optim_state['param_groups'][0]['lr']))
            log.info('Epoch resetting to: {:d}'.format(start_epoch))
            log.info('iter resetting to: {:d}'.format(start_iter))
        else:
            start_epoch = 0
            start_iter = 0

        avg_loss = float(package.get('avg_loss', 0))

        loss_results, cer_results, wer_results = package['loss_results'], package['cer_results'], package['wer_results']
        if len(loss_results) < args.epochs:
            loss_results = torch.cat((loss_results, torch.Tensor(args.epochs - len(loss_results)).zero_()))
        if len(cer_results) < args.epochs:
            cer_results = torch.cat((cer_results, torch.Tensor(args.epochs - len(cer_results)).zero_()))
        if len(wer_results) < args.epochs:
            wer_results = torch.cat((wer_results, torch.Tensor(args.epochs - len(wer_results)).zero_()))

        if args.visdom and package['loss_results'] is not None and start_epoch > 0:  # Add previous scores to visdom graph
            x_axis = epochs[0:start_epoch]
            y_axis = torch.stack((loss_results[0:start_epoch], wer_results[0:start_epoch], cer_results[0:start_epoch]),
                                 dim=1)
            viz_window = viz.line(
                X=x_axis,
                Y=y_axis,
                opts=opts,
            )
        if args.tensorboard and package['loss_results'] is not None and start_epoch > 0:  # Previous scores to tensorboard logs
            for i in range(start_epoch):
                values = {
                    'Avg Train Loss': loss_results[i],
                    'Avg WER': wer_results[i],
                    'Avg CER': cer_results[i]
                }
                tensorboard_writer.add_scalars(args.id, values, i + 1)

        #if not args.no_shuffle and start_epoch != 0:
        #    print("Shuffling batches for the following epochs")
        #    train_sampler.shuffle()
    else:
        avg_loss = 0
        start_epoch = 0
        start_iter = 0
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()

    log.info(model)
    log.info("Number of parameters: %d" % DeepSpeech.get_param_size(model))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    if args.tensorboard:
        bcnt = 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        end = time.time()
        for i, (data) in enumerate(train_loader, start=start_iter):
            if i == len(train_sampler):
                break
            inputs, targets, input_percentages, target_sizes = data
            # measure data loading time
            data_time.update(time.time() - end)
            inputs = Variable(inputs, requires_grad=False)
            target_sizes = Variable(target_sizes, requires_grad=False)
            targets = Variable(targets, requires_grad=False)

            if args.cuda:
                inputs = inputs.cuda(async=True)

            out = model(inputs)
            out = out.transpose(0, 1)  # TxNxH

            seq_length = out.size(0)
            sizes = Variable(input_percentages.mul_(int(seq_length)).int(), requires_grad=False)

            loss = criterion(out, targets, sizes, target_sizes)
            loss = loss / inputs.size(0)  # average the loss by minibatch

            loss_sum = loss.data.sum()
            inf = float("inf")
            if loss_sum == inf or loss_sum == -inf:
                log.warn("received an inf loss, setting loss value to 0")
                loss_value = 0
            else:
                loss_value = loss.data[0]

            avg_loss += loss_value
            losses.update(loss_value, inputs.size(0))

            # compute gradient
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm(model.parameters(), args.max_norm)
            # SGD step
            optimizer.step()

            if args.cuda:
                torch.cuda.synchronize()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.silent:
                log.info('Epoch {0:03d}:  Batch {1:06d} / {2:06d}  '
                         'Time {batch_time.val:6.3f} (avg {batch_time.avg:6.3f})  '
                         'Data {data_time.val:6.3f} (avg {data_time.avg:6.3f})  '
                         'Loss {loss.val:8.4f} (avg {loss.avg:8.4f})'.format(
                         (epoch + 1), (i + 1), len(train_sampler), batch_time=batch_time,
                         data_time=data_time, loss=losses))

            if args.checkpoint_per_batch > 0 and i > 0 and (i + 1) % args.checkpoint_per_batch == 0:
                file_path = '%s/deepspeech_checkpoint_epoch_%03d_iter_%06d.pth.tar' % (save_folder, epoch + 1, i + 1)
                log.info("Saving checkpoint model to %s" % file_path)
                torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, iteration=i,
                                                loss_results=loss_results, wer_results=wer_results,
                                                cer_results=cer_results, avg_loss=avg_loss),
                           file_path)

            if args.tensorboard:
                bcnt += 1
                values = {
                    'Batch Loss': losses.val,
                    'Batch Loss Average': losses.avg,
                }
                tensorboard_writer.add_scalars("Batches", values, bcnt)

            del loss
            del out

        avg_loss /= len(train_sampler)

        log.info('Training Summary Epoch {0:03d}:  '
                 'Average Loss {loss:8.4f}'.format((epoch + 1), loss=avg_loss))

        start_iter = 0 # Reset start iteration for next epoch, and change to full range of train_data
        total_cer, total_wer = 0, 0
        model.eval()
        for i, (data) in enumerate(tqdm(test_loader)):  # test
            inputs, targets, input_percentages, target_sizes = data

            inputs = Variable(inputs, volatile=True)

            # unflatten targets
            split_targets = []
            offset = 0
            for size in target_sizes:
                split_targets.append(targets[offset:offset + size])
                offset += size

            if args.cuda:
                inputs = inputs.cuda(async=True)

            out = model(inputs)
            out = out.transpose(0, 1)  # TxNxH
            seq_length = out.size(0)
            sizes = input_percentages.mul_(int(seq_length)).int()

            decoded_output = decoder.decode(out.data, sizes)
            target_strings = decoder.process_strings(decoder.convert_to_strings(split_targets))
            wer, cer = 0, 0
            for x in range(len(target_strings)):
                wer += decoder.wer(decoded_output[x], target_strings[x]) / float(len(target_strings[x].split()))
                cer += decoder.cer(decoded_output[x], target_strings[x]) / float(len(target_strings[x]))
            total_cer += cer
            total_wer += wer

            if args.cuda:
                torch.cuda.synchronize()
            del out
        wer = total_wer / len(test_loader.dataset)
        cer = total_cer / len(test_loader.dataset)
        wer *= 100
        cer *= 100
        loss_results[epoch] = avg_loss
        wer_results[epoch] = wer
        cer_results[epoch] = cer
        log.info('Validation Summary Epoch {0:03d}:  '
                 'Average WER {wer:7.3f}  '
                 'Average CER {cer:7.3f}  '.format((epoch + 1), wer=wer, cer=cer))

        if args.visdom:
            x_axis = epochs[0:epoch + 1]
            y_axis = torch.stack((loss_results[0:epoch + 1], wer_results[0:epoch + 1], cer_results[0:epoch + 1]), dim=1)
            if viz_window is None:
                viz_window = viz.line(
                    X=x_axis,
                    Y=y_axis,
                    opts=opts,
                )
            else:
                viz.line(
                    X=x_axis.unsqueeze(0).expand(y_axis.size(1), x_axis.size(0)).transpose(0, 1),  # Visdom fix
                    Y=y_axis,
                    win=viz_window,
                    update='replace',
                )
        if args.tensorboard:
            values = {
                'Avg Train Loss': avg_loss,
                'Avg WER': wer,
                'Avg CER': cer
            }
            tensorboard_writer.add_scalars(args.id, values, epoch + 1)
            if args.log_params:
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    tensorboard_writer.add_histogram(tag, to_np(value), epoch + 1)
                    tensorboard_writer.add_histogram(tag + '/grad', to_np(value.grad), epoch + 1)
        if args.checkpoint:
            file_path = '%s/deepspeech_%03d.pth.tar' % (save_folder, epoch + 1)
            log.info("Saving checkpoint model to %s" % file_path)
            torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, loss_results=loss_results,
                                            wer_results=wer_results, cer_results=cer_results),
                       file_path)
        # anneal lr
        if args.optim != "yellowfin":
            optim_state = optimizer.state_dict()
            optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / args.learning_anneal
            optimizer.load_state_dict(optim_state)
            log.info('Learning rate annealed to: {lr:.6f}'.format(lr=optim_state['param_groups'][0]['lr']))

        if best_wer is None or best_wer > wer:
            log.info("Found better validated model, saving to %s" % args.model_path)
            torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, loss_results=loss_results,
                                            wer_results=wer_results, cer_results=cer_results),
                       args.model_path)
            best_wer = wer

        avg_loss = 0

        if args.sortagrad:
            print("Shuffling batches...")
            train_sampler.shuffle()


if __name__ == '__main__':
    main()

