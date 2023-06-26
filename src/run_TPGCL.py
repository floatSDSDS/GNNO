
import os
import sys
from tqdm import tqdm
import pickle
import logging
import argparse
import torch

from helpers import *
from models.general import *
from models.sequential import *
from models.developing import *

from utils import utils
from models.sequential.ProbG import ProbGraph


def parse_global_args(parser):
    parser.add_argument('--gpu', type=str, default='',
                        help='Set CUDA_VISIBLE_DEVICES, default for CPU only')
    parser.add_argument('--verbose', type=int, default=logging.INFO,
                        help='Logging Level, 0, 10, ..., 50')
    parser.add_argument('--log_file', type=str, default='',
                        help='Logging file path')
    parser.add_argument('--random_seed', type=int, default=0,
                        help='Random seed of numpy and pytorch')
    parser.add_argument('--load', type=int, default=0,
                        help='Whether load model and continue to train')
    parser.add_argument('--train', type=int, default=1,
                        help='To train the model or not.')
    parser.add_argument('--regenerate', type=int, default=0,
                        help='Whether to regenerate intermediate files')

    parser.add_argument('--key_g', type=str, default='witg',
                        help='build graph in [witg, ui]')
    parser.add_argument('--min_seq_len', type=int, default=5,
                        help='Whether to regenerate intermediate files')
    parser.add_argument('--short_len', type=int, default=3,
                        help='Whether to regenerate intermediate files')
    parser.add_argument('--k0', type=int, default=5,
                        help='')
    parser.add_argument('--kt', type=int, default=7,
                        help='')
    return parser


def main():
    model, data_dict = load_data()
    probe_g = ProbGraph(args, model, data_dict)
    return


def load_data():
    # read data
    corpus_path = os.path.join(args.path, args.dataset, model_name.reader + '.pkl')
    corpus = pickle.load(open(corpus_path, 'rb'))

    # define model
    model = model_name(args, corpus).to(args.device)
    model.load_model()

    # load data
    data_dict = dict()
    for phase in ['train', 'dev', 'test']:
        data_dict[phase] = model_name.Dataset(model, corpus, phase)
        data_dict[phase].prepare()
    return model, data_dict


if __name__ == '__main__':
    init_parser = argparse.ArgumentParser(description='Model')
    init_parser.add_argument('--model_name', type=str, default='STRec', help='Choose a model to run.')
    init_args, init_extras = init_parser.parse_known_args()
    model_name = eval('{0}.{0}'.format(init_args.model_name))
    reader_name = eval('{0}.{0}'.format(model_name.reader))  # model chooses the reader
    runner_name = eval('{0}.{0}'.format(model_name.runner))  # model chooses the runner

    # Args
    parser = argparse.ArgumentParser(description='')
    parser = parse_global_args(parser)
    parser = reader_name.parse_data_args(parser)
    parser = runner_name.parse_runner_args(parser)
    parser = model_name.parse_model_args(parser)
    args, extras = parser.parse_known_args()

    # Logging configuration
    log_args = [init_args.model_name, args.dataset, str(args.random_seed)]
    for arg in ['lr', 'l2'] + model_name.extra_log_args:
        log_args.append(arg + '=' + str(eval('args.' + arg)))
    log_file_name = '__'.join(log_args).replace(' ', '__')
    if args.log_file == '':
        args.log_file = '../log/{}/{}.txt'.format(init_args.model_name, log_file_name)
    if args.model_path == '':
        args.model_path = '../model/{}/{}.pt'.format(init_args.model_name, log_file_name)

    utils.check_dir(args.log_file)

    # Random seed
    utils.init_seed(args.random_seed)

    # GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cpu')
    if args.gpu != '' and torch.cuda.is_available():
        args.device = torch.device('cuda')

    main()
