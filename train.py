import math
import argparse
import utils, model
import time, datetime
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--time_slot', type = int, default = 5,
                    help = 'a time step is 5 mins')
parser.add_argument('--P', type = int, default = 12,
                    help = 'history steps')
parser.add_argument('--Q', type = int, default = 12,
                    help = 'prediction steps')
parser.add_argument('--L', type = int, default = 1,
                    help = 'number of STAtt Blocks')
parser.add_argument('--K', type = int, default = 2,
                    help = 'number of attention heads')
parser.add_argument('--d', type = int, default = 8,
                    help = 'dims of each head attention outputs')
parser.add_argument('--train_ratio', type = float, default = 0.7,
                    help = 'training set [default : 0.7]')
parser.add_argument('--val_ratio', type = float, default = 0.1,
                    help = 'validation set [default : 0.1]')
parser.add_argument('--test_ratio', type = float, default = 0.2,
                    help = 'testing set [default : 0.2]')
parser.add_argument('--batch_size', type = int, default = 32,
                    help = 'batch size')
parser.add_argument('--max_epoch', type = int, default = 1000,
                    help = 'epoch to run')
parser.add_argument('--patience', type = int, default = 10,
                    help = 'patience for early stop')
parser.add_argument('--learning_rate', type=float, default = 0.001,
                    help = 'initial learning rate')
parser.add_argument('--decay_epoch', type=int, default = 5,
                    help = 'decay epoch')
parser.add_argument('--traffic_file', default = 'data/PeMS.h5',
                    help = 'traffic file')
parser.add_argument('--SE_file', default = 'data/SE(PeMS).txt',
                    help = 'spatial emebdding file')
parser.add_argument('--model_file', default = 'data/GMAN(PeMS)',
                    help = 'save the model to disk')
parser.add_argument('--log_file', default = 'data/log(PeMS)',
                    help = 'log file')
args = parser.parse_args()

start = time.time()

log = open(args.log_file, 'w')
utils.log_string(log, str(args)[10 : -1])

# load data
utils.log_string(log, 'loading data...')
(trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, SE,
 mean, std) = utils.loadData(args)
utils.log_string(log, 'trainX: %s\ttrainY: %s' % (trainX.shape, trainY.shape))
utils.log_string(log, 'valX:   %s\t\tvalY:   %s' % (valX.shape, valY.shape))
utils.log_string(log, 'testX:  %s\t\ttestY:  %s' % (testX.shape, testY.shape))
utils.log_string(log, 'data loaded!')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#transform data to tensors
trainX = torch.FloatTensor(trainX).to(device)
SE = torch.FloatTensor(SE).to(device)
trainTE = torch.LongTensor(trainTE).to(device)

TEmbsize = (24*60//args.time_slot)+7 #number of slots in a day + number of days in a week
gman = model.GMAN(args.K, args.d, SE.shape[1], TEmbsize, args.P, args.L).to(device)
optimizer = torch.optim.Adam(gman.parameters(), lr=args.learning_rate, weight_decay=0.00001)

output = gman(trainX[0:10], SE, trainTE[0:10])
print(output.shape)
