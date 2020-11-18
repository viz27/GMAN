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
parser.add_argument('--batch_size', type = int, default = 10,
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
trainTE = torch.LongTensor(trainTE).to(device)
trainY = torch.FloatTensor(trainY).to(device)
valX = torch.FloatTensor(valX).to(device)
valTE = torch.LongTensor(valTE).to(device)
valY = torch.FloatTensor(valY).to(device)
SE = torch.FloatTensor(SE).to(device)

TEmbsize = (24*60//args.time_slot)+7 #number of slots in a day + number of days in a week
gman = model.GMAN(args.K, args.d, SE.shape[1], TEmbsize, args.P, args.L, device).to(device)
optimizer = torch.optim.Adam(gman.parameters(), lr=args.learning_rate, weight_decay=0.00001)

# ~ pred = gman(trainX[0:10], SE, trainTE[0:10])
# ~ label = trainY[0:10]
# ~ loss = model.mae_loss(pred, label)
# ~ print("loss:", loss.item())

num_train, _, N = trainX.shape
num_val = valX.shape[0]
wait = 0
val_loss_min = np.inf
for epoch in range(args.max_epoch):
    if wait >= args.patience:
        utils.log_string(log, 'early stop at epoch: %04d' % (epoch))
        break
    # shuffle
    permutation = np.random.permutation(num_train)
    trainX = trainX[permutation]
    trainTE = trainTE[permutation]
    trainY = trainY[permutation]
    # train loss
    start_train = time.time()
    train_loss = 0
    num_batch = math.ceil(num_train / args.batch_size)
    for batch_idx in range(num_batch):
        gman.train()
        optimizer.zero_grad()
        print("Batch: ", batch_idx+1, "out of", num_batch, end=" | ")
        start_idx = batch_idx * args.batch_size
        end_idx = min(num_train, (batch_idx + 1) * args.batch_size)
        batchX = trainX[start_idx : end_idx]
        batchTE = trainTE[start_idx : end_idx]
        batchlabel = trainY[start_idx : end_idx]
        #print("batchXShape:", batchX.shape)
        #print("SEShape:", SE.shape)
        #print("batchTEShape:", batchTE.shape)
        batchpred = gman(batchX, SE, batchTE)
        batchloss = model.mae_loss(batchpred, batchlabel, device)
        print("Loss: ", batchloss.item())
        batchloss.backward()
        optimizer.step()
        train_loss += batchloss.item() * (end_idx - start_idx)
    train_loss /= num_train
    end_train = time.time()
    # val loss
    start_val = time.time()
    val_loss = 0
    num_batch = math.ceil(num_val / args.batch_size)
    for batch_idx in range(num_batch):
        gman.eval()
        start_idx = batch_idx * args.batch_size
        end_idx = min(num_val, (batch_idx + 1) * args.batch_size)
        batchX = valX[start_idx : end_idx]
        batchTE = valTE[start_idx : end_idx]
        batchlabel = valY[start_idx : end_idx]
        batchpred = gman(batchX, SE, batchTE)
        batchloss = model.mae_loss(batchpred, batchlabel, device)
        val_loss += batchloss.item() * (end_idx - start_idx)
    val_loss /= num_val
    end_val = time.time()
    utils.log_string(
        log,
        '%s | epoch: %04d/%d, training time: %.1fs, inference time: %.1fs' %
        (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1,
         args.max_epoch, end_train - start_train, end_val - start_val))
    utils.log_string(
        log, 'train loss: %.4f, val_loss: %.4f' % (train_loss, val_loss))
    # ~ if val_loss <= val_loss_min:
        # ~ utils.log_string(
            # ~ log,
            # ~ 'val loss decrease from %.4f to %.4f, saving model to %s' %
            # ~ (val_loss_min, val_loss, args.model_file))
        # ~ wait = 0
        # ~ val_loss_min = val_loss
        # ~ saver.save(sess, args.model_file)
    # ~ else:
        # ~ wait += 1
