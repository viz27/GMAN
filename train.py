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
parser.add_argument('--patience', type = int, default = 5,
                    help = 'patience for early stop')
parser.add_argument('--learning_rate', type=float, default = 0.001,
                    help = 'initial learning rate')
parser.add_argument('--decay_epoch', type=int, default = 5,
                    help = 'decay epoch')
parser.add_argument('--path', default = './',
                    help = 'traffic file')
parser.add_argument('--dataset', default = 'PeMS',
                    help = 'Traffic dataset name')
parser.add_argument('--load_model', default = 'F',
                    help = 'Set T if pretrained model is to be loaded before training start')

args = parser.parse_args()
LOG_FILE = args.path+'data/log('+args.dataset+')'
MODEL_FILE = args.path+'data/GMAN('+args.dataset+')'

start = time.time()

log = open(LOG_FILE, 'w')
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
testX = torch.FloatTensor(testX).to(device)
testTE = torch.LongTensor(testTE).to(device)
testY = torch.FloatTensor(testY).to(device)
SE = torch.FloatTensor(SE).to(device)

TEmbsize = (24*60//args.time_slot)+7 #number of slots in a day + number of days in a week
gman = model.GMAN(args.K, args.d, SE.shape[1], TEmbsize, args.P, args.L, device).to(device)
optimizer = torch.optim.Adam(gman.parameters(), lr=args.learning_rate, weight_decay=0.00001)

# ~ pred = gman(trainX[0:10], SE, trainTE[0:10])
# ~ label = trainY[0:10]
# ~ loss = model.mae_loss(pred, label)
# ~ print("loss:", loss.item())

utils.log_string(log, '**** training model ****')
if args.load_model == 'T':
    utils.log_string(log, 'loading pretrained model from %s' % MODEL_FILE)
    gman.load_state_dict(torch.load(MODEL_FILE))
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
        if (batch_idx+1) % 100 == 0:
            print("Batch: ", batch_idx+1, "out of", num_batch, end=" | ")
            print("Loss: ", batchloss.item(), flush=True)
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
    if val_loss <= val_loss_min:
        utils.log_string(
            log,
            'val loss decrease from %.4f to %.4f, saving model to %s' %
            (val_loss_min, val_loss, MODEL_FILE))
        wait = 0
        val_loss_min = val_loss
        torch.save(gman.state_dict(), MODEL_FILE)
    else:
        wait += 1


# test model
utils.log_string(log, '**** testing model ****')
utils.log_string(log, 'loading model from %s' % MODEL_FILE)
gman.load_state_dict(torch.load(MODEL_FILE))
utils.log_string(log, 'model restored!')
utils.log_string(log, 'evaluating...')

num_test = testX.shape[0]

trainPred = []
num_batch = math.ceil(num_train / args.batch_size)
for batch_idx in range(num_batch):
    start_idx = batch_idx * args.batch_size
    end_idx = min(num_train, (batch_idx + 1) * args.batch_size)
    batchX = trainX[start_idx : end_idx]
    batchTE = trainTE[start_idx : end_idx]
    batchlabel = trainY[start_idx : end_idx]
    batchpred = gman(batchX, SE, batchTE)
    trainPred.append(batchpred.detach().cpu().numpy())
trainPred = np.concatenate(trainPred, axis = 0)

valPred = []
num_batch = math.ceil(num_val / args.batch_size)
for batch_idx in range(num_batch):
    start_idx = batch_idx * args.batch_size
    end_idx = min(num_val, (batch_idx + 1) * args.batch_size)
    batchX = valX[start_idx : end_idx]
    batchTE = valTE[start_idx : end_idx]
    batchlabel = valY[start_idx : end_idx]
    batchpred = gman(batchX, SE, batchTE)
    valPred.append(batchpred.detach().cpu().numpy())
valPred = np.concatenate(valPred, axis = 0)

testPred = []
num_batch = math.ceil(num_test / args.batch_size)
start_test = time.time()
for batch_idx in range(num_batch):
    start_idx = batch_idx * args.batch_size
    end_idx = min(num_test, (batch_idx + 1) * args.batch_size)
    batchX = testX[start_idx : end_idx]
    batchTE = testTE[start_idx : end_idx]
    batchlabel = testY[start_idx : end_idx]
    batchpred = gman(batchX, SE, batchTE)
    testPred.append(batchpred.detach().cpu().numpy())
end_test = time.time()
testPred = np.concatenate(testPred, axis = 0)

trainY = trainY.cpu().numpy()
valY = valY.cpu().numpy()
testY = testY.cpu().numpy()

train_mae, train_rmse, train_mape = utils.metric(trainPred, trainY)
val_mae, val_rmse, val_mape = utils.metric(valPred, valY)
test_mae, test_rmse, test_mape = utils.metric(testPred, testY)
utils.log_string(log, 'testing time: %.1fs' % (end_test - start_test))
utils.log_string(log, '                MAE\t\tRMSE\t\tMAPE')
utils.log_string(log, 'train            %.2f\t\t%.2f\t\t%.2f%%' %
                 (train_mae, train_rmse, train_mape * 100))
utils.log_string(log, 'val              %.2f\t\t%.2f\t\t%.2f%%' %
                 (val_mae, val_rmse, val_mape * 100))
utils.log_string(log, 'test             %.2f\t\t%.2f\t\t%.2f%%' %
                 (test_mae, test_rmse, test_mape * 100))
utils.log_string(log, 'performance in each prediction step')
MAE, RMSE, MAPE = [], [], []
for q in range(args.Q):
    mae, rmse, mape = utils.metric(testPred[:, q], testY[:, q])
    MAE.append(mae)
    RMSE.append(rmse)
    MAPE.append(mape)
    utils.log_string(log, 'step: %02d         %.2f\t\t%.2f\t\t%.2f%%' %
                     (q + 1, mae, rmse, mape * 100))
average_mae = np.mean(MAE)
average_rmse = np.mean(RMSE)
average_mape = np.mean(MAPE)
utils.log_string(
    log, 'average:         %.2f\t\t%.2f\t\t%.2f%%' %
    (average_mae, average_rmse, average_mape * 100))
end = time.time()
utils.log_string(log, 'total time: %.1fmin' % ((end - start) / 60))
log.close()
