# GMAN
This is a Pytorch implementation of the model described in the paper "[GMAN: A Graph Multi-Attention Network for Traffic Prediction](https://arxiv.org/abs/1911.08415)". This is not an official implementation. The official implementation by the authors(TensorFlow) can be found [here](https://github.com/zhengchuanpan/GMAN).

## Data
The datasets can be downloaded from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g), provided by [DCRNN](https://github.com/liyaguang/DCRNN). Place the traffic data files in the data folder and rename these as PeMS.h5 and METR.h5 for the respective datasets

## Usage

Train and evaluate the model by executing
```
python train.py --dataset PeMS --K 2 --max_epoch 150 --batch_size 10
python train.py --dataset METR --K 4 --max_epoch 100 --batch_size 10
```
