import argparse
import torch
from torch.utils.data import DataLoader
from dataloader.train_dataload import load_train
from dataloader.test_dataload import load_test
from Train import initiate
import logging
import time
parser = argparse.ArgumentParser()
"""
train setting
"""
parser.add_argument('--BATCH_SIZE', default=32, type=int, metavar='N')
parser.add_argument('--Epochs', default=500, type=int, metavar='N')
parser.add_argument('--lr', default=0.00001, type=float, metavar='LR', dest='lr')
parser.add_argument('--milestones', default=[100,200,300,400], type=int, metavar='milestones', )
parser.add_argument('--gamma', default=0.9, type=float, metavar='gamma', )
parser.add_argument('--imgload', default='test1.jpg', type=str)

"""
data setting
"""
parser.add_argument('--resize', default=128, type=int, metavar='N')
parser.add_argument('--clip_len', default=24, type=int, metavar='N')
parser.add_argument('--loads', default=[100, 200, 300, 400], type=str)
args = parser.parse_args()


load = "LMTformer/csv_load/"  # csv_load Path
Path = "data/AVEC2013/"   # AVEC2013 Data Path

train_data = load_train(root=Path, csv_load=load, mode='train', args=args)
test_data = load_test(root=Path, csv_load=load, mode='test', args=args)

dataloaders = {
    'train': DataLoader(train_data, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True),  # train dataloader
    'test': DataLoader(test_data, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)  # test dataloader
}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
times = time.strftime('%m-%d %H_%M')
logging.basicConfig(filename='./log/'+str(times)+".log", filemode="a", format="%(message)s", level=logging.INFO)
logging.info(time.strftime('%Y-%m-%d %H:%M'))
logging.info('2013,LMTformer')
logging.info('BATCH_SIZE {:3d} ; Epochs {:3d} ; lr {:7.5f}'.
                format(args.BATCH_SIZE, args.Epochs, args.lr))
if __name__ == '__main__':
    # start training
    initiate(dataloaders['train'], dataloaders['test'],device, args)
