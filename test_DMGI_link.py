import numpy as np
np.random.seed(0)
import torch
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import argparse
from OpenAttMultiGL.utils.process import *
import scipy.io as sio

def parse_args():
    # input arguments
    parser = argparse.ArgumentParser(description='DMGI')

    parser.add_argument('--embedder', nargs='?', default='DMGI')
    parser.add_argument('--dataset', nargs='?', default='acm')
    #parser.add_argument('--metapaths', nargs='?', default='PLP,PAP')

    parser.add_argument('--nb_epochs', type=int, default=5)
    parser.add_argument('--hid_units', type=int, default=64)
    
    parser.add_argument('--training_ratio', type=float, default=0.3,
                        help='Training Ratio')
    parser.add_argument('--validing_ratio', type=float, default=0.1,
                        help='Validing Ratio')
    parser.add_argument('--lr', type = float, default = 0.001)
    parser.add_argument('--l2_coef', type=float, default=0.0001)
    parser.add_argument('--drop_prob', type=float, default=0.5)
    parser.add_argument('--reg_coef', type=float, default=0.001)
    parser.add_argument('--sup_coef', type=float, default=0.1)
    parser.add_argument('--sc', type=float, default=3.0, help='GCN self connection')
    parser.add_argument('--margin', type=float, default=0.1)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--nheads', type=int, default=1)
    parser.add_argument('--activation', nargs='?', default='relu')
    parser.add_argument('--isSemi', action='store_true', default=False)
    parser.add_argument('--isBias', action='store_true', default=False)
    parser.add_argument('--isAttn', action='store_true', default=False)
    parser.add_argument('--test_view', type=int, default=0, help='the id of gpu to use')
    return parser.parse_known_args()

def printConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))
    #print(args_names)
    #print(args_vals)

def main():
    args, unknown = parse_args()

    if args.embedder == 'DMGI':
        from OpenAttMultiGL.model.DMGI.DMGI_link import DMGI
        embedder = DMGI(args)
    elif args.embedder == 'DGI':
        from OpenAttMultiGL.model.DMGI.DGI_link import DGI
        embedder = DGI(args)

    #embedder.training()
    
    AUC, ap, hits = embedder.training()
    print("Average-percision:", np.mean(ap), np.std(ap))
    print("Average-AUC:", np.mean(AUC), np.std(AUC))
    
    return AUC, hits, ap

if __name__ == '__main__':
    args, unknown = parse_args()
    print(args.dataset)
    print(args.embedder)
    
    AUC, hits, ap = main()



