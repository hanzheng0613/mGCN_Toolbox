import argparse
import numpy as np
import torch
import numpy as np
import pickle as pkl
import scipy.io as sio
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix, to_scipy_sparse_matrix

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--embedder', nargs='?', default='HDMI', help='HDMI or HDI')
    parser.add_argument('--hid_units', type=int, default=256, help='hidden dimension')
    parser.add_argument('--same_discriminator', type=bool, default=False,
                        help='whether to use the same discriminator for the layers and fusion module')

    parser.add_argument('--dataset', nargs='?', default='acm')
    parser.add_argument('--sc', type=float, default=3.0, help='GCN self connection')
    parser.add_argument('--sparse', type=bool, default=True, help='sparse adjacency matrix')
 
    parser.add_argument('--nb_epochs', type=int, default=1, help='the number of epochs')
    parser.add_argument('--training_ratio', type=float, default=0.3,
                        help='Training Ratio')
    parser.add_argument('--validing_ratio', type=float, default=0.1,
                        help='Validing Ratio')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
    parser.add_argument('--gpu_num', type=int, default=0, help='the id of gpu to use')
    parser.add_argument('--coef_layers', type=list, default=[1, 2, 0.001],
                        help='different layers of the multiplex network: '
                             'coefficients for the extrinsic, intrinsic and joint signals')
    parser.add_argument('--coef_fusion', type=list, default=[0.01, 0.1, 0.001],
                        help='fusion module: coefficient for the extrinsic, intrinsic and joint signals')
    parser.add_argument('--save_root', type=str, default="./saved_model", help='root for saving the model')
    parser.add_argument('--test_view', type=int, default=0, help='the id of gpu to use')
    return parser.parse_known_args()


def printConfig(args):
    arg2value = {}
    for arg in vars(args):
        arg2value[arg] = getattr(args, arg)
    print(arg2value)


def main():
    args, unknown = parse_args()
    printConfig(args)

    if args.embedder == "HDI":
        from OpenAttMultiGL.model.hdmi.hdi_link import HDI
        embedder = HDI(args)
    elif args.embedder == "HDMI":
        from OpenAttMultiGL.model.hdmi.hdmi_link import HDMI
        embedder = HDMI(args)

    AUC, ap, hits = embedder.training()
    print("Average-precision:", np.mean(ap), np.std(ap))
    print("Average-AUC:", np.mean(AUC), np.std(AUC))
    # f = open("results/final_results_0.4", 'a+')
    # f.write(args.dataset + '_' + str(args.test_view)+ f'   Average-percision: {ap:.4f} '
    #     + f'   Precision@20: {hits[0]:.4f} '
    #     + f'   Precision@50: {hits[1]:.4f} '
    #     + f'   Precision@100: {hits[2]:.4f} '
    #     + f'   Precision@200: {hits[3]:.4f} '
    #     + f'   Precision@500: {hits[4]:.4f} '
    #     + f'   Precision@1000: {hits[5]:.4f} '
    #     + f'   Precision@10000: {hits[6]:.4f}\n')
    return AUC, hits, ap


if __name__ == '__main__':
   
    AUC, hits, ap = main()



