import argparse

from OpenAttMultiGL.model.X_GOAL.goal_link import GOAL

from sklearn.metrics import normalized_mutual_info_score, pairwise, f1_score
from OpenAttMultiGL.utils.dataset import dataset
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', nargs='?', default='dblp')
    parser.add_argument('--model', type=str, default='goal')

    parser.add_argument('--hid_units', type=int, default=128, help='hidden dimension')
    parser.add_argument('--nb_epochs', type=int, default=1, help='the maximum number of epochs')
    parser.add_argument('--training_ratio', type=float, default=0.3,
                        help='Training Ratio')
    parser.add_argument('--validing_ratio', type=float, default=0.1,
                        help='Validing Ratio')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
    parser.add_argument('--gpu_num', type=int, default=0, help='the id of gpu to use')

    parser.add_argument('--save_root', type=str, default="./saved_model", help='root for saving the model')
    parser.add_argument('--pretrained_model_path', type=str, default="",
                        help='path to the pretrained model')
    parser.add_argument('--test_view', type=int, default=0, help='the id of gpu to use')
    # hyper-parameters for info-nce
    parser.add_argument('--p_drop', type=float, default=0.5, help='dropout rate for attributes')

    # hyper-parameters for clusters
    parser.add_argument('--k', type=list, default=4, help='the numbers of clusters')
    parser.add_argument('--tau', type=list, default=1, help='the temperature of clusters')
    parser.add_argument('--w_cluster', type=list, default=1e-2, help='weight for cluster loss')
    parser.add_argument('--cluster_step', type=int, default=5, help='every n steps to perform clustering')

    # warm-up
    parser.add_argument('--is_warmup', type=bool, default=True, help='whether to warm up or not')
    parser.add_argument('--warmup_lr', type=float, default=5e-3, help='learning rate')

    # for GOAL only
    parser.add_argument('--layer', type=int, default=0, help='the layer index')
        
    return parser.parse_known_args()


def printConfig(args):
    arg2value = {}
    for arg in vars(args):
        arg2value[arg] = getattr(args, arg)
    print(arg2value)


def main():
    args, unknown = parse_args()
    printConfig(args)
    #print(args.nb_epochs)
    ##print(args.ft_size)
    #print(args.dataset)
    #print(args.p_drop)
    model = GOAL(args)
    t = dataset(args)
    #print(type(t.edge_list))
    #print(model.idx_test.shape)
    AUC, ap, hits = model.train()
    print("Average-precision:", np.mean(ap), np.std(ap))
    print("Average-AUC:", np.mean(AUC), np.std(AUC))
    #print(model.embeds)
    #print(np.eye(model.embeds.shape[0]))
    #c = pairwise.cosine_similarity(model.embeds) - np.eye(model.embeds.shape[0])
    #print(c)
    #model.train()
    #model.evaluate()


if __name__ == '__main__':
    main()


