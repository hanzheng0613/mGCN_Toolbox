import torch
from sklearn.metrics import f1_score
from mGCN_Toolbox.utils.dataset import dataset
from mGCN_Toolbox.utils.process import * 
import datetime
import errno
import os
import pickle
import random
from pprint import pprint

import dgl


from dgl.data.utils import _get_dgl_url, download, get_download_dir
import numpy as np

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average="micro")
    macro_f1 = f1_score(labels, prediction, average="macro")

    return accuracy, micro_f1, macro_f1


def evaluate(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1


def main(args):
    # If args['hetero'] is True, g would be a heterogeneous graph.
    # Otherwise, it will be a list of homogeneous graphs.
    
    t = dataset(args["dataset"])

    if hasattr(torch, "BoolTensor"):
        t.train_mask = t.train_mask.bool()
        t.val_mask = t.val_mask.bool()
        t.test_mask = t.test_mask.bool()

    t.HAN_features = t.HAN_features.to(args["device"])
    t.HAN_labels = t.HAN_labels.to(args["device"])
    t.train_mask = t.train_mask.to(args["device"])
    t.val_mask = t.val_mask.to(args["device"])
    t.test_mask = t.test_mask.to(args["device"])
    print(args["hetero"])
    if args["hetero"]:
        from GCN_Toolbox.model.HAN.model_hetero import HAN
        
        model = HAN(
            meta_paths=[["pa", "ap"], ["pf", "fp"]],
            in_size=t.HAN_features.shape[1],
            hidden_size=args["hidden_units"],
            out_size=t.HAN_num_classes,
            num_heads=args["num_heads"],
            dropout=args["dropout"],
        ).to(args["device"])
        t.gs = t.gs.to(args["device"])
        
    else:
        from mGCN_Toolbox.model.HAN.model import HAN

        model = HAN(
            num_meta_paths=len(t.gs),
            in_size=t.HAN_features.shape[1],
            hidden_size=args["hidden_units"],
            out_size=t.HAN_num_classes,
            num_heads=args["num_heads"],
            dropout=args["dropout"],
        ).to(args["device"])
        t.gs = [graph.to(args["device"]) for graph in t.gs]

    stopper = EarlyStopping(patience=args["patience"])
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"]
    )

    for epoch in range(args["num_epochs"]):
        model.train()
        logits = model(t.gs, t.HAN_features)
        loss = loss_fcn(logits[t.train_mask], t.HAN_labels[t.train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc, train_micro_f1, train_macro_f1 = score(
            logits[t.train_mask], t.HAN_labels[t.train_mask]
        )
        val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(
            model, t.gs, t.HAN_features, t.HAN_labels, t.val_mask, loss_fcn
        )
        early_stop = stopper.step(val_loss.data.item(), val_acc, model)

        print(
            "Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | "
            "Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}".format(
                epoch + 1,
                loss.item(),
                train_micro_f1,
                train_macro_f1,
                val_loss.item(),
                val_micro_f1,
                val_macro_f1,
            )
        )

        if early_stop:
            break

    stopper.load_checkpoint(model)
    test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(
        model, t.gs, t.HAN_features, t.HAN_labels, test_mask, loss_fcn
    )
    print(
        "Test loss {:.4f} | Test Acc {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}".format(
            test_loss.item(), test_acc, test_micro_f1, test_macro_f1
        )
    )


if __name__ == "__main__":
    import argparse

    #from utils import setup

    parser = argparse.ArgumentParser("HAN")
    parser.add_argument("-s", "--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "-ld",
        "--log-dir",
        type=str,
        default="results",
        help="Dir for saving training results",
    )
    parser.add_argument(
        "--hetero",
        action="store_true",
        help="Use metapath coalescing with DGL's own dataset",
    )
    parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
    args = parser.parse_args().__dict__

    args = setup(args)

    main(args)