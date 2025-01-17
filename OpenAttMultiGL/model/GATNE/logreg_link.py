import torch
import torch.nn as nn


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, edges, edges_neg):
        
        #print("shape:",seq.shape)
        #print("edges:", edges.shape)
        #print("edges_neg:", edges_neg.shape)
        ret = self.fc(seq)
        #print("ret:",ret)
        #print("ret shape:",ret.shape)
        #print("ret info:",ret[edges[:, 0]])
        results = torch.cat(((ret[edges[:, 0]] * ret[edges[:, 1]]).sum(dim=-1),
                             (ret[edges_neg[:, 0]] * ret[edges_neg[:, 1]]).sum(dim=-1)))
        return results

