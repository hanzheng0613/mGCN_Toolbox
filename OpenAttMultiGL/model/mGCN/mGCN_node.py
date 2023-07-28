from torch_geometric.nn import GCNConv, GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F

class Combine(nn.Module):
    """
    Combine embeddings from different dimensions to generate a general embedding
    """
    def __init__(self, input_len=6, output_size = 3,cuda=False, dropout_rate=0):
        """
        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        """
        super(Combine,self).__init__()
        self.cuda = cuda
        self.input_len = input_len
        self.output_size = output_size
        self.linear_layer = nn.Linear(self.input_len,self.output_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        #print('input_len type: ',type(input_len))
        #print('output_size type : ',type(output_size))
        self.act = nn.ELU()
    def forward(self,dim_embs):

        emb = torch.cat(dim_embs,1)
        emb_combine = self.linear_layer(emb)
        emb_combine_act = self.act(emb_combine)
        #emb_combine_act_drop = self.dropout(emb_combine_act)

        return emb_combine_act

class Attention(nn.Module):
    def __init__(self, input_size, cuda=False):
        super(Attention, self).__init__()

        self.cuda = cuda
        self.bilinear_layer = nn.Bilinear(input_size, input_size, 1)
        self.softmax = nn.Softmax(dim=1)
        if self.cuda:
            self.bilinear_layer.cuda()
            self.softmax.cuda()

    def forward(self, Ws):
        """
        Measuring relations between all the dimensions
        """

        # W = torch.Tensor(output_size,output_size)

        num_dims = len(Ws)

        attention_matrix = torch.empty((num_dims, num_dims), dtype=torch.float)
        if self.cuda:
            attention_matrix = attention_matrix.cuda()
        for i, wi in enumerate(Ws):
            for j, wj in enumerate(Ws):
                # attention_matrix[i,j] = torch.trace(wi.transpose(0,1).mm(W).mm(wj))
                attention_matrix[i, j] = torch.sum(self.bilinear_layer(wi, wj))

        attention_matrix_softmax = self.softmax(attention_matrix)

        return attention_matrix_softmax

class mGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, weight, views, alpha, output_channels, dropout=0.5, self_loop=True):
        super(mGCN, self).__init__()
        self.conv1 = nn.ModuleList()
        self.conv2 = nn.ModuleList()
        self.attentions1 = Attention(in_channels)
        self.attentions2 = Attention(hidden_channels)
        self.comb1 = Combine(hidden_channels* views, hidden_channels)
        self.comb2 = Combine(hidden_channels * views, output_channels)
        self.self_loop = self_loop
        self.act = nn.ELU()
        self.alpha = alpha

        for i in range(0, views):
            self.conv1.append(GCNConv(in_channels, hidden_channels))

        for i in range(0, views):
            self.conv2.append(GCNConv(hidden_channels, hidden_channels))

        self.dropout = nn.Dropout(p = dropout)
        self.weight = weight
        self.views = views

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def get_cross_rep(self, x_inner, x, attentions, last=True):
        x_cross = []
        for i in range(0, len(x)):
            temp = torch.zeros(x[i].shape[0], x[i].shape[1]).to(x[i])
            for j in range(len(x)):
                if self.self_loop:
                    temp = temp + self.alpha*attentions[i,j]*x[j]
                else:
                    if j!=i:
                        temp = temp + self.alpha*attentions[i,j]*x[j]
            x_cross.append(temp)
        if last:
            x_res = [(1-self.alpha) + emb_dim_inner + self.dropout(self.act(emb_dim)) for emb_dim, emb_dim_inner in zip(x_cross, x_inner)]
        else:
            x_res = [(1 - self.alpha) + emb_dim_inner + emb_dim for emb_dim, emb_dim_inner in zip(x_cross, x_inner)]

        return x_res


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x_multi = []
        Ws = []
        x_lin = []
        for i, conv in enumerate(self.conv1):
            x_temp = self.act(conv(x, edge_index[i]))
            x_temp = self.dropout(x_temp)
            x_multi.append(x_temp)
            x_lin.append(conv.lin(x))
            Ws.append(conv.lin.weight)

        attentions = self.attentions1(Ws)
        x_multi = self.get_cross_rep(x_multi, x_lin, attentions)
        x = self.comb1(x_multi)

        x_lin = []
        x_multi = []
        Ws = []
        for i, conv in enumerate(self.conv2):
            x_temp = conv(x, edge_index[i])
            x_multi.append(x_temp)
            x_lin.append(conv.lin(x))
            Ws.append(conv.lin.weight)

        attentions = self.attentions2(Ws)
        x_multi = self.get_cross_rep(x_multi, x_lin, attentions, last=True)
        x = self.comb2(x_multi)
        # if self.weight:
        #     weight = data.graph['weight']
        #     x = F.relu(self.conv1(x, edge_index, weight))
        #     x = F.dropout(x, p=self.dropout, training=self.training)
        #     x = self.conv2(x, edge_index, weight)
        # else:
        #     # x = self.linear(x)
        #     # x = self.conv2(x, edge_index)
        #     x = F.relu(self.conv1(x, edge_index))
        #     x = F.dropout(x, p=self.dropout, training=self.training)
        #     x = self.conv2(x, edge_index)


        return F.log_softmax(x, dim=1)