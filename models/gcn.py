import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.nn import GCNConv, global_max_pool as gmp
from .gnn_layers import get_simple_gnn_layer, EDGE_GNN_TYPES
from einops import repeat
from .layers_gcn import StructureExtractor




# GCN based model
class GCNNet(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=128,num_features_xd=78, num_features_xt=25, output_dim=128, dropout=0.1):

        super(GCNNet, self).__init__()


        # SMILES graph branch
        self.n_output = n_output
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd )
        self.conv3 = GCNConv(num_features_xd, num_features_xd)
        self.fc_g1 = torch.nn.Linear(num_features_xd, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)





        # protein sequence branch (1d conv)
        # protein sequence branch (1d conv)cnn（3次卷积）+transformer
        self.p_embed = nn.Embedding(num_features_xt + 1, embed_dim)  # (26,128)
        # target：512，1000 ，512，1000，128
        self.p_conv1 = nn.Conv1d(1000, 1000, kernel_size=3, padding=1, stride=1)
        # 输入的通道数为1000，输出的通道数为1000，卷积核的大小为3，添加到输入两侧是1，卷积的步幅是1.
        self.p_bn1 = nn.BatchNorm1d(1000)  # 归一化函数，需要归一化的维度是1000
        # 512，1000，128
        self.p_conv2 = nn.Conv1d(1000, 1000, kernel_size=3, padding=1, stride=1)
        self.p_bn2 = nn.BatchNorm1d(1000)

        self.p_conv3 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.p_bn3 = nn.BatchNorm1d(32)
        # 512，32，121
        # Transformer的预处理：
        self.p_fc1 = nn.Linear(121, 128)  # 输入神经元个数为121，输出神经元个数是128
        # 512，32，128
        self.cnn_attn = TransformerEncoder(d_model=128, n_head=8,
                                           nlayers=3)  ##encoder部分的层数为3层，embedding维度为128，多头注意力为8个头
        # 512，32，128
        # Q,K,V

        # 512，32，16

        self.p_fc2 = nn.Linear(32 * 128, output_dim)
        self.p_bn4 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        # combined layers.py
        self.com_fc1 = nn.Linear(2 * output_dim, 1024)
        self.com_bn1 = nn.BatchNorm1d(1024)
        self.com_fc2 = nn.Linear(1024, 512)
        self.com_bn2 = nn.BatchNorm1d(512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data,return_attn=False):#前向传播
        print("forward")
        # get graph input
        x, edge_index, batch,edge_attr = data.x, data.edge_index, data.batch,data.edge_attr
        # get protein input
        target = data.target
        #药物进行向前传播
        x = self.conv1(x, edge_index)
        x = self.relu(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)  # global max pooling

        # flatten
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)




#蛋白质进行前向传播
        # 1d conv layers
        p_embed = self.p_embed(target)  # (8,1000)-->(8,1000,128)
        # target: 512,1000--> 512,1000,128
        protein = self.p_conv1(p_embed)  # 8,1000,128
        protein = self.p_bn1(protein)  # 8，1000，128
        protein = self.relu(protein)  # 8，1000，128

        protein = self.p_conv2(protein)  # 8，1000，128
        protein = self.p_bn2(protein)  # 8，1000，128
        protein = self.relu(protein)

        protein = self.p_conv3(protein)  # 8，32，121因为卷积核是8，步长是1，所以最后是128-7=121
        protein = self.p_bn3(protein)  # 8，32，121
        protein = self.relu(protein)  # 8，32，121

        protein = self.p_fc1(protein)  # 8，32，128
        protein = self.relu(protein)  # 8，32，128
        protein = self.dropout(protein)  # 8，32，128

        protein = self.cnn_attn(protein)  # 进行了transformerencoder进行得到。#8,32,128
        # 512,32,128

        # flatten
        protein = protein.view(-1, 32 * 128)  # 32*128维，-1代表动态调整这个维度上的元素个数，以保证元素的总数不变。#8,4096
        protein = self.p_fc2(protein)  # 8,128
        protein = self.relu(protein)
        protein = self.p_bn4(protein)
        protein = self.dropout(protein)

        # 512,128

        # concat
        xc = torch.cat((x, protein), 1)  # 8*128，8*128
        # 8*256
        xc = self.com_fc1(xc)  # 8*1024
        xc = self.relu(xc)  # 8*1024
        xc = self.com_bn1(xc)  # 8*1024
        xc = self.dropout(xc)  # 8*1024

        xc = self.com_fc2(xc)  # 8*512
        xc = self.relu(xc)
        xc = self.com_bn2(xc)
        xc = self.dropout(xc)

        out = self.out(xc)  # 8*1
        return out
        # 多模态
        # fasta_common=self.fas_mlp(protein)
        # out=self.premodel(smi_common,fas_common)

        # return fasta_common,sim_common

    #

    #####
    # by xmm
    # Attention(Q,K,V) = softmax(Q*Kt/sqrt(dk)) *V
class ScaledDotProductAttention(nn.Module):  # Attention(Q,K,V) = softmax(Q*Kt/sqrt(dk)) *V
    """
          Compute 'Scaled Dot Product Attention'
          Attention(Q,K,V) = softmax(Q*Kt/sqrt(dk)) *V
          """
    """ for test 
            q = torch.randn(4, 8, 10, 64)  # (batch, n_head, seqLen, dim)
            k = torch.randn(4, 8, 10, 64)
            v = torch.randn(4, 8, 10, 64)
            mask = torch.ones(4, 8, 10, 10)
            model = ScaledDotProductAttention()
            res = model(q, k, v, mask)
            print(res[0].shape)  # torch.Size([4, 8, 10, 64])
    """

    def forward(self, query, key, value, attn_mask=None, dropout=None):
        """
                    当QKV来自同一个向量的矩阵变换时称作self-attention;
                    当Q和KV来自不同的向量的矩阵变换时叫soft-attention;
                    url:https://www.e-learn.cn/topic/3764324
                    url:https://my.oschina.net/u/4228078/blog/4497939
                      :param query: (batch, n_head, seqLen, dim)  其中n_head表示multi-head的个数，且n_head*dim = embedSize
                      :param key: (batch, n_head, seqLen, dim)
                      :param value: (batch, n_head, seqLen, dim)
                      :param mask:  (batch, n_head, seqLen,seqLen) 这里的mask应该是attn_mask；原来attention的位置为0，no attention部分为1
                      :param dropout:
                      """

            # (batch, n_head, seqLen,seqLen) attention weights的形状是L*L，因为每个单词两两之间都有一个weight
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)  # 保留位置为0的值，其他位置填充极小的数

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn  # (batch, n_head, seqLen, dim)

    # by xmm
class MultiHeadAttention(nn.Module):
    """
           for test :
                       q = torch.randn(4, 10, 8 * 64)  # (batch, n_head, seqLen, dim)
                       k = torch.randn(4, 10, 8 * 64)
                       v = torch.randn(4, 10, 8 * 64)
                       mask = torch.ones(4, 8, 10, 10)
                       model = MultiHeadAttention(h=8, d_model=8 * 64)
                       res = model(q, k, v, mask)
                       print(res.shape)  # torch.Size([4, 10, 512])
           """

    def __init__(self, h, d_model, dropout=0.1):  # (8,128,0.1)
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

            # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()  # 获得注意力矩阵

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, attn_mask=None):  # (protein,protein,protein,atten_mask=None)
        """
                   :param query: (batch,seqLen, d_model)
                   :param key: (batch,seqLen, d_model)
                   :param value: (batch,seqLen, d_model)
                   :param mask: (batch, seqLen,seqLen)
                   :return: (batch,seqLen, d_model)
                   """
        batch_size = query.size(0)

            # 1, Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                                for l, x in
                                zip(self.linear_layers, (query, key, value))]  # （k：8，6，32，16；q：8，8，32，16；v：8，8，32，16）

            # 2,Apply attention on all the projected vectors in batch.
            # if attn_mask:
            #     attn_mask = attn_mask.unsqueeze(1).repeat(1, self.h, 1, 1)  # (batch, n_head,seqLen,seqLen)
        x, atten = self.attention(query, key, value, attn_mask=attn_mask,
                                    dropout=self.dropout)  # 这个X就是公式中的Z，atten就是softmax中的那一坨内积
            # ：x:8,8,32,16
            # 3, "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)  # view函数表示要重新定义矩阵的形状。#x:8,32,128
        return self.output_linear(x)

    # by xmm
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, dim_feedforward, dropout, activation):  # (128,1024,0.1,relu)
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, dim_feedforward)  # 128,1024
        self.w_2 = nn.Linear(dim_feedforward, d_model)  # 1024,128
        self.dropout = nn.Dropout(dropout)  # 0.1
        self.activation = activation  # relu

    def forward(self, x):  # protein
        xx = self.dropout(self.w_2(self.activation(self.w_1(x))))  # 8,32,128
        return xx  # 线性操作，激活函数，线性操作，再拟合

    # by xmm
class aTransformerEncoderLayer(nn.Module):


    def __init__(self, d_model, n_head, dim_feedforward, dropout=0.1, activation="relu"):  # (128,8,1024,0.1,relu)
        """
                   :param d_model:
                   :param n_head:
                   :param dim_feedforward:
                   :param dropout:
                   :param activation: default :relu
                   """

        super().__init__()
        self.self_attn = MultiHeadAttention(h=n_head, d_model=d_model, dropout=dropout)  # (8,128,0.1)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        if activation == "relu":
            self.activation = F.relu
        if activation == "gelu":
            self.activation = F.gelu

        self.PositionwiseFeedForward = PositionwiseFeedForward(d_model=d_model, dim_feedforward=dim_feedforward,
                                                                   dropout=dropout, activation=self.activation)
            # (128,1024,0.1,relu)

    def forward(self, x, atten_mask):  # (protein,none)
        """
                  :param x: (batch, seqLen, em_dim)
                  :param mask: attn_mask
                  :return:
                  """
            # add & norm 1
        attn = self.dropout(
            self.self_attn(x, x, x, attn_mask=atten_mask))  # (protein,protein,protein,none)#8,32,128
        x = self.norm1((x + attn))  # 残差连接和归一化处理

            # # add & norm 2
        x = self.norm2(x + self.PositionwiseFeedForward(x))  # x

        return x

class TransformerEncoder(nn.Module):
    """
            Example:
                   x = torch.randn(4, 10, 128)  # (batch, seqLen, em_dim)
                model = TransformerEncoder(d_model=128, n_head=8, nlayers=3)
                res = model.forward(x)
                print(res.shape)  # torch.Size([4, 10, 128])
            """

        ##(128,8,3,1024,0.1,relu)
    def __init__(self, d_model, n_head, nlayers, dim_feedforward=1024, dropout=0.1, activation="relu"):
        super(TransformerEncoder, self).__init__()
        self.encoder = nn.ModuleList(
            [aTransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout, activation)
                for _ in range(nlayers)])  # (128,8,1024,0.1,relu)

    def forward(self, x, atten_mask=None):  # (protein,none)
        """:param x: input dim == out dim
                   :param atten_mask: 对应源码的src_mask，没有实现src_key_padding_mask
                   :return:
                   """
        for layer in self.encoder:
            x = layer.forward(x, atten_mask)
        return x


