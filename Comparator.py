import torch
from torch import nn
from Util import *
import torch.nn.functional as F
import math

dtype = torch.FloatTensor

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""

    def __init__(self, input_size, hidden_size):
        super(RelationNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, a, b):  # 475*100  #valid 25*100

        n_examples = a.size()[0] * a.size()[1]
        a = a.reshape(n_examples, -1)
        b = b.reshape(n_examples, -1)
        cosine = F.cosine_similarity(a, b)
        return cosine

        # not converage
        x = torch.cat((a, b), 1)
        x = F.relu(self.fc1(x))  # hiddensize->hiddensize
        x = F.sigmoid(self.fc2(x))  # hiddensize->scale
        return x

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_filters, filter_sizes, hidden_size,
                 dropout, pad_idx):
        super().__init__()
        self.filter_sizes = filter_sizes
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=Constants.PAD)
        # self.embedding.weight.data.uniform_(-1, 1)
        # self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=Constants.PAD)
        # self.covlayer=nn.ModuleList
        self.convs = [nn.Conv2d(in_channels=1, out_channels=n_filters,
                                kernel_size=(filter_size, embed_dim), bias=True)
                      for filter_size in filter_sizes]  # [3,4,5]
        # self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, text):  # batch_size*seq_len
        seq_len = text.shape[1]
        # text = [sent len, batch size]
        # text = text.permute(1, 0)
        # text = [batch size, sent len]
        embedded = self.embedding(text)  # batch_size*se1_len*emb_dim
        # embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)  # batch_size*1*seq_len#emb_dim
        # embedded = [batch size, 1, sent len, emb dim]
        pooled = []
        for i in range(len(self.filter_sizes)):
            conved = F.relu(self.convs[i](embedded))  # batch_size*out_channels*(seq_len-2)*1
            conved = F.max_pool2d(conved, (seq_len - self.filter_sizes[i] + 1, 1))  # batch_size*out_channels*1*1
            pooled.append(conved)

        # conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]  # batch_size*n_filters*[18,17,16]
        # # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        # pooled = [F.max_pool2d(conv,()) for conv in conved]  # 提取最显著的词
        # pooled_n = [batch size, n_filters]
        # cat = self.dropout(torch.cat(pooled, dim=1)).squeeze(-1).squeeze(-1)
        # cat = [batch size, n_filters * len(filter_sizes)]
        cat = torch.cat(pooled, dim=1).squeeze(-1).squeeze(-1)
        return cat



class Classifier(nn.Module):
    def __init__(self, vocab_size, n_labels):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(vocab_size, n_labels)

    def forward(self, vector):
        return F.log_softmax(self.linear(vector), dim=1);