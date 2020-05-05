import torch
from torch import nn
import torch.nn.functional as F
from Util import load_weights
import torch
# from pytorch_transformers import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import Constants


class StructuredSelfAttention(torch.nn.Module):
    """
    The class is an implementation of the paper A Structured Self-Attentive Sentence Embedding including regularization
    and without pruning. Slight modifications have been done for speedup
    https://github.com/kaushalshetty/Structured-Self-Attention
    """

    def __init__(self, config):
        vocab_size = config["vocab_size"]
        use_pretrained_embeddings = config["use_pretrained_embeddings"]
        super(StructuredSelfAttention, self).__init__()
        self.use_bert = config["use_bert"]
# multipal lstm layers
        # self.lstm = torch.nn.LSTM(config["emb_dim"], config["lstm_hid_dim"], batch_first=True, bidirectional=True)
        # self.linear_first = torch.nn.Linear(config["lstm_hid_dim"] * 2, config["d_a"])
        self.gru=torch.nn.GRU(config["emb_dim"], config["lstm_hid_dim"], batch_first=True, bidirectional=True)
        # self.gru = torch.nn.GRU(config["emb_dim"], config["lstm_hid_dim"], batch_first=True, bidirectional=False)
        # self.lstm = torch.nn.LSTM(config["emb_dim"], config["lstm_hid_dim"], batch_first=True, bidirectional=False)
        # self.lstm = torch.nn.LSTM(config["emb_dim"], config["lstm_hid_dim"], batch_first=True, bidirectional=True)
        self.linear_first = torch.nn.Linear(2*config["lstm_hid_dim"] , config["d_a"])

        self.r = config["r"]  # =1 只取句向量
        self.linear_second = torch.nn.Linear(config["d_a"], self.r)
        self.dropout = torch.nn.Dropout(0.1)
        if self.use_bert:
            self.bert_model = BertModel.from_pretrained('bert-base-chinese').eval().to(device)  #eval
        else:
            self.embeddings = self._load_embeddings(config, use_pretrained_embeddings, vocab_size, config["emb_dim"])
            self.embeddings.requires_grad = False

    def get_bert_features(self, input_tensor):
        with torch.no_grad():
            last_hidden_states = self.bert_model(input_tensor)
            last_hidden_states=last_hidden_states[0]  # Models outputs are now tuples
        return last_hidden_states  # batch*seq*768

    def _load_embeddings(self, config, use_pretrained_embeddings, vocab_size, emb_dim):
        """Load the embeddings based on flag"""
        weights = load_weights(config["word2index"], "data/char_vector.txt")

        if not use_pretrained_embeddings:
            word_embeddings = torch.nn.Embedding(vocab_size, emb_dim, padding_idx=Constants.PAD)

        elif use_pretrained_embeddings:
            word_embeddings = torch.nn.Embedding(weights.size(0), weights.size(1))
            word_embeddings.weight = torch.nn.Parameter(weights)

        return word_embeddings  # weights=wocab_size*emb_dim

    def forward(self, x):  # batch_size*max_len
        x = x.to(device)
        if self.use_bert:
            embeddings = self.get_bert_features(x)
        else:
            embeddings = self.embeddings(x)  # batch_size*max_len*emb_dim
        # return embeddings.sum(1)  # batch*emb_dim

        # outputs, _ = self.lstm(embeddings)  # batch_size*max_len*emb_dim
        outputs,_ = self.gru(embeddings)  # batch_size*max_len*emb_dim
        # last=outputs[:,-1,:].squeeze()
        # return last

        # outputs, (hiddens,cells) = self.lstm(embeddings)  # batch_size*max_len*emb_dim  #10*256# outputs batch_size*max_len*lstm_hid_dim
        # x = F.tanh(self.linear_first(self.dropout(outputs)))  # batch_size*max_len*d_a

        x = torch.tanh(self.linear_first(outputs))  # batch_size*max_len*d_a
        x = self.linear_second(x)  # batch_size*max_len*r
        # x = self.softmax(x, 1) #batch*seq*64
        x = F.softmax(x, dim=1)
        attention = x.transpose(1, 2)  # batch_size*r*max_len
        sentence_embeddings = attention @ outputs  # batch_size*r*lstm_hid_dim
        avg_sentence_embeddings = torch.sum(sentence_embeddings, 1) / self.r  # batch_size*lstm_hid_dim  # 不如让r=1
        # return F.log_softmax(avg_sentence_embeddings)
        return avg_sentence_embeddings  # batch*128


class Classifier(nn.Module):
    def __init__(self, vocab_size, n_labels):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(vocab_size, n_labels)

    def forward(self, vector):
        return F.log_softmax(self.linear(vector), dim=1);

class TEXTCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_filters, filter_sizes, output_dim,
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
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

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
        cat = self.dropout(torch.cat(pooled, dim=1)).squeeze(-1).squeeze(-1)
        # cat = [batch size, n_filters * len(filter_sizes)] [1,2,3,4,5 ngram]
        return self.fc(cat)  # batch_size*n_classes
