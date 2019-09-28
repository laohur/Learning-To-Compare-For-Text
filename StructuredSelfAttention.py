import torch
from torch import nn
import torch.nn.functional as F
from Util import load_weights

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
        self.embeddings = self._load_embeddings(config,use_pretrained_embeddings, vocab_size, 300)
        self.embeddings.requires_grad = False
        self.lstm = torch.nn.LSTM(config["emb_dim"], config["lstm_hid_dim"], batch_first=True, bidirectional=True)
        self.linear_first = torch.nn.Linear(config["lstm_hid_dim"]* 2, config["d_a"])
        self.r=config["r"]  #=1 只取句向量
        self.linear_second = torch.nn.Linear(config["d_a"], self.r)
        self.dropout = torch.nn.Dropout(0.1)

    def _load_embeddings(self, config,use_pretrained_embeddings, vocab_size, emb_dim):
        """Load the embeddings based on flag"""
        weights = load_weights(config["word2index"], "data/char_vector.txt")

        if not use_pretrained_embeddings:
            word_embeddings = torch.nn.Embedding(vocab_size, emb_dim, padding_idx=Constants.PAD)

        elif use_pretrained_embeddings:
            word_embeddings = torch.nn.Embedding(weights.size(0), weights.size(1))
            word_embeddings.weight = torch.nn.Parameter(weights)

        return word_embeddings  # weights=wocab_size*emb_dim

    def softmax(self, input, axis=1):
        """
        Softmax applied to axis=n
        Args:
           input: {Tensor,Variable} input on which softmax is to be applied
           axis : {int} axis on which softmax is to be applied
        Returns:
            softmaxed tensors
        """
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def forward(self, x):  # batch_size*max_len
        x = x.to(device)
        embeddings = self.embeddings(x)  # batch_size*max_len*emb_dim
        # return embeddings.sum(1)  #batch*emb_dim
        outputs, _ = self.lstm(embeddings)  # batch_size*max_len*emb_dim  #10*256
        # outputs batch_size*max_len*lstm_hid_dim
        # x = F.tanh(self.linear_first(self.dropout(outputs)))  # batch_size*max_len*d_a
        x = F.tanh(self.linear_first(outputs))  # batch_size*max_len*d_a
        x = self.linear_second(x)  # batch_size*max_len*r
        # x = self.softmax(x, 1) #batch*seq*64
        x=F.softmax(x,dim=1)
        attention = x.transpose(1, 2)  # batch_size*r*max_len
        sentence_embeddings = attention @ outputs  # batch_size*r*lstm_hid_dim
        avg_sentence_embeddings = torch.sum(sentence_embeddings, 1) / self.r # batch_size*lstm_hid_dim  # 不如让r=1
        # return F.log_softmax(avg_sentence_embeddings)
        return avg_sentence_embeddings  #batch*128

    # Regularization
    def l2_matrix_norm(self, m):
        """
        Frobenius norm calculation
        Args:
           m: {Variable} ||AAT - I||
        Returns:
            regularized value
        """
        return torch.sum(torch.sum(torch.sum(m ** 2, 1), 1) ** 0.5).to(device)

class Classifier(nn.Module):
    def __init__(self, vocab_size,n_labels):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(vocab_size, n_labels)

    def forward(self, vector):
        return F.log_softmax(self.linear(vector), dim=1);