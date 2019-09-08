__all__ = [
    "ABLSTMText"
]

import torch
import torch.nn as nn
import torch.nn.functional as F  

from ..core.const import Const as C
from ..core.utils import seq_len_to_mask
from ..modules import encoder
from ..embeddings import embedding



class ABLSTMText(torch.nn.Module):
    """
    别名：:class:`fastNLP.models.ABLSTMText`  :class:`fastNLP.models.ablstm_text_classification.ABLSTMText`

    使用ABLSTM进行文本分类的模型
    'Peng Zhou. 2016. Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classiﬁcation'
    
    :param tuple(int,int),torch.FloatTensor,nn.Embedding,numpy.ndarray init_embed: Embedding的大小(传入tuple(int, int),
        第一个int为vocab_zie, 第二个int为embed_dim); 如果为Tensor, Embedding, ndarray等则直接使用该值初始化Embedding
    :param int num_classes: 一共有多少类
    :param int hidden_dim: LSTM隐藏层输出维度
    :param int num_layers: LSTM隐藏层层数
    :param float dropout: Dropout的大小
    """

    def __init__(self, init_embed,
                 num_classes,
                 hidden_dim=150, 
                 num_layers=1,
                 dropout=0.5):
        super(ABLSTMText, self).__init__()
        

        self.embed = embedding.Embedding(init_embed)
        #self.embed = encoder.StaticEmbedding(vocab, model_dir_or_name='en-word2vec-300', requires_grad=True)
        self.lstm = nn.LSTM(self.embed.embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),nn.ReLU(inplace=True))

    def attention_net_with_w(self, lstm_out):
        '''
        实现attention结构
        
        :param lstm_out: [batch_size, time_step(output dim of LSTM), hidden_dims * num_directions(=2)]
        :return: dict of torch.LongTensor, [batch_size, hidden_dim] 
        '''
        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)   # ->[N,L,2*C]
        h = lstm_tmp_out[0] + lstm_tmp_out[1]   # ->[N,L,C]
        atten_w = self.attention_layer(h)  # ->[N,L,C] 
        m = nn.Tanh()(h)   # ->[N,L,C] 
        atten_context = torch.bmm(m, atten_w.transpose(1, 2)) # ->[N,L,L] 
        softmax_w = F.softmax(atten_context, dim=-1)  # ->[N,L,L] 
        context = torch.bmm(h.transpose(1,2), softmax_w)  # ->[N,C,L] 
        context_with_attn = h.transpose(1, 2) + context   # 实现一个残差结构
        result = torch.sum(context_with_attn, dim=-1)  # ->[N,C]
        return result
        



    def forward(self, words, seq_len=None):
        """

        :param torch.LongTensor words: [batch_size, seq_len]，句子中word的index
        :param torch.LongTensor seq_len:  [batch,] 每个句子的长度
        :return output: dict of torch.LongTensor, [batch_size, num_classes]
        """
        x = self.embed(words)  # [N,L] -> [N,L,C]
        x = torch.transpose(x, 1, 2) # ->[N,C,L]
        if seq_len is not None:
            mask = seq_len_to_mask(seq_len)
            mask = mask.unsqueeze(1)  # B x 1 x L
            x = x.masked_fill_(mask.eq(0), float('-inf')) # ->[N,C,L]
        x = torch.transpose(x, 1, 2) # ->[N,L,C]
        x = self.dropout(x)
        x = torch.transpose(x, 0,1) # ->[L,N,C]
        output, (hidden, cell) = self.lstm(x)
        x = torch.transpose(output, 0,1) # ->[N,L,C]
        atten_out = self.attention_net_with_w(x) # ->[N,C]
        pred = self.fc(atten_out)
        return {C.OUTPUT: pred}



    def predict(self, words, seq_len=None):
        """
        :param torch.LongTensor words: [batch_size, seq_len]，句子中word的index
        :param torch.LongTensor seq_len:  [batch,] 每个句子的长度

        :return predict: dict of torch.LongTensor, [batch_size, ]
        """
        output = self(words, seq_len)
        _, predict = output[C.OUTPUT].max(dim=1)
        return {C.OUTPUT: predict}
    
    