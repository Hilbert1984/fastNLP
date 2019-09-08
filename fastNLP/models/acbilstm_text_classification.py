__all__ = [
    "ACBiLSTMText"
]

import torch
import torch.nn as nn
import torch.nn.functional as F  

from ..core.const import Const as C
from ..core.utils import seq_len_to_mask
from ..modules import encoder
from ..embeddings import embedding


class ACBiLSTMText(torch.nn.Module):
    """
    别名：:class:`fastNLP.models.CLSTMText`  :class:`fastNLP.models.clstm_text_classification.CLSTMText`

    使用C-LSTM进行文本分类的模型
    'Chunting Zhou. 2015. A C-LSTM Neural Network for Text Classiﬁcation.'
    
    :param tuple(int,int),torch.FloatTensor,nn.Embedding,numpy.ndarray init_embed: Embedding的大小(传入tuple(int, int),
        第一个int为vocab_zie, 第二个int为embed_dim); 如果为Tensor, Embedding, ndarray等则直接使用该值初始化Embedding
    :param int num_classes: 一共有多少类
    :param int,tuple(int) out_channels: 输出channel的数量。如果为list，则需要与kernel_sizes的数量保持一致
    :param int,tuple(int) kernel_sizes: 输出channel的kernel大小。
    :param int hidden_dim:
    :param int num_layers:
    :param float dropout: Dropout的大小
    """

    def __init__(self, init_embed,
                 num_classes,
                 kernel_nums=50,
                 kernel_sizes=3, 
                 hidden_dim=64, 
                 num_layers=2,
                 dropout=0.5):
        super(ACBiLSTMText, self).__init__()

        self.embed = embedding.Embedding(init_embed)
        
        self.convs = nn.Conv1d(in_channels=self.embed.embedding_dim,out_channels=kernel_nums,kernel_size=kernel_sizes)  
        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(kernel_nums, hidden_dim, num_layers=num_layers, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def attention(self, lstm_out):
        '''
        :param lstm_out: [batch_size, time_step, hidden_dims * num_directions(=2)]
        :return:
        '''
        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)
        # h [batch_size, time_step, hidden_dims]
        h = lstm_tmp_out[0] + lstm_tmp_out[1]
        # atten_w [batch_size, time_step, hidden_dims]
        atten_w = self.attention_layer(h)
        # m [batch_size, time_step, hidden_dims]
        m = nn.Tanh()(h)
        # atten_context [batch_size, time_step, time_step]
        atten_context = torch.bmm(m, atten_w.transpose(1, 2))
        # softmax_w [batch_size, time_step, time_step]
        softmax_w = F.softmax(atten_context, dim=-1)
        # context [batch_size, hidden_dims, time_step]
        context = torch.bmm(h.transpose(1,2), softmax_w)
        context_with_attn = h.transpose(1, 2) + context
        # result [batch_size, hidden_dims]
        # result = torch.sum(context, dim=-1)
        result = torch.sum(context_with_attn, dim=-1)
        return result

    def forward(self, words, seq_len=None):
        """

        :param torch.LongTensor words: [batch_size, seq_len]，句子中word的index
        :param torch.LongTensor seq_len:  [batch,] 每个句子的长度
        :return output: dict of torch.LongTensor, [batch_size, num_classes]
        """
        x = self.embed(words)  # [N,L] -> [N,L,C]
        x = torch.transpose(x, 1, 2) # [N,C,L]
        x = self.convs(x)  #[N,C,L] [2,30,23]
        x = F.relu(x)
        x = self.dropout(x)
        x = torch.transpose(x, 0, 1) # [C,N,L]
        x = torch.transpose(x, 0, 2) # [L,N,C]
        output, (hidden, cell) = self.lstm(x)
        output = torch.transpose(output,0,1)
        res = self.attention(output)
        # hidden: (batch_size, hidden_dim * 2)

        pred = self.fc(res)
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
    

