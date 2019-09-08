__all__ = [
    "CLSTMText"
]

import torch
import torch.nn as nn
import torch.nn.functional as F  

from ..core.const import Const as C
from ..core.utils import seq_len_to_mask
from ..modules import encoder
from ..embeddings import embedding
from ..embeddings import StaticEmbedding

class CLSTMText(torch.nn.Module):
    """
    别名：:class:`fastNLP.models.CLSTMText`  :class:`fastNLP.models.clstm_text_classification.CLSTMText`

    使用C-LSTM进行文本分类的模型
    'Chunting Zhou. 2015. A C-LSTM Neural Network for Text Classiﬁcation.'
    
    :param tuple(int,int),torch.FloatTensor,nn.Embedding,numpy.ndarray init_embed: Embedding的大小(传入tuple(int, int),
        第一个int为vocab_zie, 第二个int为embed_dim); 如果为Tensor, Embedding, ndarray等则直接使用该值初始化Embedding
    :param int num_classes: 一共有多少类
    :param int,tuple(int) out_channels: 输出channel的数量。如果为list，则需要与kernel_sizes的数量保持一致
    :param int,tuple(int) kernel_sizes: 输出channel的kernel大小。
    :param int hidden_dim: LSTM隐藏层输出维度
    :param int num_layers: LSTM隐藏层层数
    :param float dropout: Dropout的大小
    """

    def __init__(self, init_embed,
                 num_classes,
                 kernel_nums=150,
                 kernel_sizes=3, 
                 hidden_dim=150, 
                 num_layers=1,
                 dropout=0.5):
        super(CLSTMText, self).__init__()

        self.embed = embedding.Embedding(init_embed)
        #self.embed = StaticEmbedding(vocab, model_dir_or_name='en-word2vec-300', requires_grad=True)
        self.convs = nn.Conv1d(in_channels=self.embed.embedding_dim,out_channels=kernel_nums,kernel_size=kernel_sizes)   
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(kernel_nums, hidden_dim, num_layers=num_layers, bidirectional=False, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

        

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

        x = self.convs(x)  # ->[N,C,L] 
        #x = F.relu(x)
        x = self.dropout(x)
        x = torch.transpose(x, 0, 1) # ->[C,N,L]
        x = torch.transpose(x, 0, 2) # ->[L,N,C]
        output, (hidden, cell) = self.lstm(x)  
        x = output[-1]  # ->[N,C]
        x = self.dropout(x)
        pred = self.fc(x.squeeze(0))
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
    

