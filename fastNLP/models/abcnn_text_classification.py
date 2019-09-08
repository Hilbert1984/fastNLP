__all__ = [
    "ABCNNText"
]

import torch
import torch.nn as nn
import torch.nn.functional as F  

from ..core.const import Const as C
from ..core.utils import seq_len_to_mask
from ..modules import encoder
from ..embeddings import embedding


class ABCNNText(torch.nn.Module):
    """
    别名：:class:`fastNLP.models.ABCNNText`  :class:`fastNLP.models.abcnn_text_classification.ABCNNText`

    使用ABCNN进行文本分类的模型
    'Yin W, Schütze H, Xiang B, et al. 2016. Abcnn: Attention-based convolutional neural network for modeling sentence pairs.' 
    
    :param tuple(int,int),torch.FloatTensor,nn.Embedding,numpy.ndarray init_embed: Embedding的大小(传入tuple(int, int),
        第一个int为vocab_zie, 第二个int为embed_dim); 如果为Tensor, Embedding, ndarray等则直接使用该值初始化Embedding
    :param int num_classes: 一共有多少类
    :param int,tuple(int) kernel_nums: 输出channel的数量。如果为list，则需要与kernel_sizes的数量保持一致
    :param int,tuple(int) kernel_sizes: 输出channel的kernel大小。
    :param float dropout: Dropout的大小
    """

    def __init__(self, init_embed,
                 num_classes,
                 kernel_nums=300,
                 kernel_sizes=3, 
                 dropout=0.25):

        super(ABCNNText, self).__init__()

        self.embed = embedding.Embedding(init_embed)
        
        self.convs = nn.Conv1d(in_channels=self.embed.embedding_dim,out_channels=kernel_nums,kernel_size=kernel_sizes)  
        self.attention_layer = nn.Sequential(
            nn.Linear(kernel_nums, kernel_nums),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(kernel_nums, num_classes)
        self.dropout = nn.Dropout(dropout)

    def attention(self, conv_out):
        '''
        :param conv_out: [batch_size, seq_len, kernel_nums]
        :return result: [batch_size, kernel_nums] 
        '''
        atten_w = self.attention_layer(conv_out) # [N, L, C]
        m = nn.Tanh()(conv_out) # [N, L, C]
        atten_context = torch.bmm(m, atten_w.transpose(1, 2)) # [N, L, C]
        softmax_w = F.softmax(atten_context, dim=-1) # [N, C, L]
        context = torch.bmm(conv_out.transpose(1,2), softmax_w)
        context_with_attn = conv_out.transpose(1, 2) + context
        result = torch.sum(context_with_attn, dim=-1) # [N, C]
        return result

    def forward(self, words, seq_len=None):
        """

        :param torch.LongTensor words: [batch_size, seq_len]，句子中word的index
        :param torch.LongTensor seq_len:  [batch,] 每个句子的长度
        :return output: dict of torch.LongTensor, [batch_size, num_classes]
        """
        x = self.embed(words)  # [N,L] -> [N,L,C]
        x = torch.transpose(x, 1, 2) # [N,C,L]
        x = self.convs(x)  #[N,C,L]
        x = F.relu(x)
        x = self.dropout(x)
        x = torch.transpose(x, 1, 2) # [N,L,C]
        res = self.attention(x)

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
    