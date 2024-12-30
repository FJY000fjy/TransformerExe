# %%
# code by Tae Hwan Jung(Jeff Jung) @graykode, Derek Miller @dmmiller612
# Reference : https://github.com/jadore801120/attention-is-all-you-need-pytorch
#           https://github.com/JayParks/transformer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps

def make_batch(sentences):
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
    return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)

def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)


## 4.get_attn_pad_mask
### 比如说，我现在句子长度是5，在后面注意力机制的部分，
## 我们在计算出来QK转置除以根号之后，softmax之前，我们得到的形状
## len_input * len * input代表每个单词对其余包含自己的单词的影响力

## 所以这里我需要一个同等大小形状的矩阵，告诉我哪个位置是PAD部分，
## 之后在计算softmax之前会把这里置为无穷大：
## 一定要需要注意的是这里得到的矩阵形状是batch_size * len_q * len_k,
## 我们是对k中的pad符号进行标识，并没有对k中的做标识，因为没必要
## seq_q和seq_k不一定一致，在交互注意力，q来自解码器，k来自编码器，
## 所以告诉模型编码这边pad符号信息就可以，解码端的pad信息在交互注意力层是没有用到的
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask

## 7.ScaledDotProductAttention
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        ## 输入进来的维度分别是[batch_size * n_heads * len_q * d_k]
        ## K:[batch_size * n_heads * len_k * d_k]
        ## V:[batch_size * n_heads * len_k * d_v]
        ## 首先经过matmul函数得到的scores形状是：[batch_size * n_heads * len_q * len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        ## 然后关键词地方来了，下面这个就是用到了我们之前重点讲的attn_mask,
        ## 把被mask的地方置为无限小，softmax之后基本就是0，对q的单词不起作用
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        ## 输入进来的QKV是相等的，我们会使用映射linear做一个映射得到参数矩阵Wq,Wk,Wv
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):

        ## 这个多头分为这几个步骤，首先映射分头，然后计算atten_scores,然后计算atten_value
        ## 输入进来的数据形状：
        # Q: [batch_size * len_q * d_model],
        # K: [batch_size * len_k * d_model],
        # V: [batch_size * len_k * d_model]
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        ## 下面这个就是先映射，后分头;一定要注意q和k分头之后维度是一致的，所以一看这里都是dk
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        ## 输入进行的attn_mask形状是 batch_size * len_q * len_k,
        ## 然后经过下面这个代码得到新的attn_mask:[batch_size * n_heads * len_q * len_k]
        ## 就是把pad信息重复到了n个头上
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        ## 然后我们计算 ScaleDotProductAttention 这个函数，去7.看一下
        ## 得到的结果有两个：context:[batch_size * n_heads * len_q * d_v]
        ## attn: [batch_size * n_heads * len_q * len_k]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)
        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]

# 8.PoswiseFeedForwardNet
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)

## 5. EncoderLayer :包含两个部分，多头注意力机制和前馈神经网络
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        ## 下面这个就是做自注意力层，输入是enc_inputs,形状是[batch_size * seq_len_q * d_model]
        ## 最初始的QKV矩阵是等同于这个输入的，去看一下enc_self_attn函数 6.
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn

## 10.
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn

## 2.Encoder 部分包含三个部分：词向量embedding,位置编码部分，注意力层及后续的前馈神经网络
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # 词向量嵌入层
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        # 位置编码层
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(src_len+1, d_model),freeze=True)
        # 多个编码器层
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    # 先将输入的源语言序列进行词向量嵌入并添加位置编码，然后通过循环调用各个编码器层依次对输入进行处理
    # 最终返回编码后的输出以及每层的自注意力权重矩阵集合
    # 这些输出将作为解码器的部分输入以及后续分析编码特征的依据
    def forward(self, enc_inputs): # enc_inputs : [batch_size x source_len]
        ## 这里我们的enc_inputs 形状是：[batch_size * source_len]
        ## 下面这个代码通过src_emb,进行索引定位，enc_outputs输出形状是[batch_size,src_len,d_model]
        enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(torch.LongTensor([[1,2,3,4,0]]))
        ## get_attn_pad_mask是为了得到句子中pad的位置，给到模型后面，
        # 在计算自注意力和交互注意力的时候去掉pad符号的影响，去看一下这个函数 4.
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        # 各层自注意力权重矩阵
        enc_self_attns = []
        for layer in self.layers:
            ## 去看EncoderLayer 层函数 5.
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

## 9.Decoder
# 整体的解码器模块，与编码器类似，包含词向量嵌入层（tgt_emb），位置编码（pos_emb）以及多个解码器层（layers）
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(tgt_len+1, d_model),freeze=True)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    # 在forward方法中，先对输入的目标语言序列进行词向量嵌入和位置向量编码，
    # 接着构建多种掩码矩阵，先对输入的目标语言序列进行词向量嵌入和位置编码
    # 接着构建多种掩码矩阵（自注意力的PAD掩码、后续位置掩码以及与编码器交互注意力的PAD掩码）
    # 然后通过循环调用各个解码器层依次处理输入，最终返回解码后的输出以及对应的自注意力和交互注意力矩阵权重集合
    def forward(self, dec_inputs, enc_inputs, enc_outputs): # dec_inputs : [batch_size x target_len]
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(torch.LongTensor([[5,1,2,3,4]]))
        ## get_attn_pad_mask 自注意力层的时候的pad部分
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        ## get_attn_subsequent_mask 这个做的是自注意层的mask部分，
        # 就是当前单词之后看不到，使用一个上三角为1的矩阵
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        ## 两个矩阵相加，大于0的为1，不大于0的为0，为1的在之后就会被fill到无限小
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        ## 这个做的是交互注意力机制中的mask矩阵，enc的输入是k
        ## 我去看这个k里面哪些是pad符号，给到后面的模型
        ## 注意哦，我q肯定也是由pad符号，但是这里我不在意的，之前说了好多次了哈
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

## 1.从整体网络结构来看，分为三个部分：编码层，解码层，输出层
# 定义了整个Transformer模型的结构，整合了编码器（encoder）、解码器（decoder）以及最后的输出投影层（projection）
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder() ## 编码层
        self.decoder = Decoder() ## 解码层
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    # 在forward方法中，先将源语言输入传入编码器获取编码表示，
    # 再将目标语言输入、源语言输入和编码器输出传入解码器获取解码输出，
    # 最后通过投影层将解码输出映射到目标语言的词表维度大小，
    # 返回预测的对数概率（dec_logits）以及编码器自注意力、
    # 解码器自注意力和解码器与编码器交互注意力的权重矩阵，
    # 用于后续的损失计算、预测以及可视化分析等操作
    def forward(self, enc_inputs, dec_inputs):
        ## 这里有两个数据进行输入，一个是enc_inputs 形状为[batch_size,src_len],
        ## 主要是作为编码段的输入，一个dec_inputs,形状为[batch_size,tgt_len],主要是作为解码端的输入

        ## enc_inputs作为输入，形状为[batch_size,src_len],
        # 输出由自己的函数内部指定，想要什么指定输出什么，可以是全部tokens的输出，可以是特定每一层的输出
        # 也可以是中间某些参数的输出
        ## enc_outputs就是主要的输出，
        # enc_self_attns这里没记错的是QK转置相乘之后softmax之后的矩阵值
        # 代表的是每个单词和其他单词的相关性
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        ## dec_outputs 是decoder主要输出，用于后续的linear映射
        ## dec_self_attns 类比于enc_self_attns 是查看每个单词对decoder中输入的其余单词的相关性
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)

        ## dec_output做映射到词表大小
        dec_logits = self.projection(dec_outputs) # dec_logits : [batch_size x src_vocab_size x tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns

# 用于可视化注意力权重矩阵。
# 它从输入的注意力权重矩阵列表（如enc_self_attns等）中
# 选取最后一层的第一个头的注意力权重矩阵，进行维度压缩，转换为numpy数组后，
# 利用matplotlib库绘制热力图，将注意力权重以可视化的形式展示出来，
# 横坐标和纵坐标分别标注对应的单词，
# 方便直观查看不同单词之间的注意力关联程度
def showgraph(attn):
    attn = attn[-1].squeeze(0)[0]
    attn = attn.squeeze(0).data.numpy()
    fig = plt.figure(figsize=(n_heads, n_heads)) # [n_heads, n_heads]
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attn, cmap='viridis')
    ax.set_xticklabels(['']+sentences[0].split(), fontdict={'fontsize': 14}, rotation=90)
    ax.set_yticklabels(['']+sentences[2].split(), fontdict={'fontsize': 14})
    plt.show()

if __name__ == '__main__':
    ## 句子的输入部分
    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

    # Transformer Parameters
    # Padding Should be Zero
    ## 构建词表
    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
    src_vocab_size = len(src_vocab)

    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
    number_dict = {i: w for i, w in enumerate(tgt_vocab)}
    tgt_vocab_size = len(tgt_vocab)

    src_len = 5 # length of source
    tgt_len = 5 # length of target
    ## 模型参数
    d_model = 512  # Embedding Size
    d_ff = 2048  # FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_layers = 6  # number of Encoder of Decoder Layer
    n_heads = 8  # number of heads in Multi-Head Attention

    model = Transformer()

    criterion = nn.CrossEntropyLoss() # 损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001) # 优化器

    enc_inputs, dec_inputs, target_batch = make_batch(sentences)

    # 在训练循环中，每次迭代先清零梯度，将输入传入模型获取输出并计算损失，
    # 打印当前轮次的损失值，然后进行反向传播和参数更新
    for epoch in range(20):
        optimizer.zero_grad()
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        loss = criterion(outputs, target_batch.contiguous().view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()

    # Test
    # 训练完成后进行测试，获取预测结果并转换为对应的单词进行打印展示，
    # 最后分别可视化编码器自注意力、解码器自注意力和解码器与编码器交互注意力的权重矩阵
    # 直观呈现模型在注意力方面的表现情况
    predict, _, _, _ = model(enc_inputs, dec_inputs)
    predict = predict.data.max(1, keepdim=True)[1]
    print(sentences[0], '->', [number_dict[n.item()] for n in predict.squeeze()])

    print('first head of last state enc_self_attns')
    showgraph(enc_self_attns)

    print('first head of last state dec_self_attns')
    showgraph(dec_self_attns)

    print('first head of last state dec_enc_attns')
    showgraph(dec_enc_attns)