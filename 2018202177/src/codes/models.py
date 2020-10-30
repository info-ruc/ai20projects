import torch
from torch import nn
import torchvision


class Encoder(nn.Module):
    """
    Encoder
    """
    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_img_size = encoded_image_size

        cnn_ext = torchvision.models.resnet50(pretrained = True)  # 使用预训练的 resnet-50
        modules = list(cnn_ext.children())[:-2]  # 去掉网络中的最后两层，考虑使用 list(cnn_ext.children())
        self.cnn_ext = nn.Sequential(*modules)  # 使用 nn.Sequential 定义好 encoder

        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))  # 使用 nn.AdaptiveAvgPool2d 将输出改变到指定的大小

    def forward(self, img):
        out = self.cnn_ext(img)  # [bs, 2048, 8, 8]
        out = self.adaptive_pool(out)  # [bs, 2048, enc_img_size, enc_img_size]
        out = out.permute(0, 2, 3, 1)  # [bs, enc_img_size, enc_img_size, 2048]
        return out

    def freeze_params(self, freeze):
        for p in self.cnn_ext.parameters():
            p.requires_grad = False

        for c in list(self.cnn_ext.children())[5:]:
            for p in c.parameters():
                p.requires_grad = (not freeze)


class AttentionModule(nn.Module):
    """
    Attention Module with Decoder
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: 图片经过 Encoder 之后的特征维度
        :param decoder_dim: 解码器隐含状态 h 的维度
        :param attention_dim: attention的维度
        """
        super(AttentionModule, self).__init__()

        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # Linear, encoder_dim -> attention_dim
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # Linear, decoder_dim -> attention_dim
        self.relu = nn.ReLU()  # relu 激活函数
        self.softmax = nn.Softmax(dim=1)  # softmax 激活函数, dim=1

    def forward():
        """
        注意力机制的前向传播过程
        待完成
        """

class Decoder(nn.Module):

    def __init__(self, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        需要加入attention机制（待完成）
        :params embed_dim: 词向量的维度
        :params decoder_dim: 解码器的维度
        :params vocab_size: 单词总数
        :params encoder_dim: 编码图像的特征维度
        :params dropout: dropout 的比例
        """
        super(Decoder, self).__init__()
        # 定义类中的参数
        self.encoder_dim = encoder_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size

        # 定义decoder中需要的网络层
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # 定义词嵌入 word embedding, (vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)  # 定义 dropout，dropout = 0.5
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim,
                                       bias=True)  # 定义 LSTMCell 作为 Decoder 中的序列模块，输入是 embed + encoder_out
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # 定义线性层将 encoder_out 转换成 hidden state
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # 定义线性层将 encoder_out 转换成 cell state
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # 定义线性层, decoder_dim -> encoder_dim
        self.sigmoid = nn.Sigmoid()  # 定义 sigoid 激活函数F
        self.fc = nn.Linear(decoder_dim, vocab_size)  # 定义输出的线性层

    def init_hidden_state(self, encoder_out):
        """
        给 LSTM 传入初始的 hidden state，其依赖于 Encoder 的输出
        :param encoder_out: 通过 Encoder 之后的特征，维度是 (bs, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        # 对所有的像素求平均
        mean_encoder_out = encoder_out.mean(dim=1)
        # 线性映射分别得到 hidden state 和 cell state
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lens):
        """
        Decoder的向前传播机制
        :param:encoder_out:编码之后的特征，维度是(bs, num_pixels, encoder_dim)
        :param:encoder_caption:字幕，维度是(bs,max_caption_len)
        :param:caption_lens:字幕真正长度，维度是 (bs, 1)

        Returns:predictions，预测的字幕
        """
        batch_size = encoder_out.shape[0]
        encoder_dim = encoder_out.shape[-1]
        vocab_size = self.vocab_size

        # flatten encode_out 特征
        encoder_out = encoder_out.view(batch_size, -1,
                                       encoder_dim)  # (bs, size_pic,size_pic,encodet_dim)-->(bs, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # 对输入的字幕长度按照降序排列,同时对encoder_out也按照该顺序进行排列
        # 对应原理：分批次训练时，字幕长度短的训练次数少
        caption_lens, sort_idx = caption_lens.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_idx]
        encoded_captions = encoded_captions[sort_idx]
        embeddings = self.embedding(
            encoded_captions)  # 得到 encoded_captions 的词向量, （bs, max_caption_len）-->(bs, max_caption_lens, embed_dim)

        # 初始化 LSTM hidden state 和 cell state
        h, c = self.init_hidden_state(encoder_out)

        # 不会对 <end> 位置进行解码，所以解码的长度是 caption_lens - 1
        decode_lens = (caption_lens - 1).tolist()

        # 定义存储预测结果的空 tensor
        predictions = torch.zeros(batch_size, max(decode_lens), vocab_size)

        # 在每个时间步，通过 decoder 上一步的 hidden state 来生成新的单词
        for t in range(max(decode_lens)):
            # 决定当前时间步的 batch_size，通过 [:batch_size_t] 可以得到当前需要的 tensor
            # t ==0; batch_size_t = all;  意味着长度越短的字幕迭代的次数越少
            batch_size_t = sum([l > t for l in decode_lens])
            print("t:", t, " bs_t:", batch_size_t);

            # 前向传播一个时间步，输入是 embeddings
            # hidden state 和 cell state 也需要输入到网络中，注意使用 batch_size_t 取得当前有效的 tensor
            h, c = self.decode_step(embeddings[:batch_size_t, t,:],
                (h[:batch_size_t], c[:batch_size_t]))

            preds = self.fc(self.dropout(h))  # 对 h 进行 dropout 和全连接层得到预测结果

            predictions[:batch_size_t, t, :] = preds

        return predictions, encoded_captions, decode_lens, sort_idx
