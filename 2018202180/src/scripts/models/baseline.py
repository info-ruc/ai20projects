import torch
import math
import torch.nn as nn

class GCAModel(nn.Module):
    def __init__(self,hparams,vocab):
        super().__init__()

        self.cdd_size = (hparams['npratio'] + 1) if hparams['npratio'] > 0 else 1
        self.metrics = hparams['metrics']
        self.device = torch.device(hparams['device'])
        self.embedding = vocab.vectors.to(self.device)

        self.batch_size = hparams['batch_size']
        self.signal_length = hparams['title_size']
        self.his_size = hparams['his_size']

        self.mus = torch.arange(-0.9,1.1,0.1,device=self.device)
        self.kernel_num = len(self.mus)
        self.sigmas = torch.tensor([0.1]*(self.kernel_num - 1) + [0.001], device=self.device)

        self.head_num = hparams['head_num']
        self.query_dim = hparams['query_dim']
        self.embedding_dim = hparams['embedding_dim']
        self.value_dim = hparams['value_dim']
        self.repr_dim = self.head_num * self.value_dim
        
        self.query_words = nn.Parameter(torch.randn((1,self.query_dim), requires_grad=True))

        # elements in the slice along dim will sum up to 1 
        self.softmax = nn.Softmax(dim=-1)
        self.gumbel_softmax = nn.functional.gumbel_softmax
        self.ReLU = nn.ReLU()
        self.DropOut = nn.Dropout(p=hparams['dropout_p'])
        self.CosSim = nn.CosineSimilarity(dim=-1)

        self.queryProject_words = nn.ModuleList([]).extend([nn.Linear(self.embedding_dim,self.embedding_dim, bias=False) for _ in range(self.head_num)])
        self.valueProject_words = nn.ModuleList([]).extend([nn.Linear(self.embedding_dim,self.value_dim, bias=False) for _ in range(self.head_num)])
        self.keyProject_words = nn.Linear(self.value_dim * self.head_num, self.query_dim, bias=True)
        self.learningToRank = nn.Linear(self.kernel_num, 1)

    def _scaled_dp_attention(self,query,key,value):
        """ calculate scaled attended output of values
        
        Args:
            query: tensor of [batch_size, *, query_num, key_dim]
            key: tensor of [batch_size, *, key_num, key_dim]
            value: tensor of [batch_size, *, key_num, value_dim]
        
        Returns:
            attn_output: tensor of [batch_size, *, query_num, value_dim]
        """

        # make sure dimension matches
        assert query.shape[-1] == key.shape[-1]
        key = key.transpose(-2,-1)

        attn_weights = torch.matmul(query,key)/math.sqrt(self.embedding_dim)
        attn_weights = self.softmax(attn_weights)
        
        attn_output = torch.matmul(attn_weights,value)

        return attn_output

    def _self_attention(self,input,head_idx):
        """ apply self attention of head#idx over input tensor
        
        Args:
            input: tensor of [batch_size, *, embedding_dim]
            head_idx: interger of attention head index

        Returns:
            self_attn_output: tensor of [batch_size, *, embedding_dim]
        """
        query = self.queryProject_words[head_idx](input)

        attn_output = self._scaled_dp_attention(query,input,input)
        self_attn_output = self.valueProject_words[head_idx](attn_output)

        return self_attn_output
    
    def _word_attention(self, query, key, value):
        """ apply word-level attention

        Args:
            attn_word_embedding_key: tensor of [batch_size, *, signal_length, query_dim]
            attn_word_embedding_value: tensor of [batch_size, *, signal_length, repr_dim]
        
        Returns:
            attn_output: tensor of [batch_size, *, repr_dim]
        """
        query = query.expand(key.shape[0], key.shape[1], 1, self.query_dim)

        attn_output = self._scaled_dp_attention(query,key,value).squeeze(dim=2)

        return attn_output


    def _multi_head_self_attention(self,input):
        """ apply multi-head self attention over input tensor

        Args:
            input: tensor of [batch_size, *, signal_length, embedding_dim]
        
        Returns:
            multi_head_self_attn: tensor of [batch_size, *, 1, repr_dim]
        """
        self_attn_outputs = [self._self_attention(input,i) for i in range(self.head_num)]

        # project the embedding of each words to query subspace
        # keep the original embedding of each words as values
        multi_head_self_attn_value = torch.cat(self_attn_outputs,dim=-1)
        multi_head_self_attn_key = torch.tanh(self.keyProject_words(multi_head_self_attn_value))

        additive_attn_embedding = self._word_attention(self.query_words, multi_head_self_attn_key,multi_head_self_attn_value)
        return additive_attn_embedding

    def _news_encoder(self,news_batch):
        """ encode set of news to news representations of [batch_size, cdd_size, tranformer_dim]
        
        Args:
            news_batch: tensor of [batch_size, cdd_size, title_size]
            word_query: tensor of [set_size, preference_dim]
        
        Returns:
            news_repr_attn: tensor of [batch_size, cdd_size, repr_dim]
            news_repr_origin: tensor of [batch_size, cdd_size, signal_length, embedding_dim] 
        """
        news_embedding_origin = self.DropOut(self.embedding[news_batch].to(self.device))
        news_embedding_attn = self._multi_head_self_attention(news_embedding_origin)
        return news_embedding_attn, news_embedding_origin

    def _kernel_pooling(self,matrixs):
        """
            apply kernel pooling on matrix, in order to get the relatedness from many levels
        
        Args:
            matrix: tensor of [batch_size, rows, columns]
        
        Returns:
            pooling_vectors: tensor of [batch_size, kernel_num]
        """
        pooling_matrixs = torch.zeros(matrixs.shape[0],matrixs.shape[1],self.kernel_num,device=self.device)
        
        for k in range(self.kernel_num):
            pooling_matrixs[:,:,k] = torch.sum(torch.exp(-(matrixs - self.mus[k])**2 / (2*self.sigmas[k]**2)),dim=2)
        
        pooling_vectors = torch.sum(torch.log(torch.clamp(pooling_matrixs,min=1e-10)),dim=1)
        return pooling_vectors

    def _news_attention(self, cdd_attn, his_attn, his_org, his_mask):
        """ apply news-level attention

        Args:
            cdd_attn: tensor of [batch_size, cdd_size, repr_dim]
            his_attn: tensor of [batch_size, his_size, repr_dim]
            his_org: tensor of [batch_size, his_size, signal_length, embedding_dim]
            his_mask: tensor of [batch_size, his_size, 1]            

        Returns:
            his_activated: tensor of [batch_size, cdd_size, signal_length, embedding_dim]
        """
        attn_weights = torch.bmm(cdd_attn,his_attn.permute(0,2,1))

        # print(attn_weights)

        # padding in history will cause 0 in attention weights, underlying the probability that gumbel_softmax may attend to those meaningless 0 vectors. Masking off these 0s will force the gumbel_softmax to attend to only non-zero histories.
        # mask in candidate also cause such a problem, however we donot need to fix it, because the whole row of attention weight matrix is zero so gumbel_softmax can only capture 0 vectors, though reuslting in redundant calculation but it is what we want of padding 0 to negtive sampled candidates as they may be less than npratio.
        attn_weights =  attn_weights.masked_fill(his_mask.permute(0,2,1), -float("inf"))
        # print(attn_weights)
        # [batch_size, cdd_size, his_size]
        attn_focus = self.gumbel_softmax(attn_weights,dim=2,hard=True)

        # print(attn_focus)

        # [batch_size * cdd_size, signal_length * embedding_dim]
        his_activated = torch.matmul(attn_focus,his_org.view(self.batch_size,self.his_size,-1)).view(self.batch_size,self.cdd_size,self.signal_length,self.embedding_dim)
        
        return his_activated
    
    def _fusion(self,his_activated,cdd_org):
        """ fuse activated history news and candidate news into interaction matrix, words in history is column and words in candidate is row, then apply KNRM on each interaction matrix

        Args:
            his_activated: tensor of [batch_size, cdd_size, signal_length, embedding_dim]
            cdd_org: tensor of [batch_size, cdd_size, signal_length, embedding_dim]
        
        Returns:
            pooling_vector: tensor of [batch_size, cdd_size, kernel_num]
        """

        fusion_matrixs = torch.zeros((self.batch_size, self.cdd_size, self.signal_length, self.signal_length), device=self.device)
        for i in range(self.signal_length):
            fusion_matrixs[ :, :, i, :] = self.CosSim(cdd_org[ :, :, i, :].unsqueeze(2), his_activated)
        
        # print(fusion_matrixs, fusion_matrixs.shape)
        pooling_vectors = self._kernel_pooling(fusion_matrixs.view(-1,self.signal_length,self.signal_length)).view(self.batch_size, -1, self.kernel_num)

        assert pooling_vectors.shape[1] == self.cdd_size or pooling_vectors.shape[1] == 1
        
        return pooling_vectors
    
    def _click_predictor(self,pooling_vectors):
        """ calculate batch of click probability              
        Args:
            pooling_vectors: tensor of [batch_size, cdd_size, kernel_num]
        
        Returns:
            score: tensor of [batch_size, cdd_size]
        """
        score = self.learningToRank(pooling_vectors)

        if self.cdd_size > 1:
            score = nn.functional.log_softmax(score,dim=1)
        else:
            score = torch.sigmoid(score)
        
        return score.squeeze()

    def forward(self,x):
        cdd_news_embedding_attn,cdd_news_embedding_origin = self._news_encoder(x['candidate_title'].long().to(self.device))
        his_news_embedding_attn,his_news_embedding_origin = self._news_encoder(x['clicked_title'].long().to(self.device))

        # mask the history news 
        click_mask = x['click_mask'].to(self.device)

        his_activated = self._news_attention(cdd_news_embedding_attn,his_news_embedding_attn,his_news_embedding_origin, click_mask)
        pooling_vectors = self._fusion(his_activated,cdd_news_embedding_origin)

        # print(pooling_vectors,pooling_vectors.shape)
        score_batch = self._click_predictor(pooling_vectors)
        # print(score_batch,score_batch.shape)
        return score_batch