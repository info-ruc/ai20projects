1. 对0的处理
   - pad的0, 在linear中会因为bias变成值, 导致attn会集合pad的那一行
   - cdd_embedding某一行是0极大可能是这个单词是数字、符号等等奇怪东东

2. 为什么fusion_matrix中会有相同的行
3. 为什么fusion_matrix中会有某一行为0
4. gumbel_softmax到底怎么用


1. KNRM会学习wordembedding, 我的word embedding是固定的
2. 为什么dataloader_test的num_workers得是0？
   - 因为是iter style的dataset, 不是0的话没法控制每一个worker读多少



1. soft-topk, 取不到精确的topK, 一定会带小数
2. transformer怎么用？
3. 没有历史记录的用户咋办啊
   1. 交互
4. 生成负例交互的话就不用npratio了吧？？!!!

5. gumbel softmax会采样到补0的history新闻上, 导致之后无法交互
   - 得用-inf mask掉每一行的末尾的0, 这样就不会选中pad的记录
   
6. 如果某impression的负例不足npratio, 那么就会补0, 导致计算candidate和history相关度的时候后面几行都是0
   - 目前的办法: 在训练样本中增添一个返回项: 补零的行数


1. KNRM也没有管词序。。。。
2. FIM在小数据集上是ok的, 但是大数据集就无了