import numpy as np
import tensorflow as tf
import time
import os
from os.path import join
from tensorflow import keras

from utils import cal_metric

__all__ = ["DKN"]


class DKN():
    """DKN model (Deep Knowledge-Aware Network)

    H. Wang, F. Zhang, X. Xie and M. Guo, "DKN: Deep Knowledge-Aware Network for News
    Recommendation", in Proceedings of the 2018 World Wide Web Conference on World
    Wide Web, 2018.
    """

    def __init__(self, hparams, iterator_creator, seed=None):
        """Initialization steps for DKN.
        Compared with the BaseModel, DKN requires two different pre-computed embeddings,
        i.e. word embedding and entity embedding.
        After creating these two embedding variables, BaseModel's __init__ method will be called.

        Args:
            hparams (obj): Global hyper-parameters.
            iterator_creator (obj): DKN data loader class.
        """
        self.graph = tf.Graph()
        with self.graph.as_default():  # 计算图
            with tf.name_scope("embedding"):  # 空间区域限定于“embedding”
                word2vec_embedding = self._init_embedding(hparams.wordEmb_file)
                self.embedding = tf.Variable(
                    word2vec_embedding, trainable=True, name="word"
                )

                if hparams.use_entity:
                    e_embedding = self._init_embedding(hparams.entityEmb_file)
                    W = tf.Variable(
                        tf.random.uniform([hparams.entity_dim, hparams.dim], -1, 1)
                    )  # hparams.dim是变换过后向量的维度
                    b = tf.Variable(tf.zeros([hparams.dim]))
                    e_embedding_transformed = tf.nn.tanh(tf.matmul(e_embedding, W) + b)
                    self.entity_embedding = tf.Variable(
                        e_embedding_transformed, trainable=True, name="entity"
                    )
                else:
                    self.entity_embedding = tf.Variable(
                        tf.constant(
                            0.0,
                            shape=[hparams.entity_size, hparams.dim],
                            dtype=tf.float32,
                        ),
                        trainable=True,
                        name="entity",
                    )

                if hparams.use_context:
                    c_embedding = self._init_embedding(hparams.contextEmb_file)
                    W = tf.Variable(
                        tf.random.uniform([hparams.entity_dim, hparams.dim], -1, 1)
                    )
                    b = tf.Variable(tf.zeros([hparams.dim]))
                    c_embedding_transformed = tf.nn.tanh(tf.matmul(c_embedding, W) + b)
                    self.context_embedding = tf.Variable(
                        c_embedding_transformed, trainable=True, name="context"
                    )
                else:
                    self.context_embedding = tf.Variable(
                        tf.constant(
                            0.0,
                            shape=[hparams.entity_size, hparams.dim],
                            dtype=tf.float32,
                        ),
                        trainable=True,
                        name="context",
                    )

        self.seed = seed  # 设置随机种子
        tf.compat.v1.set_random_seed(seed)
        np.random.seed(seed)
        self.iterator = iterator_creator(hparams, self.graph)

        with self.graph.as_default():
            self.hparams = hparams

            self.layer_params = []
            self.embed_params = []
            self.cross_params = []
            self.layer_keeps = tf.compat.v1.placeholder(tf.float32, name="layer_keeps")
            self.keep_prob_train = None
            self.keep_prob_test = None
            self.is_train_stage = tf.compat.v1.placeholder(
                tf.bool, shape=(), name="is_training"
            )
            self.group = tf.compat.v1.placeholder(tf.int32, shape=(), name="group")

            self.initializer = self._get_initializer()  # Xavier初始化

            self.logit = self._build_graph()
            self.pred = self._get_pred(self.logit, self.hparams.method)  # sigmod函数

            self.loss = self._get_loss()  # 损失函数
            self.saver = tf.compat.v1.train.Saver(max_to_keep=self.hparams.epochs)
            self.update = self._build_train_opt()
            self.extra_update_ops = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.UPDATE_OPS
            )
            self.init_op = tf.compat.v1.global_variables_initializer()
            self.merged = self._add_summaries()

        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        self.sess = tf.compat.v1.Session(
            graph=self.graph, config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)
        )
        self.sess.run(self.init_op)

    def _init_embedding(self, file_path):
        """Load pre-trained embeddings as a constant tensor.

        Args:
            file_path (str): the pre-trained embeddings filename.
        Returns:
            obj: A constant tensor.
        """
        return tf.constant(np.load(file_path).astype(np.float32))

    def _get_initializer(self):  # 初始化方法
        if self.hparams.init_method == "tnormal":
            return tf.truncated_normal_initializer(
                stddev=self.hparams.init_value, seed=self.seed
            )
        elif self.hparams.init_method == "uniform":
            return tf.random_uniform_initializer(
                -self.hparams.init_value, self.hparams.init_value, seed=self.seed
            )
        elif self.hparams.init_method == "normal":
            return tf.random_normal_initializer(
                stddev=self.hparams.init_value, seed=self.seed
            )
        elif self.hparams.init_method == "xavier_normal":
            return tf.contrib.layers.xavier_initializer(uniform=False, seed=self.seed)
        elif self.hparams.init_method == "xavier_uniform":
            return tf.contrib.layers.xavier_initializer(uniform=True, seed=self.seed)
        elif self.hparams.init_method == "he_normal":
            return tf.contrib.layers.variance_scaling_initializer(
                factor=2.0, mode="FAN_IN", uniform=False, seed=self.seed
            )
        elif self.hparams.init_method == "he_uniform":
            return tf.contrib.layers.variance_scaling_initializer(
                factor=2.0, mode="FAN_IN", uniform=True, seed=self.seed
            )
        else:
            return tf.truncated_normal_initializer(
                stddev=self.hparams.init_value, seed=self.seed
            )

    def _build_graph(self):
        hparams = self.hparams
        self.keep_prob_train = 1 - np.array(hparams.dropout)
        self.keep_prob_test = np.ones_like(hparams.dropout)
        with tf.compat.v1.variable_scope("DKN") as scope:
            logit = self._build_dkn()
            return logit

    def _get_pred(self, logit, task):
        """Make final output as prediction score, according to different tasks.

        Args:
            logit (obj): Base prediction value.
            task (str): A task (values: regression/classification)

        Returns:
            obj: Transformed score
        """
        if task == "classification":
            pred = tf.sigmoid(logit)
        else:
            raise ValueError(
                "method must be classification, but now is {0}".format(
                    task
                )
            )
        return pred

    def _get_loss(self):
        """Make loss function, consists of data loss and regularization loss

        Returns:
            obj: Loss value
        """
        self.data_loss = self._compute_data_loss()
        self.regular_loss = self._compute_regular_loss()
        self.loss = tf.add(self.data_loss, self.regular_loss)
        return self.loss

    def _build_train_opt(self):
        """Construct gradient descent based optimization step
        In this step, we provide gradient clipping option. Sometimes we what to clip the gradients
        when their absolute values are too large to avoid gradient explosion.
        Returns:
            obj: An operation that applies the specified optimization step.
        """
        train_step = self._train_opt()  # 设定优化方法（某学习率下）
        gradients, variables = zip(*train_step.compute_gradients(self.loss))
        if self.hparams.is_clip_norm:
            gradients = [
                None
                if gradient is None
                else tf.clip_by_norm(gradient, self.hparams.max_grad_norm)
                for gradient in gradients
            ]
        return train_step.apply_gradients(zip(gradients, variables))

    def _build_dkn(self):
        """The main function to create DKN's logic.

        Returns:
            obj: Prediction score made by the DKN model.
        """
        hparams = self.hparams
        # build attention model for clicked news and candidate news
        click_news_embed_batch, candidate_news_embed_batch = self._build_pair_attention(
            self.iterator.candidate_news_index_batch,
            self.iterator.candidate_news_entity_index_batch,
            self.iterator.click_news_index_batch,
            self.iterator.click_news_entity_index_batch,
            hparams,
        )

        nn_input = tf.concat(
            [click_news_embed_batch, candidate_news_embed_batch], axis=1
        )

        dnn_channel_part = 2
        last_layer_size = dnn_channel_part * self.num_filters_total
        layer_idx = 0
        hidden_nn_layers = []
        hidden_nn_layers.append(nn_input)
        with tf.compat.v1.variable_scope(
                "nn_part", initializer=self.initializer
        ) as scope:
            for idx, layer_size in enumerate(hparams.layer_sizes):
                curr_w_nn_layer = tf.compat.v1.get_variable(
                    name="w_nn_layer" + str(layer_idx),
                    shape=[last_layer_size, layer_size],
                    dtype=tf.float32,
                )
                curr_b_nn_layer = tf.compat.v1.get_variable(
                    name="b_nn_layer" + str(layer_idx),
                    shape=[layer_size],
                    dtype=tf.float32,
                )
                curr_hidden_nn_layer = tf.compat.v1.nn.xw_plus_b(
                    hidden_nn_layers[layer_idx], curr_w_nn_layer, curr_b_nn_layer
                )
                if hparams.enable_BN is True:
                    curr_hidden_nn_layer = tf.layers.batch_normalization(
                        curr_hidden_nn_layer,
                        momentum=0.95,
                        epsilon=0.0001,
                        training=self.is_train_stage,
                    )

                activation = hparams.activation[idx]
                curr_hidden_nn_layer = self._active_layer(
                    logit=curr_hidden_nn_layer, activation=activation
                )
                hidden_nn_layers.append(curr_hidden_nn_layer)
                layer_idx += 1
                last_layer_size = layer_size
                self.layer_params.append(curr_w_nn_layer)
                self.layer_params.append(curr_b_nn_layer)

            w_nn_output = tf.compat.v1.get_variable(
                name="w_nn_output", shape=[last_layer_size, 1], dtype=tf.float32
            )
            b_nn_output = tf.compat.v1.get_variable(
                name="b_nn_output", shape=[1], dtype=tf.float32
            )
            self.layer_params.append(w_nn_output)
            self.layer_params.append(b_nn_output)
            nn_output = tf.compat.v1.nn.xw_plus_b(
                hidden_nn_layers[-1], w_nn_output, b_nn_output
            )
            return nn_output

    def _add_summaries(self):
        tf.compat.v1.summary.scalar("data_loss", self.data_loss)
        tf.compat.v1.summary.scalar("regular_loss", self.regular_loss)
        tf.compat.v1.summary.scalar("loss", self.loss)
        merged = tf.compat.v1.summary.merge_all()
        return merged

    def _compute_data_loss(self):
        if self.hparams.loss == "cross_entropy_loss":
            data_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=tf.reshape(self.logit, [-1]),
                    labels=tf.reshape(self.iterator.labels, [-1]),
                )
            )
        elif self.hparams.loss == "square_loss":
            data_loss = tf.sqrt(
                tf.reduce_mean(
                    tf.squared_difference(
                        tf.reshape(self.pred, [-1]),
                        tf.reshape(self.iterator.labels, [-1]),
                    )
                )
            )
        elif self.hparams.loss == "log_loss":
            data_loss = tf.reduce_mean(
                tf.compat.v1.losses.log_loss(
                    predictions=tf.reshape(self.pred, [-1]),
                    labels=tf.reshape(self.iterator.labels, [-1]),
                )
            )
        else:
            raise ValueError("this loss not defined {0}".format(self.hparams.loss))
        return data_loss

    def _compute_regular_loss(self):
        """Construct regular loss. Usually it's comprised of l1 and l2 norm.
        Users can designate which norm to be included via config file.
        Returns:
            obj: Regular loss.
        """
        regular_loss = self._l2_loss() + self._l1_loss() + self._cross_l_loss()
        return tf.reduce_sum(regular_loss)

    def _train_opt(self):
        """Get the optimizer according to configuration. Usually we will use Adam.
        Returns:
            obj: An optimizer.
        """
        lr = self.hparams.learning_rate
        optimizer = self.hparams.optimizer

        if optimizer == "adadelta":
            train_step = tf.train.AdadeltaOptimizer(lr)
        elif optimizer == "adagrad":
            train_step = tf.train.AdagradOptimizer(lr)
        elif optimizer == "sgd":
            train_step = tf.train.GradientDescentOptimizer(lr)
        elif optimizer == "adam":
            train_step = tf.compat.v1.train.AdamOptimizer(lr)
        elif optimizer == "ftrl":
            train_step = tf.train.FtrlOptimizer(lr)
        elif optimizer == "gd":
            train_step = tf.train.GradientDescentOptimizer(lr)
        elif optimizer == "padagrad":
            train_step = tf.train.ProximalAdagradOptimizer(lr)
        elif optimizer == "pgd":
            train_step = tf.train.ProximalGradientDescentOptimizer(lr)
        elif optimizer == "rmsprop":
            train_step = tf.train.RMSPropOptimizer(lr)
        elif optimizer == "lazyadam":
            train_step = tf.contrib.opt.LazyAdamOptimizer(lr)
        else:
            train_step = tf.train.GradientDescentOptimizer(lr)
        return train_step

    def _build_pair_attention(
            self,
            candidate_word_batch,
            candidate_entity_batch,
            click_word_batch,
            click_entity_batch,
            hparams,
    ):
        """This function learns the candidate news article's embedding and user embedding.
        User embedding is generated from click history and also depends on the candidate news article via attention mechanism.
        Article embedding is generated via KCNN module.
        Args:
            candidate_word_batch (obj): tensor word indices for constructing news article
            candidate_entity_batch (obj): tensor entity values for constructing news article
            click_word_batch (obj): tensor word indices for constructing user clicked history
            click_entity_batch (obj): tensor entity indices for constructing user clicked history
            hparams (obj): global hyper-parameters
        Returns:
            click_field_embed_final_batch: user embedding
            news_field_embed_final_batch: candidate news article embedding

        """
        doc_size = hparams.doc_size
        attention_hidden_sizes = hparams.attention_layer_sizes

        clicked_words = tf.reshape(click_word_batch, shape=[-1, doc_size])
        clicked_entities = tf.reshape(click_entity_batch, shape=[-1, doc_size])

        with tf.compat.v1.variable_scope(
                "attention_net", initializer=self.initializer
        ) as scope:

            # use kims cnn to get conv embedding
            with tf.compat.v1.variable_scope(
                    "kcnn", initializer=self.initializer, reuse=tf.compat.v1.AUTO_REUSE
            ) as cnn_scope:
                news_field_embed = self._kims_cnn(
                    candidate_word_batch, candidate_entity_batch, hparams
                )
                click_field_embed = self._kims_cnn(
                    clicked_words, clicked_entities, hparams
                )
                click_field_embed = tf.reshape(
                    click_field_embed,
                    shape=[
                        -1,
                        hparams.history_size,
                        hparams.num_filters * len(hparams.filter_sizes),
                    ],
                )

            avg_strategy = False  # ？？？？？
            if avg_strategy:
                click_field_embed_final = tf.reduce_mean(
                    click_field_embed, axis=1, keepdims=True
                )
            else:
                news_field_embed = tf.expand_dims(news_field_embed, 1)
                news_field_embed_repeat = tf.add(
                    tf.zeros_like(click_field_embed), news_field_embed
                )
                attention_x = tf.concat(
                    axis=-1, values=[click_field_embed, news_field_embed_repeat]
                )
                attention_x = tf.reshape(
                    attention_x, shape=[-1, self.num_filters_total * 2]
                )
                attention_w = tf.compat.v1.get_variable(
                    name="attention_hidden_w",
                    shape=[self.num_filters_total * 2, attention_hidden_sizes],
                    dtype=tf.float32,
                )
                attention_b = tf.compat.v1.get_variable(
                    name="attention_hidden_b",
                    shape=[attention_hidden_sizes],
                    dtype=tf.float32,
                )
                curr_attention_layer = tf.compat.v1.nn.xw_plus_b(
                    attention_x, attention_w, attention_b
                )

                if hparams.enable_BN is True:
                    curr_attention_layer = tf.layers.batch_normalization(
                        curr_attention_layer,
                        momentum=0.95,
                        epsilon=0.0001,
                        training=self.is_train_stage,
                    )

                activation = hparams.attention_activation
                curr_attention_layer = self._active_layer(
                    logit=curr_attention_layer, activation=activation
                )
                attention_output_w = tf.compat.v1.get_variable(
                    name="attention_output_w",
                    shape=[attention_hidden_sizes, 1],
                    dtype=tf.float32,
                )
                attention_output_b = tf.compat.v1.get_variable(
                    name="attention_output_b", shape=[1], dtype=tf.float32
                )
                attention_weight = tf.compat.v1.nn.xw_plus_b(
                    curr_attention_layer, attention_output_w, attention_output_b
                )
                attention_weight = tf.reshape(
                    attention_weight, shape=[-1, hparams.history_size, 1]
                )
                norm_attention_weight = tf.nn.softmax(attention_weight, axis=1)
                click_field_embed_final = tf.reduce_sum(
                    tf.multiply(click_field_embed, norm_attention_weight),
                    axis=1,
                    keepdims=True,
                )
                if attention_w not in self.layer_params:
                    self.layer_params.append(attention_w)
                if attention_b not in self.layer_params:
                    self.layer_params.append(attention_b)
                if attention_output_w not in self.layer_params:
                    self.layer_params.append(attention_output_w)
                if attention_output_b not in self.layer_params:
                    self.layer_params.append(attention_output_b)
            self.news_field_embed_final_batch = tf.squeeze(news_field_embed)
            click_field_embed_final_batch = tf.squeeze(click_field_embed_final)

        return click_field_embed_final_batch, self.news_field_embed_final_batch

    def _l2_loss(self):
        hparams = self.hparams
        l2_loss = tf.zeros([1], dtype=tf.float32)
        # embedding_layer l2 loss
        l2_loss = tf.add(
            l2_loss, tf.multiply(hparams.embed_l2, tf.nn.l2_loss(self.embedding))
        )
        if hparams.use_entity:
            l2_loss = tf.add(
                l2_loss,
                tf.multiply(hparams.embed_l2, tf.nn.l2_loss(self.entity_embedding)),
            )
        if hparams.use_entity and hparams.use_context:
            l2_loss = tf.add(
                l2_loss,
                tf.multiply(hparams.embed_l2, tf.nn.l2_loss(self.context_embedding)),
            )
        params = self.layer_params
        for param in params:
            l2_loss = tf.add(
                l2_loss, tf.multiply(hparams.layer_l2, tf.nn.l2_loss(param))
            )
        return l2_loss

    def _l1_loss(self):
        hparams = self.hparams
        l1_loss = tf.zeros([1], dtype=tf.float32)
        # embedding_layer l1 loss
        l1_loss = tf.add(
            l1_loss, tf.multiply(hparams.embed_l1, tf.norm(self.embedding, ord=1))
        )
        if hparams.use_entity:
            l1_loss = tf.add(
                l1_loss,
                tf.multiply(hparams.embed_l1, tf.norm(self.entity_embedding, ord=1)),
            )
        if hparams.use_entity and hparams.use_context:
            l1_loss = tf.add(
                l1_loss,
                tf.multiply(hparams.embed_l1, tf.norm(self.context_embedding, ord=1)),
            )
        params = self.layer_params
        for param in params:
            l1_loss = tf.add(
                l1_loss, tf.multiply(hparams.layer_l1, tf.norm(param, ord=1))
            )
        return l1_loss

    def _cross_l_loss(self):
        """Construct L1-norm and L2-norm on cross network parameters for loss function.
        Returns:
            obj: Regular loss value on cross network parameters.
        """
        # hparams.cross_l1 = hparams.cross_l2 = 0
        cross_l_loss = tf.zeros([1], dtype=tf.float32)
        for param in self.cross_params:
            cross_l_loss = tf.add(
                cross_l_loss, tf.multiply(self.hparams.cross_l1, tf.norm(param, ord=1))
            )
            cross_l_loss = tf.add(
                cross_l_loss, tf.multiply(self.hparams.cross_l2, tf.norm(param, ord=2))
            )
        return cross_l_loss

    def _dropout(self, logit, keep_prob):
        """Apply drops upon the input value.
        Args:
            logit (obj): The input value.
            keep_prob (float): The probability of keeping each element.

        Returns:
            obj: A tensor of the same shape of logit.
        """
        return tf.nn.dropout(x=logit, keep_prob=keep_prob)

    def _activate(self, logit, activation):
        if activation == "sigmoid":
            return tf.nn.sigmoid(logit)
        elif activation == "softmax":
            return tf.nn.softmax(logit)
        elif activation == "relu":
            return tf.nn.relu(logit)
        elif activation == "tanh":
            return tf.nn.tanh(logit)
        elif activation == "elu":
            return tf.nn.elu(logit)
        elif activation == "identity":
            return tf.identity(logit)
        else:
            raise ValueError("this activations not defined {0}".format(activation))

    def _active_layer(self, logit, activation, layer_idx=-1):
        """Transform the input value with an activation. May use dropout.

        Args:
            logit (obj): Input value.
            activation (str): A string indicating the type of activation function.
            layer_idx (int): Index of current layer. Used to retrieve corresponding parameters

        Returns:
            obj: A tensor after applying activation function on logit.
        """
        if layer_idx >= 0 and self.hparams.user_dropout:
            logit = self._dropout(logit, self.layer_keeps[layer_idx])
        return self._activate(logit, activation)

    def _kims_cnn(self, word, entity, hparams):
        """The KCNN module. KCNN is an extension of traditional CNN that incorporates symbolic knowledge from
        a knowledge graph into sentence representation learning.
        Args:
            word (obj): word indices for the sentence.
            entity (obj): entity indices for the sentence. Entities are aligned with words in the sentence.
            hparams (obj): global hyper-parameters.

        Returns:
            obj: Sentence representation.
        """
        # kims cnn parameter
        filter_sizes = hparams.filter_sizes
        num_filters = hparams.num_filters

        dim = hparams.dim
        embedded_chars = tf.nn.embedding_lookup(self.embedding, word)
        if hparams.use_entity and hparams.use_context:
            entity_embedded_chars = tf.nn.embedding_lookup(
                self.entity_embedding, entity
            )
            context_embedded_chars = tf.nn.embedding_lookup(
                self.context_embedding, entity
            )
            concat = tf.concat(
                [embedded_chars, entity_embedded_chars, context_embedded_chars], axis=-1
            )
        elif hparams.use_entity:
            entity_embedded_chars = tf.nn.embedding_lookup(
                self.entity_embedding, entity
            )
            concat = tf.concat([embedded_chars, entity_embedded_chars], axis=-1)
        else:
            concat = embedded_chars
        concat_expanded = tf.expand_dims(concat, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.compat.v1.variable_scope(
                    "conv-maxpool-%s" % filter_size, initializer=self.initializer
            ):
                # Convolution Layer
                if hparams.use_entity and hparams.use_context:
                    filter_shape = [filter_size, dim * 3, 1, num_filters]
                elif hparams.use_entity:
                    filter_shape = [filter_size, dim * 2, 1, num_filters]
                else:
                    filter_shape = [filter_size, dim, 1, num_filters]
                W = tf.compat.v1.get_variable(
                    name="W" + "_filter_size_" + str(filter_size),
                    shape=filter_shape,
                    dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                )
                b = tf.compat.v1.get_variable(
                    name="b" + "_filter_size_" + str(filter_size),
                    shape=[num_filters],
                    dtype=tf.float32,
                )
                if W not in self.layer_params:
                    self.layer_params.append(W)
                if b not in self.layer_params:
                    self.layer_params.append(b)
                conv = tf.nn.conv2d(
                    concat_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv",
                )
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool2d(
                    h,
                    ksize=[1, hparams.doc_size - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool",
                )
                pooled_outputs.append(pooled)
        # Combine all the pooled features
        # self.num_filters_total is the kims cnn output dimension
        self.num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, axis=-1)
        h_pool_flat = tf.reshape(h_pool, [-1, self.num_filters_total])
        return h_pool_flat

    def train(self, sess, feed_dict):
        """Go through the optimization step once with training data in feed_dict.

        Args:
            sess (obj): The model session object.
            feed_dict (dict): Feed values to train the model. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of values, including update operation, total loss, data loss, and merged summary.
        """
        feed_dict[self.layer_keeps] = self.keep_prob_train
        feed_dict[self.is_train_stage] = True
        return sess.run(
            [
                self.update,
                self.extra_update_ops,
                self.loss,
                self.data_loss,
                self.merged,
            ],
            feed_dict=feed_dict,
        )

    def group_labels(self, labels, preds, group_keys):
        """Devide labels and preds into several group according to values in group keys.
        Args:
            labels (list): ground truth label list.
            preds (list): prediction score list.
            group_keys (list): group key list.
        Returns:
            all_labels: labels after group.
            all_preds: preds after group.
        """
        all_keys = list(set(group_keys))
        group_labels = {k: [] for k in all_keys}
        group_preds = {k: [] for k in all_keys}
        for l, p, k in zip(labels, preds, group_keys):
            group_labels[k].append(l)
            group_preds[k].append(p)
        all_labels = []
        all_preds = []
        for k in all_keys:
            all_labels.append(group_labels[k])
            all_preds.append(group_preds[k])
        return all_labels, all_preds

    def eval(self, sess, feed_dict):
        """Evaluate the data in feed_dict with current model.

        Args:
            sess (obj): The model session object.
            feed_dict (dict): Feed values for evaluation. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of evaluated results, including total loss value, data loss value,
                predicted scores, and ground-truth labels.
        """
        feed_dict[self.layer_keeps] = self.keep_prob_test
        feed_dict[self.is_train_stage] = False
        return sess.run([self.pred, self.iterator.labels], feed_dict=feed_dict)

    def run_eval(self, filename):
        """Evaluate the given file and returns some evaluation metrics.

        Args:
            filename (str): A file name that will be evaluated.

        Returns:
            dict: A dictionary contains evaluation metrics.
        """
        load_sess = self.sess
        preds = []
        labels = []
        imp_indexs = []
        for batch_data_input, imp_index, data_size in self.iterator.load_data_from_file(
                filename
        ):
            step_pred, step_labels = self.eval(load_sess, batch_data_input)
            preds.extend(np.reshape(step_pred, -1))
            labels.extend(np.reshape(step_labels, -1))
            imp_indexs.extend(np.reshape(imp_index, -1))
        res = cal_metric(labels, preds, self.hparams.metrics)
        if self.hparams.pairwise_metrics is not None:
            group_labels, group_preds = self.group_labels(labels, preds, imp_indexs)
            res_pairwise = cal_metric(
                group_labels, group_preds, self.hparams.pairwise_metrics
            )
            res.update(res_pairwise)
        return res

    def fit(self, train_file, valid_file, test_file=None):
        """Fit the model with train_file. Evaluate the model on valid_file per epoch to observe the training status.
        If test_file is not None, evaluate it too.

        Args:
            train_file (str): training data set.
            valid_file (str): validation set.
            test_file (str): test set.

        Returns:
            obj: An instance of self.
        """
        print("fitting")
        if self.hparams.write_tfevents:  # 可视化
            self.writer = tf.summary.FileWriter(
                self.hparams.SUMMARIES_DIR, self.sess.graph
            )

        train_sess = self.sess
        for epoch in range(1, self.hparams.epochs + 1):
            step = 0
            self.hparams.current_epoch = epoch

            epoch_loss = 0
            train_start = time.time()
            for (
                    batch_data_input,
                    impression,
                    data_size,
            ) in self.iterator.load_data_from_file(train_file):  # 获取训练数据
                step_result = self.train(train_sess, batch_data_input)
                (_, _, step_loss, step_data_loss, summary) = step_result
                if self.hparams.write_tfevents:
                    self.writer.add_summary(summary, step)
                epoch_loss += step_loss
                step += 1
                if step % self.hparams.show_step == 0:
                    print(
                        "step {0:d} , total_loss: {1:.4f}, data_loss: {2:.4f}".format(
                            step, step_loss, step_data_loss
                        )
                    )

            train_end = time.time()
            train_time = train_end - train_start

            if self.hparams.save_model:
                if not os.path.exists(self.hparams.MODEL_DIR):
                    os.makedirs(self.hparams.MODEL_DIR)
                if epoch % self.hparams.save_epoch == 0:
                    save_path_str = join(self.hparams.MODEL_DIR, "epoch_" + str(epoch))
                    checkpoint_path = self.saver.save(
                        sess=train_sess, save_path=save_path_str
                    )

            eval_start = time.time()
            eval_res = self.run_eval(valid_file)
            train_info = ",".join(
                [
                    str(item[0]) + ":" + str(item[1])
                    for item in [("logloss loss", epoch_loss / step)]
                ]
            )
            eval_info = ", ".join(
                [
                    str(item[0]) + ":" + str(item[1])
                    for item in sorted(eval_res.items(), key=lambda x: x[0])
                ]
            )
            if test_file is not None:
                test_res = self.run_eval(test_file)
                test_info = ", ".join(
                    [
                        str(item[0]) + ":" + str(item[1])
                        for item in sorted(test_res.items(), key=lambda x: x[0])
                    ]
                )
            eval_end = time.time()
            eval_time = eval_end - eval_start

            if test_file is not None:
                print(
                    "at epoch {0:d}".format(epoch)
                    + "\ntrain info: "
                    + train_info
                    + "\neval info: "
                    + eval_info
                    + "\ntest info: "
                    + test_info
                )
            else:
                print(
                    "at epoch {0:d}".format(epoch)
                    + "\ntrain info: "
                    + train_info
                    + "\neval info: "
                    + eval_info
                )
            print(
                "at epoch {0:d} , train time: {1:.1f} eval time: {2:.1f}".format(
                    epoch, train_time, eval_time
                )
            )

        if self.hparams.write_tfevents:
            self.writer.close()

        return self