import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


from reco_utils.recommender.newsrec.models.base_model import BaseModel
from reco_utils.recommender.newsrec.models.layers import (
    AttLayer2,
    ComputeMasking,
    OverwriteMasking,
)

__all__ = ["LSTURNAML"]
class LSTURNAML(BaseModel):
    def __init__(self, hparams, iterator_creator, seed=None):
	
	    self.word2vec_embedding = self._init_embedding(hparams.wordEmb_file)
        self.hparam = hparams

        super().__init__(hparams, iterator_creator, seed=seed)
    def _get_input_label_from_iter(self, batch_data):
        input_feat = [
            batch_data["user_index_batch"],
            batch_data["clicked_title_batch"],
			batch_data["clicked_ab_batch"],
            batch_data["candidate_title_batch"],
			batch_data["candidate_ab_batch"],
        ]
        input_label = batch_data["labels"]
        return input_feat, input_label
		
	def _get_user_feature_from_iter(self, batch_data):
        input_feature = [
		    batch_data["user_index_batch"],
            batch_data["clicked_title_batch"],
            batch_data["clicked_ab_batch"],
        ]
        input_feature = np.concatenate(input_feature, axis=-1)
        return input_feature
	
	def _get_news_feature_from_iter(self, batch_data):
        input_feature = [
            batch_data["candidate_title_batch"],
            batch_data["candidate_ab_batch"],
        ]
        input_feature = np.concatenate(input_feature, axis=-1)
        return input_feature
		
	def _build_graph(self):
        model, scorer = self._build_lsturnaml()
        return model, scorer
	
    def _build_userencoder(self, titleencoder, type="ini"):

        hparams = self.hparams
        his_input_title = keras.Input(
            shape=(hparams.his_size, hparams.title_size), dtype="int32"
        )
        user_indexes = keras.Input(shape=(1,), dtype="int32")

        user_embedding_layer = layers.Embedding(
            len(self.train_iterator.uid2index),
            hparams.gru_unit,
            trainable=True,
            embeddings_initializer="zeros",
        )

        long_u_emb = layers.Reshape((hparams.gru_unit,))(
            user_embedding_layer(user_indexes)
        )
        click_title_presents = layers.TimeDistributed(titleencoder)(his_input_title)

        if type == "ini":
            user_present = layers.GRU(
                hparams.gru_unit,
                kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
                recurrent_initializer=keras.initializers.glorot_uniform(seed=self.seed),
                bias_initializer=keras.initializers.Zeros(),
            )(
                layers.Masking(mask_value=0.0)(click_title_presents),
                initial_state=[long_u_emb],
            )
        elif type == "con":
            short_uemb = layers.GRU(
                hparams.gru_unit,
                kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
                recurrent_initializer=keras.initializers.glorot_uniform(seed=self.seed),
                bias_initializer=keras.initializers.Zeros(),
            )(layers.Masking(mask_value=0.0)(click_title_presents))
            #连接
            user_present = layers.Concatenate()([short_uemb, long_u_emb])
            #创建一个全连接层
            user_present = layers.Dense(
                hparams.gru_unit,
                bias_initializer=keras.initializers.Zeros(),
                kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
            )(user_present)

        model = keras.Model(
            [his_input_title, user_indexes], user_present, name="user_encoder"
        )
        return model	
	
    def _build_newsencoder(self, embedding_layer):
        hparams = self.hparams
        input_title_body_verts = keras.Input(
            shape=(hparams.title_size + hparams.body_size + 2,), dtype="int32"
        )
        #按照不同的类型建不同的层
        sequences_input_title = layers.Lambda(lambda x: x[:, : hparams.title_size])(
            input_title_body_verts
        )
        sequences_input_body = layers.Lambda(
            lambda x: x[:, hparams.title_size : hparams.title_size + hparams.body_size]
        )(input_title_body_verts)
        #对各种类型进行encoder
        title_repr = self._build_titleencoder(embedding_layer)(sequences_input_title)
        body_repr = self._build_bodyencoder(embedding_layer)(sequences_input_body)
        #连接
        concate_repr = layers.Concatenate(axis=-2)(
            [title_repr, body_repr]
        )
		#attention机制
        news_repr = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(
            concate_repr
        )

        model = keras.Model(input_title_body_verts, news_repr, name="news_encoder")
        return model

    def _build_titleencoder(self, embedding_layer):
	#和lstur的新闻编码器相同参数不同
        hparams = self.hparams
        sequences_input_title = keras.Input(shape=(hparams.title_size,), dtype="int32")
        embedded_sequences_title = embedding_layer(sequences_input_title)

        y = layers.Dropout(hparams.dropout)(embedded_sequences_title)
        y = layers.Conv1D(
            hparams.filter_num,
            hparams.window_size,
            activation=hparams.cnn_activation,
            padding="same",
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        )(y)
        y = layers.Dropout(hparams.dropout)(y)
        pred_title = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)
        pred_title = layers.Reshape((1, hparams.filter_num))(pred_title)

        model = keras.Model(sequences_input_title, pred_title, name="title_encoder")
        return model

    def _build_bodyencoder(self, embedding_layer):
        hparams = self.hparams
        sequences_input_body = keras.Input(shape=(hparams.body_size,), dtype="int32")
        embedded_sequences_body = embedding_layer(sequences_input_body)

        y = layers.Dropout(hparams.dropout)(embedded_sequences_body)
        y = layers.Conv1D(
            hparams.filter_num,
            hparams.window_size,
            activation=hparams.cnn_activation,
            padding="same",
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        )(y)
        y = layers.Dropout(hparams.dropout)(y)
        pred_body = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)
        pred_body = layers.Reshape((1, hparams.filter_num))(pred_body)

        model = keras.Model(sequences_input_body, pred_body, name="body_encoder")
        return model
	def _build_lsturnaml(self):
        hparams = self.hparams

        his_input_title = keras.Input(
            shape=(hparams.his_size, hparams.title_size), dtype="int32"
        )
        his_input_body = keras.Input(
            shape=(hparams.his_size, hparams.body_size), dtype="int32"
        )

        pred_input_title = keras.Input(
            shape=(hparams.npratio + 1, hparams.title_size), dtype="int32"
        )
        pred_input_body = keras.Input(
            shape=(hparams.npratio + 1, hparams.body_size), dtype="int32"
        )
        pred_input_title_one = keras.Input(
            shape=(1, hparams.title_size,), dtype="int32"
        )
        pred_input_body_one = keras.Input(shape=(1, hparams.body_size,), dtype="int32")
        his_title_body = layers.Concatenate(axis=-1)(
            [his_input_title, his_input_body]
        )
        pred_title_body = layers.Concatenate(axis=-1)(
            [pred_input_title, pred_input_body]
        )

        pred_title_body_one = layers.Concatenate(axis=-1)(
            [
                pred_input_title_one,
                pred_input_body_one,
            ]
        )
        pred_title_body_one = layers.Reshape((-1,))(pred_title_body_one)

        embedding_layer = layers.Embedding(
            self.word2vec_embedding.shape[0],
            hparams.word_emb_dim,
            weights=[self.word2vec_embedding],
            trainable=True,
        )

        self.newsencoder = self._build_newsencoder(embedding_layer)
        self.userencoder = self._build_userencoder(self.newsencoder)

        user_present = self.userencoder(his_title_body)
        news_present = layers.TimeDistributed(self.newsencoder)(pred_title_body)
        news_present_one = self.newsencoder(pred_title_body_one)

        preds = layers.Dot(axes=-1)([news_present, user_present])
        preds = layers.Activation(activation="softmax")(preds)

        pred_one = layers.Dot(axes=-1)([news_present_one, user_present])
        pred_one = layers.Activation(activation="sigmoid")(pred_one)

        model = keras.Model(
            [
                his_input_title,
                his_input_body,
                pred_input_title,
                pred_input_body,
            ],
            preds,
        )

        scorer = keras.Model(
            [
                his_input_title,
                his_input_body,
                pred_input_title_one,
                pred_input_body_one,
            ],
            pred_one,
        )

        return model, scorer