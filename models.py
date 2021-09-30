import tensorflow as tf
import numpy as np
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from tensorflow import keras
from tensorflow.keras import layers

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'W1': self.W1,
            'W2': self.W2,
            'V': self.V
        })
        return config

    def call(self, values, query):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

def flatten_layers(root_layer):
    if isinstance(root_layer, tf.keras.layers.Layer):
        yield root_layer
    for layer in root_layer._layers:
        for sub_layer in flatten_layers(layer):
            yield sub_layer

def freeze_all_bert_layers(l_bert):
    """
    Freezes all but LayerNorm and adapter layers - see arXiv:1902.00751.
    """
    l_bert.trainable = False
    l_bert.encoders_layer.trainable = False

    for layer in l_bert.submodules:
        if layer.name in ["LayerNorm", "adapter-down", "adapter-up"]:
            layer.trainable = True
        else:
            layer.trainable = False

def freeze_bert_layers(l_bert):
    """
    Freezes all but LayerNorm and adapter layers - see arXiv:1902.00751.
    """
    for layer in flatten_layers(l_bert):
        if layer.name in ["LayerNorm", "adapter-down", "adapter-up"]:
            layer.trainable = True
        # elif len(layer._layers) == 0:
        #     layer.trainable = False
        else:
            layer.trainable = False
        l_bert.embeddings_layer.trainable = False

def encode_gru_with_attention(config, input):
    if config["text_use_attention"]:
        gru_forward = tf.keras.layers.GRU(config["rnn_layer_size"], return_sequences=True, return_state=True,
                                          activation='relu')

        attention_layer = BahdanauAttention(config["text_attention_size"])

        # apply forward GRU, attention
        forward_seq, forward_hidden_state = gru_forward(input)
        forward_attention_result, forward_attention_weights = attention_layer(forward_seq, forward_hidden_state)

        # concatenate attention results
        text_encoding = forward_attention_result
    else:
        gru_forward = tf.keras.layers.GRU(config["rnn_layer_size"], activation='relu')

        # apply forward GRU, attention
        text_encoding = gru_forward(input)

    return text_encoding

def encode_lstm_with_attention(config, input):
    if config["text_use_attention"]:
        lstm_forward = tf.keras.layers.LSTM(config["rnn_layer_size"], return_sequences=True, return_state=True,
                                          activation='tanh')

        attention_layer = BahdanauAttention(config["text_attention_size"])

        # apply forward GRU, attention
        forward_seq, forward_hidden_state, forward_cell_state = lstm_forward(input)
        forward_attention_result, forward_attention_weights = attention_layer(forward_seq, forward_hidden_state)

        # concatenate attention results
        text_encoding = forward_attention_result
    else:
        lstm_forward = tf.keras.layers.LSTM(config["rnn_layer_size"])

        # apply forward GRU, attention
        text_encoding = lstm_forward(input)

    return text_encoding

def encode_bigru_with_attention(config, input):
    if config["text_use_attention"]:
        gru_forward = tf.keras.layers.GRU(config["rnn_layer_size"], return_sequences=True, return_state=True,
                                          activation='tanh')
        gru_backward = tf.keras.layers.GRU(config["rnn_layer_size"], go_backwards=True, return_sequences=True,
                                           return_state=True, activation='tanh')

        attention_layer = BahdanauAttention(config["text_attention_size"])

        # apply forward GRU, attention
        forward_seq, forward_hidden_state = gru_forward(input)
        forward_attention_result, forward_attention_weights = attention_layer(forward_seq, forward_hidden_state)

        # apply backward GRU, attention
        backward_seq, backward_hidden_state = gru_backward(input)
        backward_attention_result, backward_attention_weights = attention_layer(backward_seq, backward_hidden_state)

        # concatenate attention results
        text_encoding = tf.keras.layers.concatenate([backward_attention_result, forward_attention_result])
    else:
        gru_forward = tf.keras.layers.GRU(config["rnn_layer_size"], activation='tanh')
        gru_backward = tf.keras.layers.GRU(config["rnn_layer_size"], go_backwards=True, activation='tanh')

        # apply forward GRU, attention
        forward_hidden_state = gru_forward(input)

        backward_hidden_state = gru_backward(input)

        # concatenate attention results
        text_encoding = tf.keras.layers.concatenate([backward_hidden_state, forward_hidden_state])
    return text_encoding

def encode_text_with_bert(config, input_layer, bert):
    bert_output = bert(input_layer)

    if config["rnn_type"] == 'gru':
        text_encoding = encode_gru_with_attention(config, bert_output)
    elif config["rnn_type"] == 'lstm':
        text_encoding = encode_lstm_with_attention(config, bert_output)
    elif config["rnn_type"] == 'bi-gru':
        text_encoding = encode_bigru_with_attention(config, bert_output)
    else:
        text_encoding = tf.keras.layers.Convolution1D(filters=100, kernel_size=5, padding='same', activation='tanh')(
            bert_output)
        text_encoding = tf.keras.layers.Convolution1D(filters=80, kernel_size=5, padding='same', activation='tanh')(
            text_encoding)
        text_encoding = tf.keras.layers.Convolution1D(filters=50, kernel_size=5, padding='same', activation='tanh')(
            text_encoding)
        text_encoding = tf.keras.layers.AvgPool1D()(text_encoding)
        text_encoding = tf.keras.layers.Flatten()(text_encoding)


    return text_encoding

def encode_text_with_hateword_list(config, input_layer):
    hate_words_encoding = tf.keras.layers.Dense(1493, name="hatewords_norm_layer_1")(input_layer)
    hate_words_encoding = tf.keras.layers.BatchNormalization()(hate_words_encoding)
    hate_words_encoding = tf.keras.layers.Activation("relu")(hate_words_encoding)

    hate_words_encoding2 = tf.keras.layers.Dense(512, name="hatewords_norm_layer_2")(hate_words_encoding)
    hate_words_encoding2 = tf.keras.layers.BatchNormalization()(hate_words_encoding2)
    hate_words_encoding2 = tf.keras.layers.Activation("relu")(hate_words_encoding2)

    return hate_words_encoding2


def encode_text_with_char_embeddings(config, input_layer):
    char_embedding_layer = tf.keras.layers.Embedding(input_dim=config["char_size"], trainable=True, output_dim=50, embeddings_initializer='uniform', name="char_embs")

    char_emb_out = char_embedding_layer(input_layer)

    if config["text_encoder"] == 'gru':
        text_encoding = encode_gru_with_attention(config, char_emb_out)
    elif config["text_encoder"] == 'lstm':
        text_encoding = encode_lstm_with_attention(config, char_emb_out)
    elif config["text_encoder"] == 'bi-gru':
        text_encoding = encode_bigru_with_attention(config, char_emb_out)
    else:
        text_encoding = tf.keras.layers.Convolution1D(filters=40, kernel_size=5, padding='same', activation='relu')(
            char_emb_out)
        text_encoding = tf.keras.layers.Convolution1D(filters=20, kernel_size=5, padding='same', activation='relu')(
            text_encoding)
        text_encoding = tf.keras.layers.Convolution1D(filters=10, kernel_size=5, padding='same', activation='relu')(
            text_encoding)
        text_encoding = tf.keras.layers.AvgPool1D()(text_encoding)
        text_encoding = tf.keras.layers.Flatten()(text_encoding)

    return text_encoding

def get_fusion_layer_sizes(individual_layer_size, num_modalitiies):
    layer_sizes = []
    first_layer_size = individual_layer_size * num_modalitiies
    layer_sizes.append(first_layer_size)

    return layer_sizes

def encode_inputs(config, bert_config_file, bert_check_point_file, adapter_size=None):
    """Creates a classification model."""

    inputs = []
    modality_outputs = []
    image_models = []

    has_bert_modality = False
    if "bert" in config["text_models"]:
        has_bert_modality = True

    if has_bert_modality:
        with tf.io.gfile.GFile(bert_config_file, "r") as reader:
            bc = StockBertConfig.from_json_string(reader.read())
            bert_params = map_stock_config_to_params(bc)
            bert_params.adapter_size = adapter_size
            bert = BertModelLayer.from_params(bert_params, name="bert")
    else:
        bert = None

    if "bert" in config["text_models"]:
        tweet_text_bert_input = tf.keras.layers.Input(shape=(config['tweet_text_seq_len'],), dtype='int32',
                                                      name="text_bert")
        inputs.append(tweet_text_bert_input)

        # encode with BERT
        tweet_text_encoding = encode_text_with_bert(config, tweet_text_bert_input, bert)
        modality_outputs.append(tweet_text_encoding)

    if "hate_words" in config["text_models"]:
        tweet_text_hate_words_input = tf.keras.layers.Input(shape=(1493,), dtype='int32',
                                                            name="text_hate_words")
        inputs.append(tweet_text_hate_words_input)

        # encode hatewords
        tweet_text_hate_words_encoding = encode_text_with_hateword_list(config, tweet_text_hate_words_input)
        modality_outputs.append(tweet_text_hate_words_encoding)

    if "char_emb" in config["text_models"]:
        tweet_text_char_input = tf.keras.layers.Input(shape=(config['tweet_text_char_len'],), dtype='int32',
                                                      name="text_char_emb")
        inputs.append(tweet_text_char_input)

        # encode with char embeddings
        tweet_text_char_encoding = encode_text_with_char_embeddings(config, tweet_text_char_input)
        modality_outputs.append(tweet_text_char_encoding)


    return inputs, modality_outputs, has_bert_modality, bert

def get_model(config, bert_config_file, bert_check_point_file, adapter_size=None):
    """Creates a classification model."""
    inputs, modality_outputs, has_bert_modality, bert = encode_inputs(config, bert_config_file, bert_check_point_file, adapter_size)
    outputs = []

    if len(modality_outputs) > 1:
        concat_embedding = tf.keras.layers.concatenate(modality_outputs)
    else:
        concat_embedding = modality_outputs[0]

    fusion_layer_output = concat_embedding

    # fusion_layer_size = len(modality_outputs) * config['feature_normalization_layer_size']
    fusion_layer_size = fusion_layer_output.shape[1]
    counter = 1
    while fusion_layer_size > config['min_feature_normalization_layer_size']:

        fusion_layer_output = tf.keras.layers.Dense(fusion_layer_size, name="fusion_layer_"+str(counter))(fusion_layer_output)
        batch_norm_layer_output = tf.keras.layers.BatchNormalization(name="batch_norm_layer_"+str(counter))(fusion_layer_output)
        activation_layer_output = tf.keras.layers.Activation("relu", name="relu_layer_"+str(counter))(batch_norm_layer_output)

        if counter == 1:
            adapted_layer_size = np.power(2, int(np.log2(fusion_layer_size)))
            if adapted_layer_size == fusion_layer_size:
                fusion_layer_size /= 2
            else:
                fusion_layer_size = adapted_layer_size

        else:
            # decrease by half
            fusion_layer_size /= 2

        counter+=1

        fusion_layer_output = activation_layer_output

    last_layer_output = tf.keras.layers.Dense(units=2, activation="softmax", name='output_label')(fusion_layer_output)
    outputs.append(last_layer_output)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)


    # load the pre-trained model weights, if BERT is used as a modality
    if has_bert_modality:
        load_stock_weights(bert, bert_check_point_file)

        # freeze weights if adapter-BERT is used
        if adapter_size is not None:
            freeze_bert_layers(bert)
        else:
            freeze_all_bert_layers(bert)

    if config["optimizer"] == "sgd":
        optimizer = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9)
    elif config["optimizer"] == "rmsprop":
        optimizer = tf.keras.optimizers.RMSProp()
    elif config["optimizer"] == "adagrad":
        optimizer = tf.keras.optimizers.Adagrad()
    else:
        optimizer = tf.keras.optimizers.Adam(0.0001)
    # Enable Mixed Precision for faster computation, less memory
    # optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")])

    model.summary()

    return model
