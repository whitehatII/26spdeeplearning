import numpy as np
import tensorflow as tf
import collections
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import layers, optimizers, datasets
import os,sys,tqdm

import random
import string

def randomString(stringLength):
    """Generate a random string with the combination of lowercase and uppercase letters """

    letters = string.ascii_uppercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def get_batch(batch_size, length):
    batched_examples = [randomString(length) for i in range(batch_size)]
    enc_x = [[ord(ch)-ord('A')+1 for ch in list(exp)] for exp in batched_examples]
    y = [[o for o in reversed(e_idx)] for e_idx in enc_x]
    dec_x = [[0]+e_idx[:-1] for e_idx in y]
    return (batched_examples, tf.constant(enc_x, dtype=tf.int32),
            tf.constant(dec_x, dtype=tf.int32), tf.constant(y, dtype=tf.int32))
print(get_batch(2, 10))


class mySeq2SeqModel(keras.Model):
    def __init__(self):
        super(mySeq2SeqModel, self).__init__()
        self.v_sz = 27
        self.hidden = 128
        self.embed_layer = tf.keras.layers.Embedding(self.v_sz, 64)

        self.encoder_cell = tf.keras.layers.SimpleRNNCell(self.hidden)
        self.decoder_cell = tf.keras.layers.SimpleRNNCell(self.hidden)

        self.encoder = tf.keras.layers.RNN(self.encoder_cell,
                                           return_sequences=True, return_state=True)
        self.decoder = tf.keras.layers.RNN(self.decoder_cell,
                                           return_sequences=True, return_state=True)
        self.W1 = tf.keras.layers.Dense(self.hidden)
        self.W2 = tf.keras.layers.Dense(self.hidden)
        self.V = tf.keras.layers.Dense(1)
        self.dense_attn = tf.keras.layers.Dense(self.hidden)
        self.dense = tf.keras.layers.Dense(self.v_sz)

    def call(self, enc_ids, dec_ids):
        # encoder
        enc_emb = self.embed_layer(enc_ids)
        enc_out, enc_state = self.encoder(enc_emb)  # [B, T, H]

        # decoder
        dec_emb = self.embed_layer(dec_ids)
        dec_out, _ = self.decoder(dec_emb, initial_state=enc_state)  # [B, T, H]

        enc_expand = tf.expand_dims(enc_out, 1)  # [B, 1, T_enc, H]
        dec_expand = tf.expand_dims(dec_out, 2)  # [B, T_dec, 1, H]

        score = self.V(
            tf.nn.tanh(
                self.W1(enc_expand) + self.W2(dec_expand)
            )
        )  # [B, T_dec, T_enc, 1]

        score = tf.squeeze(score, axis=-1)  # [B, T_dec, T_enc]

        attn_weights = tf.nn.softmax(score, axis=-1)

        # 加权求和
        context = tf.matmul(attn_weights, enc_out)  # [B, T_dec, H]

        # 拼接
        concat = tf.concat([dec_out, context], axis=-1)
        logits = self.dense(concat)
        return logits

    @tf.function
    def encode(self, enc_ids):
        enc_emb = self.embed_layer(enc_ids)  # shape(b_sz, len, emb_sz)
        enc_out, enc_state = self.encoder(enc_emb)
        return enc_out, enc_state

    def get_next_token(self, x, state, enc_out):
        # embedding
        x = self.embed_layer(x)
        x = tf.expand_dims(x, 1)  # [B,1,H]

        # decoder
        dec_out, state = self.decoder(x, initial_state=state)  # [B,1,H]

        # ====== attention ======
        enc_expand = enc_out  # [B,T,H]
        dec_expand = dec_out  # [B,1,H]

        enc_expand = tf.expand_dims(enc_expand, 1)  # [B,1,T,H]
        dec_expand = tf.expand_dims(dec_expand, 2)  # [B,1,1,H]

        score = self.V(
            tf.nn.tanh(
                self.W1(enc_expand) + self.W2(dec_expand)
            )
        )  # [B,1,T,1]

        score = tf.squeeze(score, axis=-1)  # [B,1,T]
        attn_weights = tf.nn.softmax(score, axis=-1)

        context = tf.matmul(attn_weights, enc_out)  # [B,1,H]

        # 拼接
        concat = tf.concat([dec_out, context], axis=-1)
        out = self.dense(concat)

        out = tf.squeeze(out, axis=1)  # [B,vocab]
        return out, state

@tf.function
def compute_loss(logits, labels):
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)
    losses = tf.reduce_mean(losses)
    return losses

@tf.function
def train_one_step(model, optimizer, enc_x, dec_x, y):
    with tf.GradientTape() as tape:
        logits = model(enc_x, dec_x)
        loss = compute_loss(logits, y)

    # compute gradient
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def train(model, optimizer, seqlen):
    loss = 0.0
    accuracy = 0.0
    for step in range(2000):
        batched_examples, enc_x, dec_x, y = get_batch(32, seqlen)
        loss = train_one_step(model, optimizer, enc_x, dec_x, y)
        if step % 500 == 0:
            print('step', step, ': loss', loss.numpy())
    return loss

optimizer = optimizers.Adam(0.0005)
model = mySeq2SeqModel()
train(model, optimizer, seqlen=20)


def sequence_reversal():
    def decode(init_state, steps, enc_out):
        b_sz = tf.shape(init_state)[0]
        cur_token = tf.zeros(shape=[b_sz], dtype=tf.int32)
        state = init_state
        collect = []
        for i in range(steps):
            logits, state = model.get_next_token(cur_token, state, enc_out)
            cur_token = tf.argmax(logits, axis=-1)
            collect.append(tf.expand_dims(cur_token, axis=-1))
        out = tf.concat(collect, axis=-1).numpy()
        out = [''.join([chr(idx + ord('A') - 1) for idx in exp]) for exp in out]
        return out

    batched_examples, enc_x, _, _ = get_batch(32, 20)
    enc_out, state = model.encode(enc_x)
    return decode(state, enc_x.get_shape()[-1], enc_out), batched_examples


def is_reverse(seq, rev_seq):
    rev_seq_rev = ''.join([i for i in reversed(list(rev_seq))])
    if seq == rev_seq_rev:
        return True
    else:
        return False


print([is_reverse(*item) for item in list(zip(*sequence_reversal()))])
print(list(zip(*sequence_reversal())))