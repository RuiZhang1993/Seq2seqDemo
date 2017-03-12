# -*- coding:utf8 -*-

x = [[5,7,8],[6,3],[3],[1]]

# 测试辅助函数
import helpers
xt, xlen = helpers.batch(x)

# tensorflow基本设置/参数初始化等
import numpy as np
import tensorflow as tf

tf.reset_default_graph()
sess = tf.InteractiveSession()

PAD = 0
EOS = 1

vocab_size = 10
input_embedding_size = 20

encoder_hidden_units = 20
decoder_hidden_units = encoder_hidden_units

encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')

# 构建编码器Encoder
from tensorflow.contrib.rnn import LSTMCell, InputProjectionWrapper, OutputProjectionWrapper

encoder_cell = LSTMCell(encoder_hidden_units)

encoder_cell = InputProjectionWrapper(cell=encoder_cell, num_proj=input_embedding_size)

encoder_inputs_onehot = tf.one_hot(encoder_inputs, depth=vocab_size, dtype=tf.float32)

encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
    encoder_cell, encoder_inputs_onehot,
    dtype=tf.float32, time_major=True,
)

del encoder_outputs

print encoder_final_state

# 构建解码器Decoder
decoder_cell = LSTMCell(decoder_hidden_units)

decoder_cell = InputProjectionWrapper(cell=decoder_cell, num_proj=input_embedding_size)

decoder_cell = OutputProjectionWrapper(cell=decoder_cell, output_size=vocab_size)

decoder_inputs_onehot = tf.one_hot(decoder_inputs, depth=vocab_size, dtype=tf.float32)

decoder_logits, decoder_final_state = tf.nn.dynamic_rnn(
    decoder_cell, decoder_inputs_onehot,
    initial_state=encoder_final_state,
    dtype=tf.float32, time_major=True, scope="plain_decoder",
)

decoder_prediction = tf.argmax(decoder_logits,2)

print decoder_logits
print decoder_prediction

# 设置模型优化方法
stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
    logits=decoder_logits,
)

loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss)

sess.run(tf.global_variables_initializer())

# 小数据测试
batch_ = [[6], [3,4], [9,8,7]]
batch_, batch_length = helpers.batch(batch_)
print('batch_encoded:\n' + str(batch_))

din_, dlen_ = helpers.batch(np.ones(shape=(3,1), dtype=np.int32),
                           max_sequence_length=4)
print('decoder inputs:\n' + str(din_))

pred_ = sess.run(decoder_prediction,
                 feed_dict={
                     encoder_inputs: batch_,
                     decoder_inputs: din_,
                 })
print('decoder predictions:\n' + str(pred_))

# 针对随机生成的语料进行训练
batch_size = 100

batches = helpers.random_sequences(length_from=3, length_to=8,
                                  vocab_lower=2, vocab_upper=10,
                                  batch_size=batch_size)
print('head of the batch:')
for seq in next(batches)[:10]:
    print(seq)

def next_feed():
    batch = next(batches)
    encoder_inputs_, _ = helpers.batch(batch)
    decoder_targets_, _ = helpers.batch(
        [(sequence) + [EOS] + [PAD] * 2 for sequence in batch]
    )
    decoder_inputs_, _ = helpers.batch(
        [[EOS] + (sequence) + [PAD] * 2 for sequence in batch]
    )

    return {
        encoder_inputs: encoder_inputs_,
        decoder_inputs: decoder_inputs_,
        decoder_targets: decoder_targets_,
    }

loss_track = []

max_batches = 3001
batches_in_epoch = 1000

try:
    for batch in range(max_batches):
        fd = next_feed()
        _, l = sess.run([train_op, loss], fd)
        loss_track.append(l)

        if batch == 0 or batch % batches_in_epoch == 0:
            print('batch {}'.format(batch))
            print('  minibatch loss: {}').format(sess.run(loss, fd))
            predict_ = sess.run(decoder_prediction, fd)
            for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
                print('  sample {}:'.format(i+1))
                print('    input     > {}'.format(inp))
                print('    predicted > {}'.format(pred))
                if i >= 2:
                    break
                print ""
except KeyboardInterrupt:
    print('training interrupted')
