# -*- coding:utf8 -*-
import numpy as np

def batch(inputs, max_sequence_length=None):
    '''
    将所有输入填充为max_sequence_length的长度,并返回一个形状为[max_time, batch_size]的输入矩阵
    :param inputs: list of sentences(integer lists)
    :param max_sequence_length: integer specifying how large should 'max_time' dimension be.
                                If None, maximum sequence length would be used.
    :return:    1. inputs_time_major: input sentences transformed into time-major matrix
                                        (shape [max_time, batch_size]) padded with 0s
                2. sequence_lengths: batch-sized list of integer specifying amount of active
                                        time steps in each input sequence
    '''

    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)

    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)

    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)

    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i,j] = element

    inputs_time_major = inputs_batch_major.swapaxes(0,1)

    return inputs_time_major, sequence_lengths


def random_sequences(length_from, length_to,
                     vocab_lower, vocab_upper,
                     batch_size):
    '''
    生成长度在[length_from, length_to]之间,包含字典中[vocab_lower, vocab_upper]范围的序列数据
    '''

    if length_from > length_to:
        raise ValueError('length_from > length_to')

    def random_length():
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)

    while True:
        yield [
            np.random.randint(low=vocab_lower,
                              high=vocab_upper,
                              size=random_length()).tolist()
            for _ in range(batch_size)
        ]