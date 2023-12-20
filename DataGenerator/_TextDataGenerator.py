import numpy as np
import tensorflow as tf


class TextDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, pairs: list, batch_size: int):
        self.pairs = pairs
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.pairs) / self.batch_size))

    def __getitem__(self, idx: int):
        start_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size

        batch_x = tf.convert_to_tensor(self.pairs[start_idx:end_idx, 0])
        batch_y = tf.convert_to_tensor(self.pairs[start_idx:end_idx, 1])

        return (batch_x, batch_y), batch_y
