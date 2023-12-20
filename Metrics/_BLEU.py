import tensorflow as tf


class BLEU(tf.keras.metrics.Metric):
    def __init__(self, text_processor, name='BLEU', **kwargs):
        super(BLEU, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        self.smooth_epsilon = tf.constant(1e-6)

        self.input_to_word = tf.keras.layers.StringLookup(
            vocabulary=text_processor.get_vocabulary(),
            mask_token='', oov_token='[UNK]',
            invert=True)

        self.output_to_word = tf.keras.layers.StringLookup(
            vocabulary=text_processor.get_vocabulary(),
            mask_token='', oov_token='[UNK]',
            invert=True)

        self.word_to_tokens = text_processor

    def update_state(self, batch_true, batch_pred, sample_weight=None):
        # Implemente a lógica de atualização do estado da métrica aqui
        # y_true: Rótulos verdadeiros
        # y_pred: Predições do modelo
        # sample_weight: Peso opcional para amostras

        # Exemplo: Incrementa o total e a contagem

        # y_true = tf.make_ndarray(tf.make_tensor_proto(y_true))
        # y_pred = tf.make_ndarray(tf.make_tensor_proto(y_pred))

        # Calcula a média dos escores BLEU
        for y_true, y_pred in zip(batch_true, batch_pred):
            average_bleu = self.calculate_bleu(y_true, y_pred)
            print(average_bleu)

            self.total.assign_add(tf.reduce_sum(average_bleu))
            self.count.assign_add(tf.cast(tf.size(average_bleu), dtype=tf.float32))

    def calculate_bleu(self, reference, candidate, max_order=4):
        # Tokenização
        reference_tokens = self.output_to_word(reference)
        candidate_tokens = self.output_to_word(candidate)

        # Contagem de n-gramas
        clipped_counts = []
        total_counts = []
        for order in range(1, max_order + 1):
            ref_ngrams = self.word_to_tokens(tf.strings.ngrams(reference_tokens, order, separator=""))
            cand_ngrams = self.word_to_tokens(tf.strings.ngrams(candidate_tokens, order, separator=""))

            clipped_count = tf.reduce_sum(tf.minimum(ref_ngrams, cand_ngrams), axis=1)
            total_count = tf.reduce_sum(ref_ngrams, axis=1)

            clipped_counts.append(clipped_count)
            total_counts.append(total_count)

        clipped_counts = tf.convert_to_tensor(tf.keras.utils.pad_sequences(clipped_counts),
                                              dtype=self.smooth_epsilon.dtype)
        total_counts = tf.convert_to_tensor(tf.keras.utils.pad_sequences(total_counts), dtype=self.smooth_epsilon.dtype)

        clipped_counts = tf.stack(clipped_counts, axis=1)
        total_counts = tf.stack(total_counts, axis=1)

        # Precisão dos n-gramas
        precisions = (clipped_counts + self.smooth_epsilon) / (tf.maximum(total_counts, 1) + self.smooth_epsilon)

        # Média das precisões
        avg_precision = tf.reduce_mean(precisions, axis=1)

        # Penalidade de Brevidade (BP)
        ref_length = tf.reduce_sum(tf.strings.length(reference_tokens))
        cand_length = tf.reduce_sum(tf.strings.length(candidate_tokens))
        brevity_penalty = tf.exp(1 - tf.minimum(ref_length / cand_length, 1.0))
        exp = tf.cast(tf.exp(tf.reduce_sum(tf.math.log(avg_precision)) / max_order), dtype=brevity_penalty.dtype)

        # BLEU Final
        bleu = tf.cast(brevity_penalty * exp, dtype=tf.float32)

        return bleu

    def result(self):
        # Implemente a lógica para calcular o resultado final da métrica
        # Neste exemplo, retorna a média
        return self.total / self.count if self.count != 0 else 0.0

    def reset_states(self):
        # Implemente a lógica para redefinir o estado da métrica entre as épocas
        self.total.assign(0.0)
        self.count.assign(0.0)

'''
out_to_word = tf.keras.layers.StringLookup(
        vocabulary=text_processor.get_vocabulary(),
        mask_token='', oov_token='[UNK]',
        invert=True)

inp_to_word = tf.keras.layers.StringLookup(
        vocabulary=input_text_processor.get_vocabulary(),
        mask_token='', oov_token='[UNK]',
        invert=True)

for (pt, en), en_labels in train_ds.take(1):
  break

#print(inp_to_word(pt))
#print('PRED:', out_to_word(en.numpy()))
#print('LABELS:', out_to_word(en_labels.numpy()))
print(BLEU().update_state(en, en_labels))
'''