import re
import tensorflow as tf

from Layers import TextIDMapper, Transformer


class Chatbot(tf.keras.models.Model):
    def __init__(
            self,
            text_processor: TextIDMapper,
            num_layers: int = 4,
            d_model: int = 128,
            num_heads: int = 8,
            dff: int = 512,
            dropout_rate: float = 0.1
    ):
        super().__init__()
        self.text_processor = text_processor
        self.transformer = Transformer(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            input_vocab_size=text_processor.vocabulary_size(),
            target_vocab_size=text_processor.vocabulary_size(),
            dropout_rate=dropout_rate,
        )
        self.MAX_TOKENS = 128

    def call(self, inputs, training=True):
        questions, answers = inputs
        batch_data, answers_labels = self._prepare_batch(questions, answers)

        return self.transformer(batch_data), answers_labels

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        with tf.GradientTape() as tape:
            y_pred, y_true = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y_true, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y_true, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def tokens_to_text(self, tokens):
        words = self.text_processor(tokens, 'output')
        result = tf.strings.reduce_join(words, axis=-1, separator=' ')
        result = tf.strings.regex_replace(result, '^ *\[START\] *', '')
        result = tf.strings.regex_replace(result, ' *\[END\] *$', '')

        return result

    def predict(self, sentence: str, max_length=128):
        # The input sentence is Portuguese, hence adding the `[START]` and `[END]` tokens.

        sentence = tf.constant(self._preprocess_input_sentence(sentence))
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]

        sentence = self.text_processor(sentence, 'input')

        encoder_input = sentence

        # As the output language is English, initialize the output with the
        # English `[START]` token.
        start_end = self.text_processor([''], "input")[0]
        start = start_end[0][tf.newaxis]
        end = start_end[1][tf.newaxis]

        # `tf.TensorArray` is required here (instead of a Python list), so that the
        # dynamic-loop can be traced by `tf.function`.
        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            predictions = self.transformer([encoder_input, output], training=False)

            # Select the last token from the `seq_len` dimension.
            predictions = predictions[:, -1:, :]  # Shape `(batch_size, 1, vocab_size)`.

            predicted_id = tf.argmax(predictions, axis=-1)

            # Concatenate the `predicted_id` to the output which is given to the
            # decoder as its input.
            output_array = output_array.write(i + 1, predicted_id[0])
            if predicted_id == end:
                break

        output = tf.transpose(output_array.stack())
        # The output shape is `(1, tokens)`.
        text = self.tokens_to_text(output)[0]  # Shape: `()`.

        # tokens = tokenizers.en.lookup(output)[0]

        # `tf.function` prevents us from using the attention_weights that were
        # calculated on the last iteration of the loop.
        # So, recalculate them outside the loop.
        self.transformer([encoder_input, output[:, :-1]], training=False)
        attention_weights = self.transformer.decoder.last_attn_scores

        return text, None, attention_weights

    def _prepare_batch(self, query, answer):
        query = self.text_processor(query, 'input')  # Output is ragged.
        query = query[:, :self.MAX_TOKENS]  # Trim to MAX_TOKENS.

        answer = self.text_processor(answer, 'input')
        answer = answer[:, :(self.MAX_TOKENS + 1)]
        answer_inputs = answer[:, :-1]  # Drop the [END] tokens
        answer_labels = answer[:, 1:]  # Drop the [START] tokens

        return (query, answer_inputs), answer_labels

    @staticmethod
    def _preprocess_input_sentence(text: str):
        text = text.lower()
        text = re.sub(r"([.!,?])", r" \1", text)
        text = text.replace('(', ' ( ')
        text = text.replace(')', ' ) ')
        text = text.replace('/', ' / ')
        text = text.replace('por que', 'porque')
        text = text.replace('por quê', 'porque')
        text = text.replace('porquê', 'porque')
        print('Processed sentence:', text)

        return text.strip()

