import json
import re
import spacy
import tensorflow as tf

from Layers.TokenGrouper import TokenPOSGrouper


class TextIDMapper(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.nlp = spacy.load("pt_core_news_md")

        # we don't need to add [UNK] and ''
        self.words = [
            '[START]',
            '[END]',
        ]
        self.id_to_word = None
        self.word_to_id = None

    def load_vocab(self, json_path: str):
        with open(json_path, 'r', encoding='utf-8') as f:
            tokens_by_type = json.loads(f.read())

        self.words = tokens_by_type['tokens'][2:]
        self.word_to_id = tf.keras.layers.TextVectorization(
            max_tokens=5000,
            vocabulary=self.words,
            split='whitespace',
            standardize=self._standardize
        )
        self.id_to_word = tf.keras.layers.StringLookup(
            vocabulary=self.word_to_id.get_vocabulary(),
            mask_token='', oov_token='[UNK]',
            invert=True
        )

    def adapt(self, dataset: list):
        special_tokens = {}
        token_grouper = TokenPOSGrouper()
        for text in dataset:
            text = self._get_special_tokens(text.lower(), special_tokens)
            for token in self.nlp(str(text)):
                tk_text = special_tokens.get(token.text.lower(), token.text.lower())
                token_grouper.add(token, tk_text)

        for grammar_block in token_grouper.tokens_by_pos.values():
            for word_group in grammar_block.values():
                self.words.extend(word_group)

        self.word_to_id = tf.keras.layers.TextVectorization(
            max_tokens=5000,
            vocabulary=self.words,
            split='whitespace',
            standardize=self._standardize
        )
        self.id_to_word = tf.keras.layers.StringLookup(
            vocabulary=self.word_to_id.get_vocabulary(),
            mask_token='', oov_token='[UNK]',
            invert=True
        )

    def call(self, batch_data, inp_type):
        if inp_type == "input":
            return self.word_to_id(batch_data)
        else:
            return self.id_to_word(batch_data)

    def id_to_text(self, ids):
        return self.id_to_word(ids)

    def vocabulary_size(self):
        return self.word_to_id.vocabulary_size()

    def get_vocabulary(self):
        return self.word_to_id.get_vocabulary()

    @staticmethod
    def _get_special_tokens(text: str, dict_tokens: dict):
        special_tokens = re.findall(r"<.*?>", text)
        for sp_tk in special_tokens:
            parsed_tk = sp_tk.replace("<", "xxx").replace(">", "xxx")
            dict_tokens[parsed_tk] = sp_tk
            text = text.replace(sp_tk, parsed_tk)

        return text

    @staticmethod
    def _standardize(text: str):
        text = tf.strings.lower(text)
        text = tf.strings.strip(text)
        text = tf.strings.join(['[START]', text, '[END]'], separator=' ')

        return text
