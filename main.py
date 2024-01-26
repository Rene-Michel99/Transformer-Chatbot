import re
import json
import sqlite3
import numpy as np
import tensorflow as tf
import argparse

from DataGenerator import TextDataGenerator
from Layers import TextIDMapper
from Losses import masked_loss
from Metrics import masked_accuracy
from Models import Chatbot
from Schedules import CustomSchedule
from Utils import download_trained_weights


def _preprocess(text):
    text = text.lower()
    text = re.sub(r"([!,?:])", r" \1", text)
    text = text.replace('(', ' ( ')
    text = text.replace(')', ' ) ')
    text = text.replace('/', ' / ')
    text = text.replace('`', ' ` ')
    text = text.replace('\\n', ' \\n ')
    text = text.replace('por que', 'porque')
    text = text.replace('por quê', 'porque')
    text = text.replace('porquê', 'porque')

    endpoints = re.findall(r"\w+\. ", text)
    for endpoint in endpoints:
        text = text.replace(endpoint, endpoint.replace('. ', ' . '))

    if text.endswith('.'):
        text = text[:len(text) - 1] + ' . '

    return text.strip()


def _get_data(dataset):
    conn = sqlite3.connect(dataset)
    cur = conn.execute("SELECT id, question, answer FROM Conversation;")
    max_sentence = 0
    data = []
    for row in cur.fetchall():
        pair = []
        for text in row[1:]:
            text = _preprocess(text)
            if text[-1] not in ['.', ',', ';', '!', '?']:
                text += ' . '

            if len(text.split()) > max_sentence:
                max_sentence = len(text.split())

            if len(text) > 0:
                pair.append(text)

        if len(pair) == 2:
            data.append(pair)

    return np.array(data)


def start_inference():
    # Use a breakpoint in the code line below to debug your script.
    new_data = []
    weights_path = download_trained_weights("./logs")
    text_processor = TextIDMapper()
    text_processor.load_vocab('./Data/tokens_by_type.json')

    chatbot = Chatbot(text_processor)
    chatbot(tf.convert_to_tensor((['olá'], ['olá'])))
    chatbot.load_weights(weights_path)

    print("\n\n\n")

    while True:
        sentence = str(input("User: "))
        if sentence == 'quit':
            break

        translated_text, translated_tokens, attention_weights = chatbot.predict(sentence)
        response = translated_text.numpy().decode('utf-8')
        print("Bot: ", response)
        not_correct = 'n' in str(input("Response is correct? [y] [n]: "))
        if not_correct:
            new_data.append([sentence, str(input("Expected response: "))])


def start_training(dataset, model_save_path):
    pairs = _get_data(dataset)
    text_processor = TextIDMapper()
    text_processor.adapt(pairs.reshape(-1))

    with open('tokens.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps({'tokens': text_processor.get_vocabulary()}))

    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.1

    transformer = Chatbot(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        text_processor=text_processor,
        dropout_rate=dropout_rate,
    )
    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

    model = Chatbot(text_processor, transformer)

    data_generator = TextDataGenerator(pairs, 64)
    model.compile(
        loss=masked_loss,
        optimizer=optimizer,
        metrics=[masked_accuracy]
    )

    model.fit(
        data_generator,
        epochs=15,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, monitor="loss")]
    )
    model(tf.convert_to_tensor((['olá'], ['olá'])), training=False)
    model.save_weights(model_save_path)


def main():
    # Criação do objeto parser
    parser = argparse.ArgumentParser(description='Script que realiza a inferência ou treinamento do Chatbot.')

    # Adição de argumentos
    parser.add_argument('mode', type=str, help='Modo inference ou training')
    parser.add_argument('--dataset', type=str, help='Caminho até a localização do dataset .db')
    parser.add_argument('--model_save_path', type=str, help='Caminho para salver o modelo', default="./logs/trained_model_weights.h5")

    # Parse dos argumentos
    args = parser.parse_args()

    # Chamada da função com os parâmetros fornecidos
    if args.mode == "inference":
        start_inference()
    else:
        start_training(args.dataset, args.model_save_path)


if __name__ == "__main__":
    main()
