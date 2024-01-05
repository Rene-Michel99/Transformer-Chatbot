import tensorflow as tf
import argparse

from Layers import TextIDMapper
from Models import Chatbot
from Utils import download_trained_weights


def start_inference():
    # Use a breakpoint in the code line below to debug your script.
    new_data = []
    weights_path = download_trained_weights("./logs")
    text_processor = TextIDMapper()
    text_processor.load_vocab('./Data/tokens_by_type.json')

    chatbot = Chatbot(text_processor)
    chatbot(tf.convert_to_tensor((['olá'], ['olá'])))
    chatbot.load_weights(weights_path)

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


def main():
    # Criação do objeto parser
    parser = argparse.ArgumentParser(description='Script que realiza a inferência ou treinamento do Chatbot.')

    # Adição de argumentos
    parser.add_argument('mode', type=str, help='Modo inference ou training')
    parser.add_argument('--dataset', type=str, help='Caminho até a localização do dataset .db')

    # Parse dos argumentos
    args = parser.parse_args()

    # Chamada da função com os parâmetros fornecidos
    if args.mode == "inference":
        start_inference()
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    main()
