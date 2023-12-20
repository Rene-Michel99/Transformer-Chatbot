import argparse

from Models import Chatbot


def start_inference():
    # Use a breakpoint in the code line below to debug your script.
    new_data = []

    while True:
        sentence = str(input("User: "))
        if sentence == 'quit':
            break

        translated_text, translated_tokens, attention_weights = predict(model, tf.constant(sentence))
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
    parser.add_argument('dataset', type=str, help='Caminho até a localização do dataset .db')

    # Parse dos argumentos
    args = parser.parse_args()

    # Chamada da função com os parâmetros fornecidos
    if args.mode == "inference":
        resultado = start_inference()
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    main()
