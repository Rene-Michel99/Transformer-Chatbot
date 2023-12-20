import os
import time
import shutil
import urllib
import zipfile


DEFAULT_TRAINED_WEIGHTS_URL = "https://github.com/Rene-Michel99/Transformer-Chatbot/releases/download/trained_weights/model_weights.zip"


def unzip_file(file_path: str):
    split_path = file_path.split('/')
    download_dir = '/'.join(split_path[:len(split_path) - 1])
    dataset_name = split_path[-1].split('.')[0]
    unzipped_path_file = os.path.join(download_dir, dataset_name)
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(unzipped_path_file)

    while not os.path.exists(unzipped_path_file):
        time.sleep(1)

    return unzipped_path_file


def download_trained_weights(verbose=1) -> str:
    """Download COCO trained weights from Releases.

    coco_model_path: local path of COCO trained weights
    """
    trained_weights_path = "./logs/trained_weights"
    if not os.path.exists("./logs"):
        os.system("mkdir logs")
    if os.path.exists(trained_weights_path):
        return os.path.join(trained_weights_path, 'model_weights', 'model_weights.tf')

    trained_weights_path = "./logs/trained_weights.zip"
    if verbose > 0:
        print("Downloading pretrained model to " + DEFAULT_TRAINED_WEIGHTS_URL + " ...")
    with urllib.request.urlopen(DEFAULT_TRAINED_WEIGHTS_URL) as resp, open(trained_weights_path, 'wb') as out:
        shutil.copyfileobj(resp, out)

    if verbose > 0:
        print("... done downloading pretrained model!")

    unzip_path = unzip_file(trained_weights_path)
    return os.path.join(unzip_path, 'model_weights', 'model_weights.tf')
