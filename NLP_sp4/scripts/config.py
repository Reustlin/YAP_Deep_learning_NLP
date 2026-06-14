import torch

class CFG:
    DATA_PATH = "/path/to/your/data"          
    IMAGES_PATH = f"{DATA_PATH}/images"

    IMAGE_SIZE = 224
    MAX_LEN = 128
    BATCH_SIZE = 4
    EPOCHS = 10

    IMAGE_LR = 1e-4
    TEXT_LR = 2e-5
    CLASSIFIER_LR = 1e-3

    SEED = 42

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
