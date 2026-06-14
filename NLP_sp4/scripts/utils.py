import random
import numpy as np
import torch
import pandas as pd
from scripts.config import CFG

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_ingredient_mapping(data_path):
    """Загружает файл ingredients.csv и возвращает словарь id -> ингредиент"""
    ingredients = pd.read_csv(f"{data_path}/ingredients.csv")
    ingredients["id"] = ingredients["id"].astype(str).str.extract(r"(\d+)$")[0].astype(int)
    id2ingr = dict(zip(ingredients["id"], ingredients["ingr"]))
    return id2ingr

def decode_ingredients(text, id2ingr):
    if pd.isna(text):
        return ""
    ids = text.split(";")
    clean_ids = [int(x.split("_")[-1]) for x in ids if x]
    names = [id2ingr.get(i, "") for i in clean_ids]
    names = [n for n in names if n]
    return ", ".join(names)
