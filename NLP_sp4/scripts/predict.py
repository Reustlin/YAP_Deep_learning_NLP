import torch
import numpy as np
from tqdm.auto import tqdm
from scripts.model import CaloriesModel
from scripts.config import CFG

def load_model(weights_path, device=CFG.DEVICE):
    model = CaloriesModel().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

def predict(model, loader, device):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predict"):
            image = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            mass = batch["mass"].to(device)

            preds = model(image, input_ids, attention_mask, mass)
            predictions.extend(preds.cpu().numpy())
            targets.extend(batch["target"].numpy())
    return np.array(predictions), np.array(targets)
