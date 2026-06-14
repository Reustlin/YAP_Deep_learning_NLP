import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm.auto import tqdm
from sklearn.metrics import mean_absolute_error
from scripts.config import CFG
from scripts.model import CaloriesModel

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Train"):
        image = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        mass = batch["mass"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()
        preds = model(image, input_ids, attention_mask, mass)
        loss = criterion(preds, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def valid_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds_all, targets_all = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Valid"):
            image = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            mass = batch["mass"].to(device)
            target = batch["target"].to(device)

            preds = model(image, input_ids, attention_mask, mass)
            loss = criterion(preds, target)
            total_loss += loss.item()
            preds_all.append(preds.cpu())
            targets_all.append(target.cpu())

    preds_all = torch.cat(preds_all).numpy()
    targets_all = torch.cat(targets_all).numpy()
    mae = mean_absolute_error(targets_all, preds_all)
    return total_loss / len(loader), mae

def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, epochs, device, save_path="best_model.pth"):
    best_mae = float("inf")
    history = {"train_loss": [], "valid_loss": [], "valid_mae": []}

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae = valid_epoch(model, val_loader, criterion, device)
        if scheduler:
            scheduler.step(val_mae)

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(val_loss)
        history["valid_mae"].append(val_mae)

        print(f"Train Loss: {train_loss:.4f} | Valid Loss: {val_loss:.4f} | Valid MAE: {val_mae:.4f}")

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), save_path)
            print("Saved best model!")
    return history
