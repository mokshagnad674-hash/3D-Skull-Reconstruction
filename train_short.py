import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
from dataset_short import SkullDataset
from model_short import AttentionUNet3D

# Config
CT_DIR = r"d:\ML_DATASET\CT"
MASK_DIR = r"d:\ML_DATASET\mask"
EPOCHS, LR, BATCH_SIZE = 50, 5e-4, 1

def dice(p, t): 
    p, t = (p > 0.5).float(), (t > 0.5).float()
    return (2 * (p * t).sum() + 1e-7) / (p.sum() + t.sum() + 1e-7)

def train():
    device = torch.device("cuda")
    
    # Data
    files = [f for f in os.listdir(CT_DIR) if f in os.listdir(MASK_DIR)]
    train_f, val_f = train_test_split(files, test_size=0.2)
    
    train_loader = DataLoader(SkullDataset(CT_DIR, MASK_DIR, train_f), BATCH_SIZE, True)
    val_loader = DataLoader(SkullDataset(CT_DIR, MASK_DIR, val_f), 1)
    
    # Model
    model = AttentionUNet3D().to(device)
    opt = torch.optim.Adam(model.parameters(), LR)
    loss_fn = nn.BCELoss()
    
    best_dice = 0
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = train_dice = 0
        for ct, mask in train_loader:
            ct, mask = ct.to(device), mask.to(device).unsqueeze(1)
            opt.zero_grad()
            out = model(ct)
            loss = loss_fn(out, mask)
            loss.backward()
            opt.step()
            train_loss += loss.item()
            train_dice += dice(out, mask).item()
        
        # Validate
        model.eval()
        val_loss = val_dice = 0
        with torch.no_grad():
            for ct, mask in val_loader:
                ct, mask = ct.to(device), mask.to(device).unsqueeze(1)
                out = model(ct)
                val_loss += loss_fn(out, mask).item()
                val_dice += dice(out, mask).item()
        
        # Stats
        tl, td = train_loss/len(train_loader), train_dice/len(train_loader)
        vl, vd = val_loss/len(val_loader), val_dice/len(val_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | TL:{tl:.4f} TD:{td:.4f} | VL:{vl:.4f} VD:{vd:.4f}")
        
        if vd > best_dice:
            best_dice = vd
            torch.save(model.state_dict(), 'best_model_short.pth')
            print(f"✓ Best: {vd:.4f}")

if __name__ == "__main__":
    train()