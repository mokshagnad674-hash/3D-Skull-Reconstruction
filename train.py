import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from config import *
from dataset import SkullDataset
from model import AttentionUNet3D
from metrics import calculate_map, iou_score, precision_recall
import gc
import glob
import json

def dice(pred, target, eps=1e-7):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    inter = (pred * target).sum()
    return (2 * inter + eps) / (pred.sum() + target.sum() + eps)

def accuracy(pred, target):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    correct = (pred == target).float().sum()
    total = target.numel()
    
    pos_mask = target == 1
    neg_mask = target == 0
    
    if pos_mask.sum() > 0:
        pos_acc = (pred[pos_mask] == target[pos_mask]).float().mean()
    else:
        pos_acc = 0
        
    if neg_mask.sum() > 0:
        neg_acc = (pred[neg_mask] == target[neg_mask]).float().mean()
    else:
        neg_acc = 0

    return correct / total

def train():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA not available. training on CPU will be slow.")
    print("Device:", device)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Track best model performance
    best_accuracy = 0.0
    best_epoch = 0
    
    # History tracking
    history = {
        'epochs': [],
        'train_loss': [],
        'train_dice': [],
        'val_loss': [],
        'val_dice': [],
        'val_map': [],
        'val_iou': [],
        'val_precision': [],
        'val_recall': []
    }

    files = sorted(os.listdir(CT_DIR))
    train_f, val_f = train_test_split(files, test_size=VAL_SPLIT)
    

    mask_files = set(os.listdir(MASK_DIR))
    train_f = [f for f in train_f if f in mask_files]
    val_f = [f for f in val_f if f in mask_files]
    print(f"Training files: {len(train_f)}, Validation files: {len(val_f)}")

    print("Creating datasets...")
    train_ds = SkullDataset(CT_DIR, MASK_DIR, train_f)
    val_ds = SkullDataset(CT_DIR, MASK_DIR, val_f)

    print("Creating data loaders...")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)

    print("Creating model...")
    model = AttentionUNet3D().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCELoss()
    print("Starting training...")

    for epoch in range(NUM_EPOCHS):
        model.train()
        t_loss, t_dice = 0, 0

        for i, (ct, mask) in enumerate(train_loader):
            try:
                if i % 10 == 0:
                    print(f"Processing batch {i+1}/{len(train_loader)}")
                    
                ct, mask = ct.to(device), mask.to(device).unsqueeze(1)
                
                opt.zero_grad()
                out = model(ct)
                loss = loss_fn(out, mask)
                loss.backward()
                opt.step()

                t_loss += loss.item()
                t_dice += dice(out, mask).item()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error processing batch {i+1}: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue


        model.eval()
        val_acc, val_loss, val_dice_score, val_map, val_iou, val_prec, val_rec = 0, 0, 0, 0, 0, 0, 0
        with torch.no_grad():
            for ct, mask in val_loader:
                ct, mask = ct.to(device), mask.to(device).unsqueeze(1)
                out = model(ct)
                
                loss = loss_fn(out, mask)
                val_loss += loss.item()
                val_dice_score += dice(out, mask).item()
                val_map += calculate_map(out, mask)
                val_iou += iou_score(out, mask).item()
                prec, rec = precision_recall(out, mask)
                val_prec += prec
                val_rec += rec
                val_acc += accuracy(out, mask).item()
        
        n = len(val_loader)
        avg_acc = val_acc / n
        avg_loss = val_loss / n
        avg_dice = val_dice_score / n
        avg_map = val_map / n
        avg_iou = val_iou / n
        avg_prec = val_prec / n
        avg_rec = val_rec / n
        
        # Save history
        history['epochs'].append(epoch + 1)
        history['train_loss'].append(t_loss / len(train_loader))
        history['train_dice'].append(t_dice / len(train_loader))
        history['val_loss'].append(avg_loss)
        history['val_dice'].append(avg_dice)
        history['val_map'].append(avg_map)
        history['val_iou'].append(avg_iou)
        history['val_precision'].append(avg_prec)
        history['val_recall'].append(avg_rec)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Loss: {t_loss/len(train_loader):.4f} | "
              f"Dice: {t_dice/len(train_loader):.4f} | "
              f"Val Loss: {avg_loss:.4f} | "
              f"Val Dice: {avg_dice:.4f} | "
              f"mAP: {avg_map:.4f} | "
              f"IoU: {avg_iou:.4f} | "
              f"Prec: {avg_prec:.4f} | "
              f"Rec: {avg_rec:.4f}")
        
        # Save only the best model
        if avg_acc > best_accuracy:
            best_accuracy = avg_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"[OK] New best model saved! Accuracy: {avg_acc*100:.1f}% (Epoch {best_epoch})")
        else:
            print(f"  Current best: {best_accuracy*100:.1f}% (Epoch {best_epoch})")
    
    # Save training history
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print("\nTraining history saved to training_history.json")

def cleanup_old_models():
    """Remove old epoch model files to save space"""
    import glob
    old_models = glob.glob('model_epoch_*.pth')
    for model_file in old_models:
        try:
            os.remove(model_file)
            print(f"Removed: {model_file}")
        except:
            pass
    if old_models:
        print(f"Cleaned up {len(old_models)} old model files")

if __name__ == "__main__":
    print("Starting training with best-model-only saving...")
    train()
    print("\nCleaning up old model files...")
    cleanup_old_models()
    print("Training completed! Only best_model.pth is saved.")
