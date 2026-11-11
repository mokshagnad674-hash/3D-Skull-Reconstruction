import matplotlib.pyplot as plt
import json
import os

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

def plot_training_curves(history_file='training_history.json', save_path='training_curves.png'):
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    epochs = history['epochs']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training')
    axes[0, 0].plot(epochs, history['val_loss'], 'r--', linewidth=2, label='Validation')
    axes[0, 0].set_title('(a) Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend(frameon=True, loc='best')
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')
    
    # Dice Score
    axes[0, 1].plot(epochs, history['train_dice'], 'b-', linewidth=2, label='Training')
    axes[0, 1].plot(epochs, history['val_dice'], 'r--', linewidth=2, label='Validation')
    axes[0, 1].set_title('(b) Dice Coefficient', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice Score')
    axes[0, 1].legend(frameon=True, loc='best')
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')
    
    # mAP
    axes[0, 2].plot(epochs, history['val_map'], 'g-', linewidth=2.5)
    axes[0, 2].set_title('(c) Mean Average Precision', fontweight='bold')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('mAP')
    axes[0, 2].grid(True, alpha=0.3, linestyle='--')
    
    # IoU
    axes[1, 0].plot(epochs, history['val_iou'], 'purple', linewidth=2.5)
    axes[1, 0].set_title('(d) Intersection over Union', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU Score')
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    
    # Precision
    axes[1, 1].plot(epochs, history['val_precision'], 'orange', linewidth=2.5)
    axes[1, 1].set_title('(e) Precision', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Precision')
    axes[1, 1].grid(True, alpha=0.3, linestyle='--')
    
    # Recall
    axes[1, 2].plot(epochs, history['val_recall'], 'teal', linewidth=2.5)
    axes[1, 2].set_title('(f) Recall', fontweight='bold')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Recall')
    axes[1, 2].grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"Publication-quality figure saved: {save_path}")
    plt.close()

def plot_combined_metrics(history_file='training_history.json', save_path='combined_metrics.png'):
    """Single combined plot for paper"""
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    epochs = history['epochs']
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(epochs, history['val_dice'], 'b-', linewidth=2.5, label='Dice Coefficient', marker='o', markersize=4, markevery=5)
    ax.plot(epochs, history['val_iou'], 'r-', linewidth=2.5, label='IoU', marker='s', markersize=4, markevery=5)
    ax.plot(epochs, history['val_map'], 'g-', linewidth=2.5, label='mAP', marker='^', markersize=4, markevery=5)
    ax.plot(epochs, history['val_precision'], 'orange', linewidth=2.5, label='Precision', marker='d', markersize=4, markevery=5)
    ax.plot(epochs, history['val_recall'], 'purple', linewidth=2.5, label='Recall', marker='*', markersize=6, markevery=5)
    
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('3D U-Net Performance Metrics', fontsize=14, fontweight='bold')
    ax.legend(frameon=True, loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"Combined metrics figure saved: {save_path}")
    plt.close()

def plot_loss_only(history_file='training_history.json', save_path='loss_curve.png'):
    """Loss curve for paper"""
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    epochs = history['epochs']
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    ax.plot(epochs, history['train_loss'], 'b-', linewidth=2.5, label='Training Loss', marker='o', markersize=4, markevery=5)
    ax.plot(epochs, history['val_loss'], 'r--', linewidth=2.5, label='Validation Loss', marker='s', markersize=4, markevery=5)
    
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=13, fontweight='bold')
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(frameon=True, loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"Loss curve saved: {save_path}")
    plt.close()

if __name__ == "__main__":
    if os.path.exists('training_history.json'):
        print("Generating publication-quality figures...")
        plot_training_curves()  # 6-panel figure
        plot_combined_metrics()  # Single combined plot
        plot_loss_only()  # Loss only
        print("\nGenerated files:")
        print("  - training_curves.png (6-panel, 600 DPI)")
        print("  - combined_metrics.png (single plot, 600 DPI)")
        print("  - loss_curve.png (loss only, 600 DPI)")
    else:
        print("No training history found. Run training first.")
