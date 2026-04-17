# SECTION 1: Imports
import torch
from torch.utils.data import DataLoader
import wandb
import os
import argparse
from Dataset import VOCDataset
from Models import get_model
import torchvision.transforms as T

# SECTION 2: WandB Setup
def setup_wandb(config):
    wandb.init(
        project="voc-detection",
        config = config
    )

# SECTION 3: Datasets and Dataloaders
def get_dataloaders(root, batch_size):
    transforms = T.Compose([T.ToTensor()])
    
    train_dataset = VOCDataset(root=root, split='train', transforms=transforms)
    val_dataset = VOCDataset(root=root, split='val', transforms=transforms)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    return train_loader, val_loader

# SECTION 4: Create the model
def get_training_model(num_classes, device, box_nms_thresh=0.5):
    model = get_model(num_classes=num_classes, box_nms_thresh=box_nms_thresh)
    model.to(device)
    return model

# SECTION 5: Optimiser
def get_optimizer(model, learning_rate, momentum, weight_decay):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay
    )
    return optimizer

# SECTION 6: The Training Loop
def train(model, train_loader, val_loader, optimizer, device, epochs, run_name='exp'):
    def _sum_losses(output):
        if isinstance(output, dict):
            return sum(loss for loss in output.values())
        raise TypeError(f"Expected loss dict from model, got {type(output).__name__}")

    for epoch in range(epochs):
        
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0
        
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = _sum_losses(loss_dict)
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            train_loss += losses.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # --- VALIDATION PHASE ---
        # CHANGE 1: Removed model.eval() — it was dead code, model.train() below overwrote it
        val_loss = 0
        
        model.train()
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                losses = _sum_losses(loss_dict)
                val_loss += losses.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # --- WANDB LOGGING ---
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        })
        
        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
        
        # CHANGE 3: Save checkpoint after every epoch, and run name
        # This means after epoch 1 you get model_epoch_1.pth, epoch 2 gives model_epoch_2.pth etc.
        # Evaluate.py and Inference.py will load from these files
        save_checkpoint(model, epoch + 1, run_name=run_name)

# SECTION 7: Save Checkpoint
def save_checkpoint(model, epoch, run_name='exp', path='checkpoints'):
    os.makedirs(path, exist_ok=True)
    checkpoint_path = os.path.join(path, f'{run_name}_epoch_{epoch}.pth')
    torch.save(model.state_dict(), checkpoint_path)
    print(f'Checkpoint saved: {checkpoint_path}')

def main():
    parser = argparse.ArgumentParser(description='Train Faster R-CNN on Pascal VOC 2007')
    
    parser.add_argument('--epochs',        type=int,   default=10,     help='Number of training epochs')
    parser.add_argument('--batch_size',    type=int,   default=2,      help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,  help='Learning rate')
    parser.add_argument('--momentum',      type=float, default=0.9,    help='SGD momentum')
    parser.add_argument('--weight_decay',  type=float, default=0.0005, help='Weight decay')
    parser.add_argument('--num_classes',   type=int,   default=5,      help='Number of classes including background')
    parser.add_argument('--box_nms_thresh', type=float, default=0.5,   help='NMS IoU threshold for box suppression')
    parser.add_argument('--run_name',      type=str, default='exp',    help='Name for this experiment run')
    
    args = parser.parse_args()
    
    config = {

        'epochs':        args.epochs,
        'batch_size':    args.batch_size,
        'learning_rate': args.learning_rate,
        'momentum':      args.momentum,
        'weight_decay':  args.weight_decay,
        'num_classes':   args.num_classes,
        'box_nms_thresh': args.box_nms_thresh,
        'run_name': args.run_name

    }
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    setup_wandb(config)
    
    root = './data/VOCdevkit/VOC2007'
    train_loader, val_loader = get_dataloaders(root, config['batch_size'])
    
    model = get_training_model(config['num_classes'], device)
    optimizer = get_optimizer(
        model,
        config['learning_rate'],
        config['momentum'],
        config['weight_decay']
    )
    
    # Train
    train(model, train_loader, val_loader, optimizer, device, config['epochs'])
    
    wandb.finish()

if __name__ == '__main__':
    main()