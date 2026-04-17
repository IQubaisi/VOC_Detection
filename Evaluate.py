# SECTION 1: Imports
import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torchvision.transforms as T
from Dataset import VOCDataset
from Models import get_model
import os
import wandb

# SECTION 2: Load the model from a checkpoint
def load_model(checkpoint_path, num_classes, device):
    model = get_model(num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()  # <-- eval mode here, this is where it belongs
    return model

# SECTION 3: Run model on validation set and collect predictions
def get_predictions(model, val_loader, device):
    all_predictions = []
    all_targets = []

    with torch.no_grad():  # no gradients needed, we're just predicting
        for images, targets in val_loader:
            images = [img.to(device) for img in images]

            # model in eval mode returns predictions, not losses
            outputs = model(images)

            for output in outputs:
                all_predictions.append({
                    'boxes': output['boxes'].cpu(),
                    'scores': output['scores'].cpu(),
                    'labels': output['labels'].cpu()
                })

            for target in targets:
                all_targets.append({
                    'boxes': target['boxes'].cpu(),
                    'labels': target['labels'].cpu()
                })

    return all_predictions, all_targets

# SECTION 4: Calculate mAP
def evaluate(checkpoint_path, root, num_classes=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load model
    model = load_model(checkpoint_path, num_classes, device)
    print(f'Loaded model from {checkpoint_path}')

    # Load validation data
    transforms = T.Compose([T.ToTensor()])
    val_dataset = VOCDataset(root=root, split='val', transforms=transforms)
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )
    print(f'Validation set: {len(val_dataset)} images')

    # Get predictions
    print('Running model on validation set...')
    predictions, targets = get_predictions(model, val_loader, device)

    # Calculate mAP using torchmetrics
    metric = MeanAveragePrecision(     # <-- Tells torchmetrics to track and return AP separately for each class label it encounters in the predictions.
    iou_type='bbox',
    class_metrics=True
    )

    metric.update(predictions, targets)
    result = metric.compute()
    print(result) #temporary debug

    # SECTION 5: Print results
    print('\n--- Evaluation Results ---')
    print(f"mAP @ IoU=0.50:0.95 : {result['map'].item():.4f}")
    print(f"mAP @ IoU=0.50      : {result['map_50'].item():.4f}")
    print(f"mAP @ IoU=0.75      : {result['map_75'].item():.4f}")

    # Per class AP
    class_names = {1: 'person', 2: 'car', 3: 'bus', 4: 'bicycle'}
    print('\n--- Per Class AP (IoU=0.50:0.95) ---')
    per_class = result['map_per_class']
    classes = result['classes']
    for cls_id, ap in zip(classes, per_class):
        name = class_names.get(cls_id.item(), f'class_{cls_id.item()}')
        print(f"  {name}: {ap.item():.4f}")

# Log results to WandB
    wandb.init(
        project="voc-detection",
        name="eval-epoch7-lr0.001",  # descriptive name so you know which run this is
        config={
            'checkpoint': checkpoint_path,
            'num_classes': num_classes
        }
    )

    wandb.log({
        'mAP': result['map'].item(),
        'mAP_50': result['map_50'].item(),
        'mAP_75': result['map_75'].item(),
        'AP_person': result['map_per_class'][0].item(),
        'AP_car': result['map_per_class'][1].item(),
        'AP_bus': result['map_per_class'][2].item(),
        'AP_bicycle': result['map_per_class'][3].item(),
    })

    wandb.finish()

if __name__ == '__main__':
    evaluate(
        checkpoint_path='checkpoints/model_epoch_7.pth',
        root='./data/VOCdevkit/VOC2007',
        num_classes=5
    )
