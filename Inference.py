# SECTION 1: Imports
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from Dataset import VOCDataset
from Models import get_model
import os

# SECTION 2: Settings — change these to control what the script does
CHECKPOINT = 'checkpoints/model_epoch_7.pth'
ROOT = './data/VOCdevkit/VOC2007'
NUM_CLASSES = 5
CONFIDENCE_THRESHOLD = 0.7  # only show predictions the model is at least 50% confident about
IMAGE_PATH = None            # None = grab from VOC dataset, or set to 'your_image.jpg'
IMAGE_INDEX = 5              # which VOC image to use if IMAGE_PATH is None

# SECTION 3: Class labels and colours — one colour per class for the boxes
CLASS_NAMES = {1: 'person', 2: 'car', 3: 'bus', 4: 'bicycle'}
CLASS_COLOURS = {1: 'red', 2: 'blue', 3: 'green', 4: 'orange'}

# SECTION 4: Load model
def load_model(checkpoint_path, num_classes, device):
    model = get_model(num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()  # eval mode — returns predictions not losses
    return model

# SECTION 5: Load image
# This is where the two approaches split
def load_image(image_path, root, image_index):
    if image_path is not None:
        # Approach B — your own image
        # Pillow opens it, we convert to RGB to make sure it has 3 channels
        print(f'Loading your image: {image_path}')
        image = Image.open(image_path).convert('RGB')
        source = 'custom'
    else:
        # Approach A — from VOC dataset
        # Dataset stores image IDs like "000032"
        # We build the full path the same way __getitem__ does
        dataset = VOCDataset(root=root, split='val', transforms=None)
        image_id = dataset.image_ids[image_index]
        image_path_voc = os.path.join(root, 'JPEGImages', f'{image_id}.jpg')
        image = Image.open(image_path_voc).convert('RGB')
        source = 'voc'
        print(f'Image path: {image_path_voc}')

    return image, source

# SECTION 6: Run model on image
def run_inference(model, image, device):
    # Convert PIL image to tensor — same transform used in training
    transform = T.ToTensor()
    tensor = transform(image).to(device)

    # Model expects a list of images, so we wrap it in a list
    # unsqueeze would also work but this matches the format from our DataLoader
    with torch.no_grad():
        outputs = model([tensor])

    # outputs is a list with one dict — we take the first element
    return outputs[0]

# SECTION 7: Draw boxes on image and show it
def draw_predictions(image, outputs, confidence_threshold):
    # Convert PIL image to numpy array for matplotlib
    image_np = np.array(image)

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image_np)

    boxes = outputs['boxes'].cpu()
    labels = outputs['labels'].cpu()
    scores = outputs['scores'].cpu()

    # Loop through every prediction
    for box, label, score in zip(boxes, labels, scores):
        # Skip predictions below confidence threshold
        if score < confidence_threshold:
            continue

        label_id = label.item()
        class_name = CLASS_NAMES.get(label_id, 'unknown')
        colour = CLASS_COLOURS.get(label_id, 'white')

        # Box coordinates — VOC format is [x_min, y_min, x_max, y_max]
        x_min, y_min, x_max, y_max = box.tolist()
        width = x_max - x_min
        height = y_max - y_min

        # Draw the rectangle
        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=2,
            edgecolor=colour,
            facecolor='none'  # transparent fill
        )
        ax.add_patch(rect)

        # Draw the label and score above the box
        ax.text(
            x_min, y_min - 5,
            f'{class_name} {score:.2f}',
            color=colour,
            fontsize=10,
            fontweight='bold',
            backgroundcolor='black'
        )

    plt.axis('off')
    plt.tight_layout()
    plt.savefig('inference_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Result saved to inference_result.png')

# SECTION 8: Main
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = load_model(CHECKPOINT, NUM_CLASSES, device)
    print(f'Model loaded from {CHECKPOINT}')

    image, source = load_image(IMAGE_PATH, ROOT, IMAGE_INDEX)
    print(f'Image loaded — source: {source}, size: {image.size}')

    outputs = run_inference(model, image, device)

    # Print what the model found before drawing
    print(f'\nRaw predictions: {len(outputs["boxes"])} boxes found')
    print(f'After confidence filter (>{CONFIDENCE_THRESHOLD}): ', end='')
    kept = (outputs['scores'] > CONFIDENCE_THRESHOLD).sum().item()
    print(f'{kept} boxes kept')

    draw_predictions(image, outputs, CONFIDENCE_THRESHOLD)

if __name__ == '__main__':
    main()