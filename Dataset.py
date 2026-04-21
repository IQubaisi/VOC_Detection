## SECTION 1:
import torch
from torch.utils.data import Dataset
import os
import xml.etree.ElementTree as ET
from PIL import Image
import albumentations as A
import numpy as np

CLASS_MAP = {
    'person': 1,
    'car': 2,
    'bus': 3,
    'bicycle': 4
}

## SECTION 2:
class VOCDataset(Dataset):
    def __init__(self, root, split='train', transforms=None, augment=False):
        self.root = root
        self.transforms = transforms
        self.augment = augment
        
        split_file = os.path.join(root, 'ImageSets', 'Main', f'{split}.txt')
        with open(split_file) as f:
            self.image_ids = [line.strip() for line in f.readlines()]
        
        self.image_ids = [id for id in self.image_ids if self.has_valid_objects(id)]

## NEW SECTION: Defining the augmentaion pipeline
    def get_augmentation(self):
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.0,
                p=0.5
            ),
            A.RandomResizedCrop(
                size=(400, 400),
                scale=(0.8, 1.0),
                p=0.5
            ),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels'],
            min_visibility=0.3
        ))

## SECTION 3:
    def __len__(self):
        return len(self.image_ids)

## SECTION 4:
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        img_path = os.path.join(self.root, 'JPEGImages', f'{image_id}.jpg')
        image = Image.open(img_path).convert('RGB')
        
        xml_path = os.path.join(self.root, 'Annotations', f'{image_id}.xml')
        boxes, labels = self.parse_xml(xml_path)
        
        ## Applying augmentaion
        if self.augment:
            aug = self.get_augmentation()
            image_np = np.array(image)
            augmented = aug(image=image_np, bboxes=boxes, labels=labels)
            image = Image.fromarray(augmented['image'])
            boxes = [list(b) for b in augmented['bboxes']]
            labels = list(augmented['labels'])

        if len(boxes) == 0:
            boxes = [[0, 0, 1, 1]]
            labels = [0]

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target

## SECTION 5:
    def has_valid_objects(self, image_id):
        xml_path = os.path.join(self.root, 'Annotations', f'{image_id}.xml')
        boxes, labels = self.parse_xml(xml_path)
        return len(labels) > 0

## SECTION 6:
    def parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            name = obj.find('name').text.lower()
            
            if name not in CLASS_MAP:
                continue
            
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CLASS_MAP[name])
        
        return boxes, labels