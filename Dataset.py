## SECTION 1:
import torch
from torch.utils.data import Dataset
import os
import xml.etree.ElementTree as ET
from PIL import Image

CLASS_MAP = {
    'person': 1,
    'car': 2,
    'bus': 3,
    'bicycle': 4
}

## SECTION 2:
class VOCDataset(Dataset):
    def __init__(self, root, split='train', transforms=None):
        self.root = root
        self.transforms = transforms
        
        split_file = os.path.join(root, 'ImageSets', 'Main', f'{split}.txt')
        with open(split_file) as f:
            self.image_ids = [line.strip() for line in f.readlines()]
        
        self.image_ids = [id for id in self.image_ids if self.has_valid_objects(id)]

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