from Dataset import VOCDataset
import torchvision.transforms as T

root = './data/VOCdevkit/VOC2007'

transforms = T.Compose([T.ToTensor()])

dataset = VOCDataset(root=root, split='train', transforms=transforms)

print(f'Number of valid training images: {len(dataset)}')

image, target = dataset[0]
print(f'Image shape: {image.shape}')
print(f'Boxes: {target["boxes"]}')
print(f'Labels: {target["labels"]}')