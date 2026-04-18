import torchvision.datasets as datasets

print('Downloading VOC 2007 train...')
datasets.VOCDetection(root='./data', year='2007', image_set='train', download=True)

print('Downloading VOC 2007 val...')
datasets.VOCDetection(root='./data', year='2007', image_set='val', download=True)

print('Done!')