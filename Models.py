import torchvision
from torchvision.models import ResNet50_Weights, ResNet101_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def _get_backbone_weights(backbone_name):
    if backbone_name == "resnet50":
        return ResNet50_Weights.DEFAULT
    if backbone_name == "resnet101":
        return ResNet101_Weights.DEFAULT
    raise ValueError(f"Unsupported backbone_name: {backbone_name}")

def get_model(num_classes, backbone_name='resnet101', box_nms_thresh=0.5):
    # Build a ResNet101 backbone with FPN
    backbone = resnet_fpn_backbone(
        backbone_name=backbone_name,
        weights=_get_backbone_weights(backbone_name)
    )
    
    # Load Faster R-CNN with our custom backbone
    # box_nms_thresh controls how aggressively duplicate boxes are suppressed
    # default is 0.5 — lower values suppress more aggressively
    model = torchvision.models.detection.FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        box_nms_thresh=box_nms_thresh
    )

    return model
        
if __name__ == '__main__':
    model = get_model(num_classes=5)
    print(model)
    print('Model loaded successfully')
