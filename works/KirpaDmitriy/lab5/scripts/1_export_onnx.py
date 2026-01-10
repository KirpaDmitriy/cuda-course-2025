import torch
import torchvision
import os

MODEL_PATH = "../models/retinanet.onnx"
CLASSES_PATH = "../models/classes.txt"

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

os.makedirs(os.path.dirname(CLASSES_PATH), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

print(f"Generating {CLASSES_PATH}...")
with open(CLASSES_PATH, "w") as f:
    for c in COCO_INSTANCE_CATEGORY_NAMES:
        if c != 'N/A':
             f.write(c + "\n")

print("Loading RetinaNet...")
model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
model.eval()

dummy_input = torch.randn(1, 3, 640, 640)

print(f"Exporting to {MODEL_PATH}...")

with torch.no_grad():
    torch.onnx.export(
        model, 
        dummy_input, 
        MODEL_PATH, 
        export_params=True,
        opset_version=11, 
        do_constant_folding=True,
        input_names=['images'],
        output_names=['boxes', 'scores', 'labels'],
        dynamic_axes={
            'images': {0: 'batch_size'},
            'boxes': {0: 'batch_size', 1: 'num_detections'},
            'scores': {0: 'batch_size', 1: 'num_detections'},
            'labels': {0: 'batch_size', 1: 'num_detections'}
        },
        dynamo=False,  # https://github.com/comfyanonymous/ComfyUI_TensorRT/issues/120#issuecomment-3503350210
    )

print("Export complete.")
