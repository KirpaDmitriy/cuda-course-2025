import torch
import torchvision
import os
import tensorrt as trt

ONNX_FILE = "../models/retinanet.onnx"
ENGINE_FILE = "../models/retinanet.engine"
CLASSES_PATH = "../models/classes.txt"
INPUT_SHAPE = (1, 3, 640, 640)

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
    'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 
    'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
    'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
os.makedirs(os.path.dirname(CLASSES_PATH), exist_ok=True)
with open(CLASSES_PATH, "w") as f:
    for c in COCO_INSTANCE_CATEGORY_NAMES:
        if c != 'N/A': f.write(c + "\n")

print("Loading RetinaNet...")
weights = torchvision.models.detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT
model = torchvision.models.detection.retinanet_resnet50_fpn(weights=weights)
model.eval()

dummy_input = torch.randn(*INPUT_SHAPE)

print(f"Exporting to {ONNX_FILE} with STATIC shapes and OPSET 11...")
with torch.no_grad():
    torch.onnx.export(
        model, 
        dummy_input, 
        ONNX_FILE, 
        export_params=True,
        opset_version=11, 
        do_constant_folding=True,
        input_names=['images'],
        output_names=['boxes', 'scores', 'labels'],
        dynamo=False,
    )

print("Building TensorRT Engine...")
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine():
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    except AttributeError:
        config.max_workspace_size = 1 << 30

    if builder.platform_has_fast_fp8:
        print("Enabling FP8...")
        config.set_flag(trt.BuilderFlag.FP8)

    with open(ONNX_FILE, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse ONNX.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False

    input_tensor = network.get_input(0)
    print(f"Input nodes: {input_tensor.name} {input_tensor.shape}")
    
    profile = builder.create_optimization_profile()
    profile.set_shape(input_tensor.name, INPUT_SHAPE, INPUT_SHAPE, INPUT_SHAPE)
    config.add_optimization_profile(profile)

    print("Building serialized network...")
    try:
        plan = builder.build_serialized_network(network, config)
    except AttributeError:
        engine = builder.build_engine(network, config)
        plan = engine.serialize()

    if plan is None:
        print("Error: Engine build failed!")
        return False

    with open(ENGINE_FILE, "wb") as f:
        f.write(plan)
    
    print(f"Success! Engine saved to {ENGINE_FILE}")
    print(f"Size: {os.path.getsize(ENGINE_FILE)/1024/1024:.2f} MB")
    return True

if __name__ == "__main__":
    build_engine()
