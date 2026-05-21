import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

class YOLOXAIWrapper(torch.nn.Module):
    """A wrapper class to extract feature maps from YOLO model and handle tuple outputs."""
    def __init__(self, model):
        super(YOLOXAIWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, tuple):
            return out[0]
        return out

def generate_targeted_xai_heatmap(image_path, model_path, output_path):
    print("="*60)
    print("Initializing TARGETED XAI Feature Extraction")
    print(f"Image: {image_path}")
    print(f"Model: {model_path}")
    print("="*60)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"CRITICAL ERROR: Could not find image at '{image_path}'")

    try:
        model = YOLO(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_path}'. Error: {e}")
    
    model_device = next(model.model.parameters()).device
    print(f"Model loaded successfully on device: {model_device}")

    # 1. Prepare the Image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (640, 640)) # Force 640x640 
    rgb_img = img[:, :, ::-1]         
    rgb_img_float = np.float32(rgb_img) / 255 
    
    tensor_img = torch.from_numpy(rgb_img_float).permute(2, 0, 1).unsqueeze(0).to(model_device)
    
    # 2. RUN INFERENCE FIRST (Using the resized 'img' array, NOT the path)
    print("Running YOLO Inference to locate objects...")
    results = model(img, verbose=False)[0] # <--- FIX: Passing the 640x640 image directly
    boxes = results.boxes.xyxy.cpu().numpy() 
    annotated_frame = results.plot() 
    
    # 3. Target the Detection Head Layers
    try:
        target_layers = [
            model.model.model[15], 
            model.model.model[18], 
            model.model.model[21]
        ] 
        print("Targeting Detection Heads [15, 18, 21] for extraction.")
    except IndexError:
        raise IndexError("Layer indices are out of bounds for this model architecture.")
    
    # 4. Initialize EigenCAM
    wrapped_model = YOLOXAIWrapper(model.model)
    cam = EigenCAM(model=wrapped_model, target_layers=target_layers)
    
    # 5. Generate the raw global heatmap
    print("Generating Raw XAI Heatmap...")
    raw_grayscale_cam = cam(input_tensor=tensor_img)[0, :]
    
    # --- 6. TARGETED MASKING LOGIC ---
    print("Applying Detection Mask to isolate predicted objects...")
    mask = np.zeros_like(raw_grayscale_cam)
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(mask.shape[1], x2), min(mask.shape[0], y2)
        
        mask[y1:y2, x1:x2] = 1.0

    targeted_cam = raw_grayscale_cam * mask
    
    if targeted_cam.max() > 0:
        targeted_cam = targeted_cam / targeted_cam.max()
    
    # 7. Overlay the targeted heatmap
    cam_image = show_cam_on_image(rgb_img_float, targeted_cam, use_rgb=True)
    
    # 8. Plot the results side-by-side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    ax1.imshow(rgb_img)
    ax1.set_title("Original Panoramic X-Ray")
    ax1.axis('off')
    
    ax2.imshow(cam_image)
    ax2.set_title("Targeted EigenCAM (Detected Objects Only)")
    ax2.axis('off')
    
    ax3.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
    ax3.set_title("Disease Detections")
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Targeted XAI analysis saved successfully to {output_path}")
    #plt.show()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Targeted Explainable AI (XAI) for YOLO Models")
    parser.add_argument("--image", type=str, required=True, help="Path to the X-ray image to analyze.")
    parser.add_argument("--model", type=str, default="Result/yolov8n/weights/best.pt", help="Path to your trained YOLO weights.")
    parser.add_argument("--output", type=str, default="xai_result.jpg", help="Filename to save the visualization.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    generate_targeted_xai_heatmap(args.image, args.model, args.output)