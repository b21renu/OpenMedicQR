# infer.py
import os
import cv2
import json
import re
import argparse
from tqdm import tqdm
from ultralytics import YOLO
from pyzbar.pyzbar import decode

def classify_qr(qr_data):
    """
    Classifies a decoded QR string based on regex patterns.
    
    You should update these patterns based on what you see in the actual data.
    These are just examples.
    """
    if re.match(r'^BATCH-\d+', qr_data):
        return "Batch"
    elif re.match(r'^MFG-[A-Z]{3}-\d+', qr_data):
        return "Manufacturer"
    elif re.match(r'^REG-\d{4}', qr_data):
        return "Regulator"
    else:
        return "Unknown"

def run_inference(model_path, input_folder, output_folder, conf_thres=0.25):
    """
    Runs the full inference pipeline: Detect -> Crop -> Decode -> Classify -> Save JSON.
    """
    # 1. Load the trained YOLOv8 model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    # 2. Prepare output lists for the JSON files
    stage1_results = []
    stage2_results = []

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Get list of all images in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_files)} images in {input_folder}. Starting inference...")

    # 3. Process each image
    for image_file in tqdm(image_files, desc="Processing Images"):
        image_path = os.path.join(input_folder, image_file)
        image_id = os.path.splitext(image_file)[0] # e.g., 'img201'
        
        # Read the original image for cropping later
        # OpenCV reads in BGR format
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            continue

        # --- A. Detection ---
        # Run the YOLO model on the image
        results = model(image_path, conf=conf_thres, verbose=False)
        
        # Prepare the per-image result dictionaries
        img_stage1 = {"image_id": image_id, "qrs": []}
        img_stage2 = {"image_id": image_id, "qrs": []}
        
        # Process each detected box
        for result in results:
            boxes = result.boxes.cpu().numpy() # Get boxes on CPU as numpy array
            for box in boxes:
                # Get the bounding box coordinates: [x_min, y_min, x_max, y_max]
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                
                # Ensure coordinates are within image bounds
                h, w, _ = original_image.shape
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(w, x_max), min(h, y_max)
                
                bbox = [x_min, y_min, x_max, y_max]
                
                # Add to Stage 1 results (Detection Only)
                img_stage1["qrs"].append({"bbox": bbox})
                
                # --- B. Crop, Decode, and Classify (Stage 2) ---
                # Crop the detected QR code from the original image
                # Note: numpy array slicing is [y_min:y_max, x_min:x_max]
                qr_crop = original_image[y_min:y_max, x_min:x_max]
                
                decoded_value = ""
                qr_type = ""
                
                # Decode using pyzbar if the crop is valid
                if qr_crop.size > 0:
                     # pyzbar expects a PIL image or a numpy array (which we have)
                    decoded_objects = decode(qr_crop)
                    if decoded_objects:
                        # Take the first decoded object
                        decoded_value = decoded_objects[0].data.decode('utf-8')
                        qr_type = classify_qr(decoded_value)
                
                # Add to Stage 2 results (Detection + Decoding + Classification)
                img_stage2["qrs"].append({
                    "bbox": bbox,
                    "value": decoded_value,
                })

        # Append the per-image results to the main lists
        stage1_results.append(img_stage1)
        stage2_results.append(img_stage2)

    # 4. Save the results to JSON files
    stage1_path = os.path.join(output_folder, 'submission_detection_1.json')
    stage2_path = os.path.join(output_folder, 'submission_decoding_2.json')
    
    print(f"\nSaving Stage 1 results to {stage1_path}...")
    with open(stage1_path, 'w') as f:
        json.dump(stage1_results, f, indent=2)
        
    print(f"Saving Stage 2 results to {stage2_path}...")
    with open(stage2_path, 'w') as f:
        json.dump(stage2_results, f, indent=2)
        
    print("\nInference complete! 🎉")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run QR detection and decoding inference.")
    parser.add_argument('--model', type=str, required=True, help='Path to the trained .pt model file.')
    parser.add_argument('--input', type=str, required=True, help='Folder containing input images.')
    parser.add_argument('--output', type=str, default='outputs/', help='Folder to save the output JSON files.')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for detection.')
    
    args = parser.parse_args()
    
    run_inference(args.model, args.input, args.output, args.conf)