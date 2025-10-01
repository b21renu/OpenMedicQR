# train.py

import os
import shutil
import yaml
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
import argparse

def prepare_dataset(base_data_path, train_ratio=0.8):
    """
    Splits the dataset into training and validation sets and creates the necessary
    directory structure and YAML configuration file for YOLOv8.
    """
    images_path = os.path.join(base_data_path, 'train_images')
    labels_path = os.path.join(base_data_path, 'train_labels')
    
    print("Verifying paths...")
    if not os.path.exists(images_path):
        print(f"Error: Image directory not found at {images_path}")
        return None
    if not os.path.exists(labels_path):
        print(f"Error: Label directory not found at {labels_path}")
        return None
    
    # Get all image file names (without extension)
    all_filenames = [os.path.splitext(f)[0] for f in os.listdir(images_path) if f.endswith('.jpg')]
    
    # Split files into training and validation sets
    train_files, val_files = train_test_split(all_filenames, train_size=train_ratio, random_state=42)
    
    print(f"Total images found: {len(all_filenames)}")
    print(f"Splitting into {len(train_files)} training and {len(val_files)} validation images.")
    
    # Define paths for the new YOLO-compliant directory structure
    dataset_root = os.path.join(base_data_path, 'yolo_dataset')
    paths = {
        'train_images': os.path.join(dataset_root, 'images', 'train'),
        'val_images': os.path.join(dataset_root, 'images', 'val'),
        'train_labels': os.path.join(dataset_root, 'labels', 'train'),
        'val_labels': os.path.join(dataset_root, 'labels', 'val'),
    }
    
    # Create directories, wiping any old ones
    if os.path.exists(dataset_root):
        shutil.rmtree(dataset_root)
    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    # Function to copy files
    def copy_files(filenames, set_type):
        for filename in filenames:
            # Copy image
            shutil.copy(os.path.join(images_path, filename + '.jpg'), paths[f'{set_type}_images'])
            # Copy label
            # shutil.copy(os.path.join(labels_path, filename + '.txt'), paths[f'{set_type}_labels'])
            label_file = os.path.join(labels_path, filename + '.txt')
            if os.path.exists(label_file):
                shutil.copy(label_file, paths[f'{set_type}_labels'])
            else:
                print(f"⚠️ Warning: No label found for {filename}. Skipping label copy.")
                
    # Copy files to their new locations
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')

    # --- Create the data.yaml file ---
    yaml_config = {
        'train': os.path.abspath(paths['train_images']),
        'val': os.path.abspath(paths['val_images']),
        'nc': 1,  # Number of classes
        'names': ['QR-Code']  # List of class names
    }
    
    yaml_path = os.path.join(dataset_root, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False)
        
    print(f"Dataset prepared successfully. 'data.yaml' created at {yaml_path}")
    return yaml_path

def main(epochs, batch_size):
    """
    Main function to prepare data and run YOLOv8 training.
    """
    # --- 1. Prepare the dataset ---
    # This will split the data and create the 'data.yaml' file
    base_data_path = 'data/'
    data_yaml_path = prepare_dataset(base_data_path, train_ratio=0.8)

    if data_yaml_path is None:
        print("Halting training due to data preparation error.")
        return

    # --- 2. Configure and Run Training ---
    # Load a pretrained YOLOv8 model (yolov8n.pt is the smallest)
    model = YOLO('yolov8n.pt')

    print("\nStarting YOLOv8 training...")
    # Train the model. Results are saved in the 'runs/' directory
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        name='yolov8_qr_detection_final'
    )
    
    print("\nTraining complete.")
    print(f"Model and results saved in: {results.save_dir}")
    print(f"The best model weights are located at: {os.path.join(results.save_dir, 'weights/best.pt')}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train YOLOv8 model for QR code detection.")
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch', type=int, default=8, help='Batch size for training.')
    args = parser.parse_args()
    
    main(args.epochs, args.batch)