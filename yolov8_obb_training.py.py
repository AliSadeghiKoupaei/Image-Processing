import os
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

# Step 0: Setup and Change Directory to "datasets"
try:
    HOME = os.getcwd()  # Define the HOME directory (current working directory)

    # Create the "datasets" directory if it doesn't already exist
    datasets_path = os.path.join(HOME, "datasets")
    os.makedirs(datasets_path, exist_ok=True)

    # Change the current working directory to the "datasets" directory
    os.chdir(datasets_path)

    # Print the new working directory to confirm the change
    print(f"Step 0 - Current working directory set to: {os.getcwd()}")
except Exception as e:
    print(f"Error during Step 0 setup: {e}")

# Step 1: Download Dataset from Roboflow
try:
    # Initialize Roboflow
    rf = Roboflow(api_key="7gBPq3PodSFSwk2z8Nwu")
    
    # Access the project and version
    project = rf.workspace("uwm-traffic").project("image-processing-4")
    version = project.version(1)
    
    # Download the dataset for YOLO-OBB
    dataset = version.download("yolov8-obb")
    print(f"Dataset downloaded to: {dataset.location}")
    
    # Dynamically set dataset_location from Roboflow
    dataset_location = os.path.abspath(dataset.location)
    print(f"Dataset location set to: {dataset_location}")
except Exception as e:
    print(f"Error downloading dataset from Roboflow: {e}")
    dataset_location = None  # Prevent further errors

# Step 2: Update data.yaml for YOLO-OBB
if dataset_location:
    data_yaml_path = os.path.join(dataset_location, "data.yaml")

    try:
        # Check if the file exists
        if not os.path.exists(data_yaml_path):
            raise FileNotFoundError(f"data.yaml file not found at {data_yaml_path}")

        # Load the existing YAML file
        with open(data_yaml_path, "r") as file:
            data = yaml.safe_load(file)
        
        # Update dataset paths for YOLO-OBB
        data['train'] = os.path.join(dataset_location, 'train', 'images')
        data['val'] = os.path.join(dataset_location, 'valid', 'images')
        data['test'] = os.path.join(dataset_location, 'test', 'images')
        
        # Remove the 'path' key if it exists
        if 'path' in data:
            del data['path']
        
        # Save the updated YAML file
        with open(data_yaml_path, "w") as file:
            yaml.dump(data, file, sort_keys=False)
        
        print("Updated data.yaml successfully for YOLO-OBB.")
    except Exception as e:
        print(f"Error occurred while updating data.yaml: {e}")
else:
    print("Skipping Step 2: dataset_location not set correctly.")

# Step 3: Train YOLO-OBB Model
if dataset_location:
    try:
        # Load the YOLO-OBB model
        model = YOLO('yolov8m-obb.pt')  # Use YOLO-OBB model
        
        # Train the model
        results = model.train(
            data=data_yaml_path,  # Path to the updated YAML file
            epochs=150,           # Number of epochs
            imgsz=640,            # Image size
            plots=True            # Enable training plots
        )
        
        print("YOLO-OBB training completed successfully.")
    except Exception as e:
        print(f"Error occurred during YOLO-OBB training: {e}")
else:
    print("Skipping Step 3: dataset_location not set correctly.")