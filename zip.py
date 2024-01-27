import streamlit as st
import base64
import timm
import zipfile
import pathlib
from pathlib import Path
import shutil
import platform
from fastai.vision.all import *


# Define the base directory relative to the WORKDIR in Docker
base_dir = Path("/usr/src/app")  # Changed to absolute path based on WORKDIR

# Create necessary directories
temp_zip_path = base_dir / "temp.zip"
temp_extract_path = base_dir / "extracted"
temp_extract_path.mkdir(parents=True, exist_ok=True)


def get_x(row, train_images):
    return os.path.join(train_images, row['image_id'])

def get_y(row): 
    return row['has_oilpalm']

def create_dataloader(batch_size=64):
    train_images = 'train_images'
    train_labels = 'train_labels_filtered.csv'
    palm_data_block = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x=partial(get_x, train_images=train_images),
        get_y=get_y,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        item_tfms=Resize(256),
        batch_tfms=aug_transforms()
    )
    return palm_data_block.dataloaders(pd.read_csv(train_labels), bs=batch_size)

def load_model(model_name):
    learn = vision_learner(create_dataloader(), model_name, metrics=accuracy)
    path_to_model = f'{model_name}.pkl'
    return load_learner(path_to_model)

# Function to perform inference on a single image
def predict_image(learn, image_path):
    # Load the image with FastAI's method
    print("image_path:", image_path)
    img = Image.open(image_path)
    print ("imgPIL:", image_path)
    img = img.resize((256, 256))
    # Perform prediction
    prediction = learn.predict(image_path)
    print("Prediction:", prediction)
    pred_class, _, confidence_scores = prediction

    return prediction

def set_permissions(path, permission):
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            os.chmod(os.path.join(root, dir), permission)
        for file in files:
            os.chmod(os.path.join(root, file), permission)

# Streamlit UI
def main():

    # Model selection
    model_name = st.selectbox("Select a Model", ["densenet169", "resnet34"])

    # File uploader
    uploaded_file = st.file_uploader("Choose a ZIP file", type="zip")
    if uploaded_file is not None:
        
        learn = load_model(model_name)
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()

        # Save the zip file
        with open(temp_zip_path, "wb") as f:
            f.write(bytes_data)

        # Extracting the zip file
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_path)

        set_permissions(temp_extract_path, 0o755)

        # Process the images
        for root, dirs, files in os.walk(temp_extract_path):
            for filename in files:
                if filename.endswith((".jpg", ".png")):  # Add other image formats if needed
                    image_path = os.path.join(root, filename)
                    
                    # Open and display the image (example)
                    image = Image.open(image_path)
                    st.image(image, caption=filename)

                    # Predict the image
                    prediction = predict_image(learn, image_path)
                    pred_class, _, confidence_scores = prediction
                    confidence_score = confidence_scores[1].item()

                    st.write(f"Prediction: {pred_class}, Confidence Score: {confidence_score}")

        # Clean up
        shutil.rmtree(temp_extract_path)

if __name__ == "__main__":
    main()
