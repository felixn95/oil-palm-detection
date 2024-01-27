import streamlit as st
import base64
import PIL
import pathlib
from pathlib import Path
import timm
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
    # img = img.resize((256, 256))
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
    
    background_image_path = Path('background') / 'bk1pic.png'
    background_image_path_str = str(background_image_path)
    set_background_for_title_area(background_image_path_str)

    # Create a container for the title
    title_container = st.container()
    with title_container:
        st.markdown('<div class="title-container"><h1>Oil Palm Detection</h1></div>', unsafe_allow_html=True)

    # st.title("Oil Palm Detection")

    # Input for paths to training data
    train_images = 'train_images'
    train_labels = 'train_labels_filtered.csv'

    # Model selection
    model_name = st.selectbox("Select a Model", ["densenet169", "resnet34"])

    # Choose upload type
    upload_type = st.radio("Choose the upload type", ["Single Image", "ZIP File"])

    if upload_type == "Single Image":
        # Existing single image upload logic
        uploaded_image_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"], key="image_uploader")
        if uploaded_image_file is not None and train_images and train_labels:
            
            # Display the uploaded image
            bytes_data = uploaded_image_file.getvalue()
            image = Image.open(io.BytesIO(bytes_data))
            st.image(image, caption=uploaded_image_file.name, use_column_width=True)

            # Save the image temporarily to pass its path to the prediction function
            temp_image_path = temp_extract_path / uploaded_image_file.name
            with open(temp_image_path, "wb") as f:
                f.write(bytes_data)

            # Initialize Model and make prediction
            with st.spinner('Loading model and making prediction...'):
                # helper = Model_Helper(train_images, train_labels)
                learn = load_model(model_name)
                # Predict the image
                prediction = predict_image(learn, temp_image_path)
                pred_class, _, confidence_scores = prediction
                confidence_score = confidence_scores[1].item()             
                print ("confidence (true):", confidence_scores)
                # Display the result based on the prediction
                if pred_class == '1':
                    st.success(f"Oil Palm Plantation detected. Confidence Score: {confidence_score}")
                else:
                    st.error(f"No Oil Palm Plantation detected. Confidence Score: {confidence_score}")

                # Display additional model information
                st.write("Model Details:")
                st.text(f"Model Architecture:\n{learn.model}")
                st.text(f"Data Loaders:\n{learn.dls}")
                st.text(f"Item Transforms:\n{learn.dls.after_item}")
                st.text(f"Batch Transforms:\n{learn.dls.after_batch}")

                if learn.recorder:
                    st.text(f"Final Validation Loss: {learn.recorder.final_record[0]}")
                    st.text(f"Metrics: {learn.recorder.final_record[1:]}")  

    elif upload_type == "ZIP File":
        print("reached here")
        uploaded_file = st.file_uploader("Upload a ZIP file containing images", type=["zip"])
        if uploaded_file is not None:
            print("reached here 0")
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)

                temp_path = pathlib.Path(temp_dir)

                # Recursively find all image files
                image_files = list(temp_path.rglob('*.png')) + list(temp_path.rglob('*.jpg')) + list(temp_path.rglob('*.jpeg'))
                image_files = [str(file) for file in image_files]  # Convert Path objects to strings

                st.write(f"ZIP-File contains {len(image_files)} images to predict")  # Debugging output

                learn = load_model(model_name)

                progress_bar = st.progress(0)
                total_images = len(image_files)

                saved_images_dir = pathlib.Path('found_images')
                saved_images_dir.mkdir(exist_ok=True)  # Create the directory if it doesn't exist

                # Header for found images
                st.subheader("Detected Oil Palms in your Set of Images:")

                # Initialize a list to store the data
                image_data = []

                # Process each image
                for idx, image_path in enumerate(image_files):
                    # Predict the image
                    prediction = predict_image(learn, image_path)
                    pred_class, _, confidence_scores = prediction
                    confidence_score = confidence_scores[1].item()
                    image_name = pathlib.Path(image_path).name
                    
                    if pred_class == '1':
                        saved_image_path = saved_images_dir / image_name
                        shutil.copy(image_path, saved_image_path)

                        # Append the data to the list
                        image_data.append([image_name, confidence_score, saved_image_path])

                    # Update the progress bar
                    progress_bar.progress((idx + 1) / total_images)

                # Create a DataFrame from the list
                df = pd.DataFrame(image_data, columns=["Image Name", "Confidence Score", "Image Path"])

                # Display the table
                st.table(df)
                #df_html = render_dataframe_with_colored_cells(df)
                # st.markdown(df_html, unsafe_allow_html=True)

                # Optionally, show images below the table
                for _, row in df.iterrows():
                    image = Image.open(row["Image Path"])
                    st.image(image, caption=row["Image Name"], use_column_width=True)

            # Cleanup
            progress_bar.empty()


def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as file:
        data = file.read()
    return base64.b64encode(data).decode()

def set_background_for_title_area(image_path):
    bin_str = get_base64_of_bin_file(image_path)
    background_css = f"""
    <style>
    .title-container {{
        background-image: url("data:image/jpg;base64,{bin_str}");
        background-size: cover;
        background-repeat: no-repeat;
        text-align: center;
        color: white;
        padding: 50px;  /* Adjust padding as needed */
    }}
    .title-container h1 {{
        color: white;
    }}
    </style>
    """
    st.markdown(background_css, unsafe_allow_html=True)

# Function to return colored cell HTML based on the score
def color_confidence_cell(score):
    if score < 0.5:
        color = "red"
    elif 0.5 <= score < 0.6:
        color = "yellow"
    else:
        color = "green"
    return f'<td style="background-color: {color};">{score:.2f}</td>'

def render_dataframe_with_colored_cells(df):
    # Convert entire DataFrame to HTML
    df_html = df.to_html(escape=False, index=False)

    # Find and replace confidence score cells with colored cells
    for score in df['Confidence Score']:
        df_html = df_html.replace(f'<td>{score}</td>', color_confidence_cell(score), 1)

    return df_html

if __name__ == "__main__":
    main()
 
