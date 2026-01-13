"""Code for the Game Component of our Streamlit App should go here"""
import os
import random
import streamlit as st
import torch
import torchvision.transforms as T
from torchvision.models import vgg16, resnet50
from model_utils import load_model, MODEL_ARCHS, get_inference_transform, generate_softmax_outputs, RESNET50_MODEL, VGG16_MODEL, CLASS_NAMES, weighted_average_ensemble, plain_average_ensemble
from PIL import Image

#TEST_ROOT = "/mri_dataset/test"
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
TEST_ROOT = os.path.join(PROJECT_ROOT, "mri_dataset", "test")



import streamlit as st


def get_random_test_image(test_root: str = TEST_ROOT):
    """
    Pick a random class folder under test_root, then a random image inside it.
    Returns (image_path, class_name).
    """
    # Get all subfolders (each one is a class)
    class_folders = [
        d
        for d in os.listdir(test_root)
        if os.path.isdir(os.path.join(test_root, d))
    ]

    if not class_folders:
        raise RuntimeError(f"No class folders found inside {test_root}")

    # Choose a random class folder
    class_name = random.choice(class_folders)
    class_folder_path = os.path.join(test_root, class_name)

    # Get all image files in that class folder
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    image_files = [
        f
        for f in os.listdir(class_folder_path)
        if f.lower().endswith(valid_exts)
    ]

    if not image_files:
        raise RuntimeError(f"No image files found in {class_folder_path}")

    # Pick a random image
    image_file = random.choice(image_files)
    image_path = os.path.join(class_folder_path, image_file)

    return image_path, class_name

def show_game_tab():
    st.header("MRI Classification Game")
    #model selection
    model_options = ["single-model", "multi-model"]
    selected_model = st.radio("Select Model Type: ", model_options)
    if selected_model == "single-model": 
        st.write("Single Model Selected")
        single_model_options = ["resnet50", "vgg16"]
        arch = st.radio("Select Single Model Architecture: ", single_model_options)
        if arch == "resnet50":
            st.write("resnet50 Selected")
        else: 
            st.write("vgg16 Selected")
    
    else:
        st.write("Multi Model Selected")
        multi_model_options = ["Plain Average Ensemble", "Weighted Average Ensemble"]
        arch = st.radio("Select Multi Model Architecture: ", multi_model_options)
        if arch == "Plain Average Ensemble":
            st.write("Plain Average Ensemble Selected")
        else: 
            st.write("Weighted Average Ensemble Selected")
    
    st.write("--------")
    #human guess
    st.subheader("Guess the impairment level!")

    if "current_image" not in st.session_state:
        image_path, class_name = get_random_test_image()
        st.session_state.current_image = image_path
        st.session_state.current_class = class_name
        st.session_state.idx = None
    if st.button("Load New Image"):
        image_path, class_name = get_random_test_image()
        st.session_state.current_image = image_path
        st.session_state.current_class = class_name
        st.session_state.idx = None
    #show chosen image
    if st.session_state.current_image:
        st.image(st.session_state.current_image, use_container_width =True) 
        st.write("Class:", st.session_state.current_class)

    class_options = ["No Impairment", "Very Mild Impairment", "Mild Impairment", "Moderate Impairment"]
    human_guess = st.radio("Select Severity:", class_options, key = 'idx', 
                           index = None if st.session_state.idx is None else class_options.index(st.session_state.idx))
    #show the recorded guess
    if human_guess:
        st.info(f"You guessed: **{human_guess}**")

    #model guess
    st.subheader("Model Prediction")
    #if st.button("Get Model Prediction"):
    #put model prediction here
    _, true_class, pred_class,_ , _ = predict_random_test_image(arch, test_root="mri_dataset/test", device="cpu")
    #check if model prediction matches human guess
    if human_guess and human_guess == pred_class:
        st.info("Your guess matches the model guess!")
        #check if those match the correct label
        if human_guess == true_class:
            st.success("You and the model guessed correctly!")
        else: 
            st.error("Neither you or the model guessed correctly!")
        
    elif human_guess: 
        st.info("Your guess does NOT match the model guess.")
        if human_guess == true_class:
            st.success("You guessed correctly, but the model didn't!")
        elif pred_class == true_class:
            st.error("The model guessed correctly, but you didn't!")
        else:
            st.error("Neither you or the model guessed correctly!")
    
    
IMAGE_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),   
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# CLASS_NAMES = [
#     "No Impairment",
#     "Very Mild Impairment",
#     "Mild Impairment",
#     "Moderate Impairment",
# ]

def predict_random_test_image(
    arch: str,
    test_root: str = "mri_dataset/test",
    device: str = "cpu",
): 
    image_path = st.session_state.current_image
    true_class = st.session_state.current_class
    img = Image.open(image_path).convert("RGB")
    transform = get_inference_transform()
    transformed = transform(img)
    if not isinstance(transformed, torch.Tensor):
        transformed = T.ToTensor()(transformed)
    transform_image = transformed.unsqueeze(0).to(device)
    
    if arch == "resnet50":
        model = RESNET50_MODEL.to(device)
        probs = generate_softmax_outputs(model, transform_image)
        pred_idx = int(probs.argmax().item())
        pred_class = CLASS_NAMES[pred_idx]
        pred_prob = probs[pred_idx].item()
    elif arch == "vgg16":
        model = VGG16_MODEL.to(device)
        probs = generate_softmax_outputs(model, transform_image)
        pred_idx = int(probs.argmax().item())
        pred_class = CLASS_NAMES[pred_idx]
        pred_prob = probs[pred_idx].item()
    elif arch == "Plain Average Ensemble": 
        (pred_class, pred_prob) = plain_average_ensemble(
        transform_image,
        models=[RESNET50_MODEL, VGG16_MODEL, ]
        )
    elif arch == "Weighted Average Ensemble":
        (pred_class, pred_prob) = weighted_average_ensemble(
            transform_image, 
            models=[RESNET50_MODEL, VGG16_MODEL]
        )
    else:
        raise ValueError(f"Unsupported architecture for prediction: {arch}")

    return image_path, true_class, pred_class, pred_prob, pred_prob

show_game_tab()