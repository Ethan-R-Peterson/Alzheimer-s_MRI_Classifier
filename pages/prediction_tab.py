"""Code for the predcition tab for our streamlit UI should go here"""
import streamlit as st
import model_utils
from PIL import Image
from gradcam import GradCAM
import numpy as np
import cv2

def show_prediction_tab():
    """Use this function to help define the actual streamlit UI tab"""
    st.title("Prediction Tab")
    st.subheader("Upload your MRI scan for prediction")
    uploaded_file = st.file_uploader("Choose a file (jpg)", type=["jpg"])
    model_options = ["single-model", "multi-model"]
    model_variants = ["Resnet50", "VGG16"]
    multi_models = ["Ensemble"]
    
    model_selected = st.radio("Select Model Variant: ", model_options)
    if model_selected == "single-model":
        st.radio("Select which model you want to use: ", model_variants)
    else:
        st.radio("Select which model you want to use (only Ensemble available): ", multi_models)
        
    st.button("Get Prediction", on_click = fetch_prediction, args=(uploaded_file, model_selected, model_variants))
   


def fetch_prediction(uploaded_file, model_options, model_variants):
    if uploaded_file is not None:      
        pil_img = Image.open(uploaded_file).convert('RGB')
        transform = model_utils.get_inference_transform()
        tensor = transform(pil_img).unsqueeze(0)
        st.image(pil_img, caption='Uploaded MRI Scan.', use_container_width=True)
        st.write("")
        
        final_output = None
        gradcam_output = None
        predicted_class = None
        with st.spinner(text="Predicting in progress...", show_time=True):
            if model_options == "single-model":
                if model_variants == "Resnet50": 
                    final_output = model_utils.generate_softmax_outputs(model_utils.RESNET50_MODEL, tensor)
                    gradcam_output = GradCAM(model_utils.RESNET50_MODEL, "layer3").generate_cam(tensor, final_output.argmax())
                else:
                    final_output = model_utils.generate_softmax_outputs(model_utils.VGG16_MODEL, tensor)     
                    gradcam_output = GradCAM(model_utils.VGG16_MODEL).generate_cam(tensor, final_output.argmax())
                
                confidence = final_output.max().item() * 100
                predicted_class = model_utils.CLASS_NAMES[final_output.argmax()]
                img_resized = np.array(pil_img.resize((224,224)))
                heatmap = cv2.resize(gradcam_output, (224,224))
                heatmap = np.uint8(255 * heatmap)
                heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = cv2.addWeighted(img_resized, 0.6, heatmap_color, 0.4, 0)
                st.success(f"Prediction: {predicted_class} with {confidence}% confidence.")
                st.image(superimposed_img, caption='GradCAM output.', use_container_width=True)
                st.write("")
            else:
                st.write("Ensemble not imported yet") #note that Ensemble model has not been added yet
    else: 
        st.write("Please upload an MRI scan to get a prediction.")
            
            
        

        

show_prediction_tab()