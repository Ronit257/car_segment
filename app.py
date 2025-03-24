import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="Car View Segmentation",
    page_icon="🚗",
    layout="wide"
)

# Define the model class
class CarSegmentationModel(nn.Module):
    def __init__(self, num_classes=3):
        super(CarSegmentationModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load the model
@st.cache_resource
def load_model(num_classes=3, path="car_segmentation_model.pth"):
    model = CarSegmentationModel(num_classes=num_classes).to(device)
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model, True
    except Exception as e:
        return None, False

# Function to predict from an image
def predict_image(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    class_labels = ["back", "front", "side"]  # Adjust based on dataset
    result = {
        "prediction": class_labels[predicted_class],
        "probabilities": {class_labels[i]: float(prob) for i, prob in enumerate(probabilities.squeeze().tolist())}
    }
    
    return result

# Main function
def main():
    st.title("🚗 Car View Segmentation App")
    
    # Sidebar for model loading
    st.sidebar.header("Model Configuration")
    
    model_path = st.sidebar.text_input(
        "Model Path", 
        "car_segmentation_model.pth",
        help="Path to your trained model file"
    )
    
    num_classes = st.sidebar.number_input(
        "Number of Classes", 
        min_value=1, 
        max_value=10, 
        value=3,
        help="Number of classes in your model"
    )
    
    # Load model button
    if st.sidebar.button("Load Model"):
        with st.spinner("Loading model..."):
            model, success = load_model(num_classes, model_path)
            if success:
                st.sidebar.success("Model loaded successfully!")
                st.session_state['model'] = model
                st.session_state['model_loaded'] = True
            else:
                st.sidebar.error(f"Failed to load model from {model_path}")
                st.session_state['model_loaded'] = False
    
    # Check if model is loaded
    if 'model_loaded' not in st.session_state:
        # Try to load the model by default
        with st.spinner("Loading default model..."):
            model, success = load_model(num_classes, model_path)
            if success:
                st.sidebar.success("Default model loaded successfully!")
                st.session_state['model'] = model
                st.session_state['model_loaded'] = True
            else:
                st.sidebar.warning("Please load a model to start predictions")
                st.session_state['model_loaded'] = False
    
    # Main content
    st.header("Upload Images for Prediction")
    
    # File uploader for multiple images
    uploaded_files = st.file_uploader(
        "Choose one or more images", 
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.session_state.get('model_loaded', False):
        st.subheader("Predictions")
        
        # Create columns for displaying results
        cols = st.columns(3)
        
        for i, uploaded_file in enumerate(uploaded_files):
            col_idx = i % 3
            
            with cols[col_idx]:
                # Display the image
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Uploaded Image {i+1}", use_container_width=True)
                
                # Make prediction
                with st.spinner("Analyzing..."):
                    result = predict_image(image, st.session_state['model'])
                
                # Display prediction results
                st.success(f"Predicted View: {result['prediction'].upper()}")
                
                # Create a bar chart for probabilities
                # st.write("Confidence Scores:")
                # for label, prob in result['probabilities'].items():
                #     st.progress(prob)
                #     st.write(f"{label.capitalize()}: {prob:.2%}")
                
                st.divider()
    
    elif not st.session_state.get('model_loaded', False):
        st.info("Please load a model from the sidebar first")
    
    # Information section
    with st.expander("About This App"):
        st.write("""
        This app uses a ResNet18-based model to classify car images into different view categories:
        - Back View
        - Front View
        - Side View
        
        Upload one or more images to get predictions. The model will analyze each image and 
        display the predicted view along with confidence scores.
        """)

if __name__ == "__main__":
    main()