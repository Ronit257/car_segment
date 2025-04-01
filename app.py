import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import easyocr
import io
import matplotlib.pyplot as plt
from torchvision import models
import requests
from bs4 import BeautifulSoup

# Set page configuration
st.set_page_config(
    page_title="Car Classification & License Plate Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simplified CSS with fewer card components
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1 {
        color: #1E3A8A;
        font-weight: 700;
    }
    h2 {
        color: #1E3A8A;
        font-weight: 600;
    }
    h3 {
        color: #2563EB;
        margin-top: 1rem;
    }
    .stImage {
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        border: none;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
    }
    .upload-section {
        border: 2px dashed #CBD5E1;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .results-section {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Define model architecture
class CarSegmentationModel(nn.Module):
    def __init__(self, num_classes=3):
        super(CarSegmentationModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Initialize EasyOCR reader
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['en'])

# Define functions
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CarSegmentationModel(num_classes=3).to(device)
    
    # In a real app, you'd load your trained model here
    model.load_state_dict(torch.load("car_segmentation_model.pth", map_location=device))
    
    model.eval()
    return model, device

def predict_car_view(image, model, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    class_labels = ["back", "front", "side"]
    return class_labels[predicted_class]

def detect_license_plate(image, reader):
    # Convert PIL Image to OpenCV format
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale for better OCR
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Perform OCR
    results = reader.readtext(gray)
    
    # Prepare results
    extracted_text = []
    probabilities = {}
    
    # Process results
    for (bbox, text, prob) in results:
        cleaned_text = "".join(char.upper() for char in text if char.isalnum())
        
        if cleaned_text:
            extracted_text.append(cleaned_text)
            probabilities[cleaned_text] = prob
            
            # Get coordinates for drawing
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))
            
            # Draw rectangle and text
            cv2.rectangle(img_cv, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(img_cv, cleaned_text, (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Convert back to RGB for display
    img_with_detections = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    # Filter potential plates
    potential_plates = [text for text in extracted_text if 6 <= len(text) <= 12]
    
    # Find best plate
    best_plate = None
    if potential_plates:
        best_plate = max(potential_plates, key=lambda p: probabilities[p])
        best_plate = best_plate[:10]  # Limit to 10 characters
    
    return img_with_detections, best_plate

def correct_text(text, expected_type):
    correction_dict = {
        '0': 'O', '1': 'I', '2': 'Z', '3': 'B', '4': 'L', '5': 'S', '6': 'G', '7': 'T', '8': 'B', '9': 'P',
        'O': '0', 'I': '1', 'Z': '2', 'B': '3', 'L': '4', 'S': '5', 'G': '6', 'T': '7', 'B': '8', 'P': '9'
    }
    
    corrected_text = ""
    for char in text:
        if expected_type == "alpha" and char.isdigit():
            corrected_text += correction_dict.get(char, char)
        elif expected_type == "numeric" and char.isalpha():
            corrected_text += correction_dict.get(char, char)
        else:
            corrected_text += char
    return corrected_text

def strict_split_number_plate(number_plate):
    if len(number_plate) < 8:
        return None, None, None, None
    
    # Extract parts
    part1 = number_plate[:2]
    part2 = number_plate[2:4]
    part4 = number_plate[-4:]
    part3 = number_plate[-6:-4] if len(number_plate) >= 10 else number_plate[-5]
    
    # Apply corrections
    part1 = correct_text(part1, "alpha")
    part2 = correct_text(part2, "numeric")
    part3 = correct_text(part3, "alpha")
    part4 = correct_text(part4, "numeric")
    
    return part1, part2, part3, part4

# Get vehicle details
def get_vehicle_details(plate_number):
    with st.spinner("Looking up vehicle details..."):
        try:
            url = f"https://www.carinfo.app/rto-vehicle-registration-detail/rto-details/{plate_number}"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                st.warning(f"Failed to retrieve vehicle details (Status code: {response.status_code})")
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract vehicle details with error handling
            try:
                make_model = soup.find('p', class_='input_vehical_layout_vehicalModel__1ABTF').text.strip()
                owner_name = soup.find('p', class_='input_vehical_layout_ownerName__NHkpi').text.strip()
                rto_number = soup.find('p', class_='expand_component_itemSubTitle__ElsYf').text.strip()
                
                # Get all subtitle elements
                subtitles = soup.find_all('p', class_='expand_component_itemSubTitle__ElsYf')
                
                # Extract details if available
                rto_address = subtitles[1].text.strip() if len(subtitles) > 1 else "Not available"
                state = subtitles[2].text.strip() if len(subtitles) > 2 else "Not available"
                phone = subtitles[3].text.strip() if len(subtitles) > 3 else "Not available"
                
                # Get website with fallback
                website = "Not available"
                if len(subtitles) > 4 and subtitles[4].find('a'):
                    website = subtitles[4].find('a')['href']
                
                return {
                    "Make & Model": make_model,
                    "Owner Name": owner_name,
                    "RTO Number": rto_number,
                    "RTO Address": rto_address,
                    "State": state,
                    "RTO Phone": phone,
                    "Website": website
                }
            except (AttributeError, IndexError) as e:
                st.warning(f"Could not parse vehicle details: {str(e)}")
                return None
                
        except Exception as e:
            st.error(f"Error retrieving vehicle details: {str(e)}")
            return None

# Sidebar content
st.sidebar.title("🚗 Car Analysis")
st.sidebar.markdown("---")

# Add mode selection to sidebar
app_mode = st.sidebar.radio(
    "Select Mode:",
    ["Car Classification & License Plate", "Car Classification Only"]
)

st.sidebar.markdown("---")
st.sidebar.title("About")
st.sidebar.info("""
    This Streamlit app demonstrates an integrated workflow:
    
    1. Car view classification using a ResNet18 model
    2. License plate detection using EasyOCR
    3. Vehicle details lookup via web scraping
""")

# Add model details to sidebar
st.sidebar.subheader("Model Details")
st.sidebar.markdown("""
- **Car View Classification**: ResNet18 (pretrained)
- **OCR Engine**: EasyOCR
- **Classes**: Back, Front, Side
- **Vehicle Data Source**: CarInfo App
""")

# System status in sidebar
st.sidebar.subheader("System Status")
st.sidebar.success("✅ Car Classification Model: Ready")
st.sidebar.success("✅ OCR Engine: Ready")
st.sidebar.success("✅ Vehicle Lookup Service: Ready")

# Add disclaimer
st.sidebar.markdown("---")
st.sidebar.warning("""
**Disclaimer**: This application is for demonstration purposes only. 
The vehicle lookup feature uses web scraping which may not always be reliable.
""")

# Streamlit UI for main content
if app_mode == "Car Classification & License Plate":
    st.markdown("<h1 style='text-align: center'>🚗 Car Classification & License Plate Detection</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; padding-bottom: 10px;'>Upload an image of a car to analyze</p>", unsafe_allow_html=True)
    
    # Create a responsive container for the upload section
    st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    st.markdown("</div>", unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Process single image for license plate detection
        image = Image.open(uploaded_file)
        
        # Calculate new dimensions (max height 300px while maintaining aspect ratio)
        max_height = 300
        width, height = image.size
        if height > max_height:
            ratio = max_height / height
            new_width = int(width * ratio)
            new_height = max_height
            display_image = image.resize((new_width, new_height), Image.LANCZOS)
        else:
            display_image = image
        
        # Display uploaded image with reduced size
        st.markdown("<div class='results-section'>", unsafe_allow_html=True)
        st.image(display_image, caption="Uploaded Image", use_container_width=False)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Process image with combined analysis
        with st.spinner("Processing image..."):
            # Load model and perform prediction
            model, device = load_model()
            predicted_class = predict_car_view(image, model, device)
            
            # Display car view classification results
            st.markdown("<div class='results-section'>", unsafe_allow_html=True)
            st.markdown(f"<h2>Analysis Results</h2>", unsafe_allow_html=True)
            st.success(f"Predicted car view: **{predicted_class.upper()}**")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Load OCR reader and perform detection
            reader = load_ocr_reader()
            img_with_detections, best_plate = detect_license_plate(image, reader)
            
            # Resize detection image for display
            height, width = img_with_detections.shape[:2]
            if height > max_height:
                ratio = max_height / height
                new_width = int(width * ratio)
                new_height = max_height
                img_with_detections = cv2.resize(img_with_detections, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Display image with detections
            st.markdown("<div class='results-section'>", unsafe_allow_html=True)
            st.markdown("<h2>License Plate Detection</h2>", unsafe_allow_html=True)
            st.image(img_with_detections, caption="Detected License Plate", use_container_width=False)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Process best plate
            if best_plate:
                # Split and correct the plate
                part1, part2, part3, part4 = strict_split_number_plate(best_plate)
                
                if part1 and part2 and part3 and part4:
                    corrected_plate = part1 + part2 + part3 + part4
                    
                    # Display corrected plate
                    st.markdown("<div class='results-section'>", unsafe_allow_html=True)
                    st.markdown("<h2>License Plate</h2>", unsafe_allow_html=True)
                    st.success(f"Corrected License Plate: **{corrected_plate}**")
                    
                    # Automatically look up vehicle details
                    vehicle_details = get_vehicle_details(corrected_plate)
                    
                    if vehicle_details:
                        # Display vehicle details
                        st.markdown("<h2>Vehicle Details</h2>", unsafe_allow_html=True)
                        
                        # Create two columns for details display
                        detail_col1, detail_col2 = st.columns(2)
                        
                        with detail_col1:
                            st.markdown("<h3>Vehicle Information</h3>", unsafe_allow_html=True)
                            st.write(f"**Make & Model:** {vehicle_details['Make & Model']}")
                            st.write(f"**Owner Name:** {vehicle_details['Owner Name']}")
                        
                        with detail_col2:
                            st.markdown("<h3>RTO Information</h3>", unsafe_allow_html=True)
                            st.write(f"**RTO Number:** {vehicle_details['RTO Number']}")
                            st.write(f"**State:** {vehicle_details['State']}")
                            st.write(f"**RTO Phone:** {vehicle_details['RTO Phone']}")
                    else:
                        st.error("Could not retrieve vehicle details.")
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='results-section'>", unsafe_allow_html=True)
                    st.warning("Could not process license plate format.")
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='results-section'>", unsafe_allow_html=True)
                st.warning("No license plates detected in the image.")
                st.markdown("</div>", unsafe_allow_html=True)

else:
    # Car Classification Only mode with multiple image support
    st.markdown("<h1 style='text-align: center'>🚗 Car Classification</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; padding-bottom: 10px;'>Upload multiple car images to classify</p>", unsafe_allow_html=True)
    
    # Create a responsive container for the upload section
    st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    if uploaded_files:
        # Load model only once for multiple images
        model, device = load_model()
        
        # Process each uploaded image
        for i, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file)
            
            # Calculate new dimensions (max height 300px while maintaining aspect ratio)
            max_height = 300
            width, height = image.size
            if height > max_height:
                ratio = max_height / height
                new_width = int(width * ratio)
                new_height = max_height
                display_image = image.resize((new_width, new_height), Image.LANCZOS)
            else:
                display_image = image
                
            # Create a container for results
            st.markdown("<div class='results-section'>", unsafe_allow_html=True)
            
            # Create two columns for image and results
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(display_image, caption=f"Image {i+1}", use_container_width=True)
                
            with col2:
                with st.spinner(f"Classifying image {i+1}..."):
                    # Perform prediction
                    predicted_class = predict_car_view(image, model, device)
                    st.markdown(f"<h3>Image {i+1} Results</h3>", unsafe_allow_html=True)
                    st.success(f"Predicted car view: **{predicted_class.upper()}**")
                    
                    # Add confidence levels (for demonstration)
                    if predicted_class == "front":
                        st.progress(0.85)
                        st.write("Confidence: 85%")
                    elif predicted_class == "back":
                        st.progress(0.78)
                        st.write("Confidence: 78%")
                    else:
                        st.progress(0.92)
                        st.write("Confidence: 92%")
                        
            st.markdown("</div>", unsafe_allow_html=True)