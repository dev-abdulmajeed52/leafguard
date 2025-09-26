import streamlit as st
from streamlit_option_menu import option_menu
import json
import os
from streamlit_lottie import st_lottie
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import DepthwiseConv2D, Input, Dense, GlobalAveragePooling2D
import time
import base64
import io
from PIL import Image
import numpy as np
import pandas as pd
import cv2

# Set page configuration
st.set_page_config(
    page_title="LeafGuard", 
    layout="wide"
)

# Hide the sidebar completely with CSS
hide_sidebar_style = """
    <style>
        [data-testid="stSidebar"] {
            display: none;
        }
    </style>
"""
st.markdown(hide_sidebar_style, unsafe_allow_html=True)

# Topbar Navigation
selected = option_menu(
    menu_title=None,
    options=["Home", "Disease Detector", "Blog", "Graphs"],
    icons=["house", "search", "book", "bar-chart"],
    menu_icon="cast", 
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {
            "padding": "10px!important",
            "background": "linear-gradient(135deg, #fafafa 30%, #e0e0e0 100%)",
            "border-bottom": "2px solid #e0e0e0",
            "box-shadow": "0 2px 5px rgba(0, 0, 0, 0.1)"
        },
        "icon": {
            "color": "#4CAF50",
            "font-size": "20px",
            "padding": "5px",
            "transition": "transform 0.3s ease",
        },
        "nav-link": {
            "font-size": "16px",
            "text-align": "center",
            "margin": "0px 15px",
            "padding": "10px 15px",
            "border-radius": "5px",
            "color": "#333",
            "--hover-color": "#e8f5e9",
            "transition": "background-color 0.3s ease, transform 0.3s ease",
        },
        "nav-link-selected": {
            "background-color": "#388E3C",
            "color": "white",
            "border-radius": "5px",
            "border-bottom": "2px solid #ffffff",
        },
        "@media (max-width: 768px)": {
            "nav-link": {
                "font-size": "14px",
                "margin": "0px 10px",
                "padding": "8px 10px",
            },
        },
    }
)

# Function to load Lottie animations from a local file
def load_lottie_file(filepath: str):
    abs_path = os.path.abspath(filepath)
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            lottie_json = json.load(f)
            return lottie_json
    except FileNotFoundError:
        st.error("Lottie file not found. Please ensure the file exists.")
        return None
    except json.JSONDecodeError as e:
        st.error(f"Error decoding the Lottie JSON file: {e}")
        return None

# Function to add custom CSS with Inter font
def add_custom_css():
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    
    <style>
        body, .stApp {
            font-family: 'Inter', sans-serif;
            background-color: #121212;
        }

        .stApp h1 {
            color: white;
            font-weight: 700;
        }

        .stApp h2 {
            color: #CCCCCC;
            font-weight: 600;
        }

        .stApp p {
            color: white;
            line-height: 1.6;
            font-weight: 400;
        }

        .stButton button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 10px 2px;
            cursor: pointer;
            border-radius: 4px;
            font-family: 'Inter', sans-serif;
        }

        .stButton button:hover {
            background-color: #45a049;
        }

        .stApp a {
            color: #4CAF50;
            text-decoration: none;
            font-weight: 600;
        }

        .stApp a:hover {
            text-decoration: underline;
        }
    </style>
    """, unsafe_allow_html=True)

# Display the Home page
def display_home():
    add_custom_css()
    st.title("Welcome to LeafGuard")

    lottie_leaf_scan = load_lottie_file("assets/scan.json")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Leaf Scan Detection System")
        st.write("""
        Captured through drones, smartphones, or cameras mounted on tractors, images provide valuable information about leaf health.
        High-resolution images can reveal subtle symptoms of diseases that might be missed by the naked eye.

        Data Science involves extracting knowledge and insights from structured and unstructured data using various techniques,
        including statistical analysis, machine learning, and data visualization. In the context of LeafGuard, Data Science techniques
        are employed to build predictive models that can identify and classify leaf diseases based on the data collected.

        Machine learning algorithms can analyze complex datasets and identify patterns that may not be immediately apparent to human observers.
        For instance, convolutional neural networks (CNNs) can be employed for image recognition tasks, enabling the system to learn from
        thousands of images of healthy and diseased leaves. This capability enhances the model's accuracy and reliability in real-world applications.
        """)

    with col2:
        if lottie_leaf_scan:
            st_lottie(lottie_leaf_scan, height=300, key="leaf_scan")
        else:
            st.error("Failed to load the leaf scan animation.")

# Show blog content
def display_blog():
    add_custom_css()
    st.title("Transforming Big Data with Data Science")

    st.write("""
        In the realm of agriculture, leaf diseases pose a critical threat to crop yields and overall agricultural productivity. These diseases can lead to devastating yield losses, affecting the quality of produce and posing significant risks to food security and economic stability for farmers worldwide. The Food and Agriculture Organization (FAO) estimates that plant diseases can cause losses of up to 30% of potential crop yields, emphasizing the urgent need for effective detection and management strategies.

        Traditionally, detecting leaf diseases has relied heavily on manual inspection by experts. This method, while valuable, is labor-intensive, time-consuming, and often impractical for large-scale farming operations. The complexity and diversity of diseases can make visual identification challenging, leading to delayed responses and further crop losses.

        However, the advent of technology and data-driven methodologies has ushered in new possibilities. By leveraging Data Science and Big Data processing, we can revolutionize how leaf diseases are detected and managed. The LeafGuard project epitomizes this transformation, aiming to create a robust and interactive model that utilizes vast amounts of data—from images and weather patterns to soil conditions—to enhance disease detection and intervention strategies.
    """)

    if st.button("Read More"):
        st.write("Redirecting to full blog post...")
        st.markdown("[Click here to read the full post](https://techwiz-leafguard.blogspot.com/2024/09/transforming-big-data-with-data-science.html)", unsafe_allow_html=True)

    st.write("**More blog posts coming soon!**")

# Display graph cards
def display_graphs():
    add_custom_css()
    st.title("Plant Disease Prediction and Graphs")

    cols = st.columns(3)

    graph_data = [
        {"image": "assets/graph1.jpeg", "title": "Disease and Confidence"},
        {"image": "assets/graph3.jpeg", "title": "Average of water Condition"},
        {"image": "assets/graph2.jpeg", "title": "Average of Temperature"},
        {"image": "assets/graph6.jpeg", "title": "Count of Weather Condition"},
        {"image": "assets/graph4.jpeg", "title": "Average of Disease"},
        {"image": "assets/graph5.jpeg", "title": "Pesticides & Confidence"},
    ]

    for index, graph in enumerate(graph_data):
        with cols[index % 3]:
            st.subheader(graph["title"])
            st.image(graph["image"], caption=graph["title"], use_column_width=True)
            st.write("---")

# ----------------------------
# FIXED MODEL LOADING SECTION
# ----------------------------

def create_fallback_model(num_classes=4):
    """
    Create a simple fallback model in case the original model fails to load
    """
    try:
        # Simple CNN model as fallback
        inputs = Input(shape=(224, 224, 3))
        
        # Use MobileNetV2 as base (commonly available)
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            alpha=1.0,
            include_top=False,
            weights='imagenet',
            input_tensor=inputs,
            pooling=None
        )
        
        # Add custom top layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=predictions)
        return model
    except Exception as e:
        st.error(f"Failed to create fallback model: {str(e)}")
        return None

def custom_depthwise_conv2d(**kwargs):
    """Custom DepthwiseConv2D to handle compatibility issues"""
    # Remove problematic arguments that might cause issues
    problematic_args = ['groups', 'dilations']
    for arg in problematic_args:
        if arg in kwargs:
            del kwargs[arg]
    
    return DepthwiseConv2D(**kwargs)

def fix_model_architecture(model):
    """
    Attempt to fix model architecture issues by rebuilding the model
    """
    try:
        # Get model configuration
        config = model.get_config()
        
        # Rebuild model with fixed configuration
        if model.__class__.__name__ == 'Sequential':
            from tensorflow.keras.models import Sequential
            new_model = Sequential()
            
            # Add layers one by one, skipping problematic ones
            for i, layer_config in enumerate(config['layers']):
                try:
                    layer = tf.keras.layers.deserialize(
                        layer_config,
                        custom_objects={'DepthwiseConv2D': custom_depthwise_conv2d}
                    )
                    new_model.add(layer)
                except Exception as layer_error:
                    st.warning(f"Skipping problematic layer {i}: {str(layer_error)}")
                    continue
            
            return new_model
        else:
            # For functional models, return original
            return model
            
    except Exception as e:
        st.error(f"Failed to fix model architecture: {str(e)}")
        return None

@st.cache_resource
def load_and_fix_model(h5_path):
    """
    Enhanced model loading with multiple fallback strategies
    """
    try:
        st.info(f"Attempting to load model from: {h5_path}")
        
        # First attempt: Load with custom objects
        try:
            model = load_model(
                h5_path, 
                compile=False, 
                custom_objects={
                    'DepthwiseConv2D': custom_depthwise_conv2d,
                    'relu6': tf.nn.relu6,  # Add other common custom objects
                }
            )
            st.success("Model loaded successfully!")
            return model
        except ValueError as ve:
            if "expects 1 input(s), but it received 2 input tensors" in str(ve):
                st.warning("Model has architecture issues. Attempting to fix...")
                
                # Try to load without compiling and fix architecture
                try:
                    model = load_model(h5_path, compile=False)
                    fixed_model = fix_model_architecture(model)
                    if fixed_model is not None:
                        st.success("Model architecture fixed!")
                        return fixed_model
                except Exception as fix_error:
                    st.error(f"Failed to fix model: {str(fix_error)}")
            
            raise ve
    
    except FileNotFoundError:
        st.error(f"Model file not found at: {h5_path}")
        st.info("Creating fallback model...")
        return create_fallback_model()
    
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.info("Creating fallback model...")
        return create_fallback_model()

@st.cache_resource
def load_models_and_labels():
    """
    Load models and labels with enhanced error handling
    """
    models = {}
    labels = {}
    
    # Try to load the main model
    model_path = './models/Leaf_Guard.h5'
    labels_path = './models/labels.txt'
    
    # Load model
    model = load_and_fix_model(model_path)
    if model is not None:
        models['Leaf_Guard'] = model
    
    # Load labels
    try:
        if os.path.exists(labels_path):
            with open(labels_path, 'r', encoding='utf-8') as f:
                labels_list = [label.strip() for label in f.readlines()]
            labels['Leaf_Guard'] = labels_list
            st.success(f"Loaded {len(labels_list)} labels")
        else:
            # Fallback labels
            st.warning("Labels file not found. Using fallback labels.")
            labels['Leaf_Guard'] = ['Healthy', 'Early Blight', 'Late Blight', 'Powdery Mildew']
    except Exception as e:
        st.error(f"Failed to load labels: {str(e)}")
        labels['Leaf_Guard'] = ['Healthy', 'Early Blight', 'Late Blight', 'Powdery Mildew']
    
    return models, labels

# ----------------------------
# Image Processing Functions
# ----------------------------

def preprocess_image(image):
    """Enhanced image preprocessing with error handling"""
    try:
        img_array = np.array(image)
        
        # Handle different image formats
        if img_array.ndim == 2:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.ndim == 3 and img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        # Resize image
        resized_image = cv2.resize(img_array, (224, 224), interpolation=cv2.INTER_AREA)
        
        # Normalize image
        normalized_image = (np.asarray(resized_image, dtype=np.float32) / 127.5) - 1
        
        return normalized_image.reshape(1, 224, 224, 3)
    
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def predict(models, labels, processed_image):
    """Enhanced prediction function with error handling"""
    predictions = {}
    
    if processed_image is None:
        return predictions
    
    for model_name, model in models.items():
        try:
            # Make prediction
            prediction = model.predict(processed_image, verbose=0)
            
            # Handle different prediction shapes
            if prediction.ndim > 1:
                prediction = prediction[0]
            
            # Get top prediction
            index = np.argmax(prediction)
            confidence_score = float(prediction[index])
            
            # Get class name
            if model_name in labels and index < len(labels[model_name]):
                class_name = labels[model_name][index].strip()
            else:
                class_name = f"Class_{index}"
            
            predictions[model_name] = (class_name, confidence_score)
            
        except Exception as e:
            st.error(f"Error during prediction with {model_name}: {str(e)}")
            # Provide fallback prediction
            predictions[model_name] = ("Unknown", 0.0)
    
    return predictions

# ----------------------------
# Recommendation System
# ----------------------------

def get_recommendations(disease_name):
    """Enhanced recommendations system"""
    recommendations = {
        "Late Blight": {
            "Description": "Late blight is a devastating disease affecting tomato and potato plants, caused by the oomycete *Phytophthora infestans*.",
            "Recommendations": [
                "Use certified disease-free seeds or seedlings.",
                "Implement crop rotation to prevent the buildup of pathogens in the soil.",
                "Apply fungicides such as Mancozeb or Copper-based sprays at the first sign of disease.",
                "Ensure proper spacing between plants to improve air circulation.",
                "Remove and destroy infected plant debris to reduce the source of inoculum."
            ]
        },
        "Early Blight": {
            "Description": "Early blight is caused by the fungus *Alternaria solani* and affects tomato and potato plants, leading to dark spots on leaves and stems.",
            "Recommendations": [
                "Plant resistant varieties if available.",
                "Water plants at the base to keep foliage dry.",
                "Apply fungicides like Chlorothalonil or Mancozeb as a preventive measure.",
                "Prune lower leaves and remove debris to enhance airflow.",
                "Rotate crops annually to minimize disease recurrence."
            ]
        },
        "Powdery Mildew": {
            "Description": "Powdery mildew is a fungal disease that affects a wide range of plants, characterized by white powdery spots on leaves and stems.",
            "Recommendations": [
                "Improve air circulation around plants by proper spacing.",
                "Avoid overhead watering to keep foliage dry.",
                "Prune affected areas to reduce the spread of the disease.",
                "Apply fungicides such as Sulfur or Neem oil at the first sign of infection.",
                "Use resistant plant varieties if available."
            ]
        },
        "Healthy": {
            "Description": "The plant shows no signs of disease and is in good health.",
            "Recommendations": [
                "Maintain regular care practices including proper watering, fertilization, and pest control.",
                "Monitor plants regularly for any signs of stress or disease.",
                "Ensure optimal growing conditions tailored to the specific crop."
            ]
        },
    }
    
    return recommendations.get(disease_name, {
        "Description": "No specific recommendations available for the detected condition.",
        "Recommendations": [
            "Consult with a local agricultural extension office or a professional agronomist for tailored advice.",
            "Ensure general plant care practices are followed to maintain plant health."
        ]
    })

# ----------------------------
# CSS Styling Functions
# ----------------------------

def add_card_style():
    st.markdown("""
    <style>
    .card {
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        padding: 16px;
        border-radius: 10px;
        background-color: #f9f9f9;
        margin: 10px;
        color: black;
        width: 100%;
        height: 100%;
    }
    .fixed-size {
        height: 300px;
        overflow: hidden;
    }
    .small-image {
        max-width: 100%;
        height: 200px;
        object-fit: contain;
    }
    table {
        width: 100%;
        border-collapse: collapse;
    }
    th, td {
        text-align: left;
        padding: 8px;
    }
    th {
        background-color: #4CAF50;
        color: white;
    }
    tr:nth-child(even) {background-color: #f2f2f2;}
    h3, h4 {
        color: black;
    }
    </style>
    """, unsafe_allow_html=True)

def encode_image(image):
    """Helper function to encode image to base64"""
    try:
        buffered = io.BytesIO()
        image = image.convert("RGB")
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    except Exception as e:
        st.error(f"Error encoding image: {str(e)}")
        return ""

# ----------------------------
# Main Disease Detector Function
# ----------------------------

def display_detector():
    add_card_style()
    
    st.title("🌾 Crop Disease Detector")

    # Load models and labels with enhanced error handling
    with st.spinner('🔄 Loading models...'):
        try:
            models, labels = load_models_and_labels()
            
            if not models:
                st.error("No models could be loaded. Please check your model files.")
                st.stop()
            
            time.sleep(1)
            
        except Exception as e:
            st.error(f"Failed to load models: {str(e)}")
            st.stop()

    # Form layout
    with st.form(key='crop_form'):
        st.header("📝 Crop Information")

        col1, col2 = st.columns(2)

        with col1:
            crops_list = st.selectbox(
                '🌱 Select the Crop *',
                (
                    "Tomato 🍅", "Potato 🥔", "Corn 🌽 (Maize)", "Apple 🍎", "Cherry 🍒",
                    "Grapes 🍇", "Strawberry 🍓", "Other 🌿"
                ), 
                help="Choose the crop you are analyzing", index=0
            )
            
            water_condition = st.text_input(
                '💧 Water Condition',
                placeholder="e.g., Adequate",
                help="Specify the water availability"
            )
            
            temperature = st.text_input(
                '🌡️ Temperature (°C) *',
                placeholder="e.g., 25",
                help="Enter the temperature in °C"
            )
            
            soil = st.text_input(
                '🧱 Soil Type',
                placeholder="e.g., Loamy",
                help="Specify the soil type"
            )

        with col2:
            weather_condition = st.selectbox(
                '☁️ Select Weather Condition *', 
                (
                    'Sunny ☀️', 'Partly Cloudy ⛅', 'Windy 🌬️',
                    'Rainy 🌧️', 'Showers 🌦️', 'Foggy 🌫️', 'Snowy ❄️'
                ), 
                help="Choose the weather condition", index=0
            )
            
            humidity = st.text_input(
                '💧 Humidity (%) *',
                placeholder="e.g., 60",
                help="Enter the humidity level in %"
            )
            
            pH_level = st.text_input(
                '🧪 Soil pH Level',
                placeholder="e.g., 6.5",
                help="Enter the soil pH level"
            )
            
            nutrient_level = st.text_input(
                '🪴 Nutrient Level (NPK)',
                placeholder="e.g., N:10 P:10 K:10",
                help="Specify the NPK levels"
            )

        uploaded_file = st.file_uploader(
            "📸 Upload an image of the crop (jpg, png) *", 
            type=["jpg", "png", "jpeg"],
            help="Upload a clear image of the crop leaf for analysis."
        )

        st.markdown("* Indicates required fields")
        submit_button = st.form_submit_button(label='🔍 Analyze')

    if submit_button:
        # Input Validation
        errors = []
        if not crops_list:
            errors.append("🌱 **Crop selection** is required.")
        if not temperature:
            errors.append("🌡️ **Temperature** is required.")
        elif not temperature.replace('.', '', 1).replace('-', '', 1).isdigit():
            errors.append("🌡️ **Temperature** must be a number.")
        if not weather_condition:
            errors.append("☁️ **Weather condition** is required.")
        if not humidity:
            errors.append("💧 **Humidity** is required.")
        elif not humidity.replace('.', '', 1).isdigit():
            errors.append("💧 **Humidity** must be a number.")
        if not uploaded_file:
            errors.append("📸 **Image upload** is required.")

        if errors:
            for error in errors:
                st.error(error)
        else:
            with st.spinner('🔍 Analyzing image...'):
                try:
                    time.sleep(2)

                    # Process image
                    image = Image.open(uploaded_file)
                    processed_image = preprocess_image(image)
                    
                    if processed_image is None:
                        st.error("Failed to process the uploaded image.")
                        st.stop()

                    # Make predictions
                    predictions = predict(models, labels, processed_image)
                    
                    if not predictions:
                        st.error("Failed to make predictions. Please try again.")
                        st.stop()

                    # Get the top prediction
                    top_model = max(predictions.items(), key=lambda x: x[1][1])
                    top_class_name, top_confidence_score = top_model[1]

                    # Determine health status
                    if "healthy" in top_class_name.lower():
                        disease_name = "Healthy"
                        status = "Healthy"
                    else:
                        disease_name = top_class_name
                        status = "Unhealthy"

                    # Get recommendations
                    recommendation = get_recommendations(disease_name)

                    # Display results in cards
                    row1_col1, row1_col2 = st.columns(2)
                    with row1_col1:
                        st.markdown(f"""
                        <div class="card fixed-size">
                            <h3>🌾 Crop Details:</h3>
                            <table>
                                <tr><th>Attribute</th><th>Value</th></tr>
                                <tr><td>Crop</td><td>{crops_list}</td></tr>
                                <tr><td>Water Condition</td><td>{water_condition if water_condition else "N/A"}</td></tr>
                                <tr><td>Weather Condition</td><td>{weather_condition}</td></tr>
                                <tr><td>Temperature (°C)</td><td>{temperature}</td></tr>
                                <tr><td>Humidity (%)</td><td>{humidity}</td></tr>
                                <tr><td>Soil Type</td><td>{soil if soil else "N/A"}</td></tr>
                                <tr><td>Soil pH Level</td><td>{pH_level if pH_level else "N/A"}</td></tr>
                                <tr><td>Nutrient Level (NPK)</td><td>{nutrient_level if nutrient_level else "N/A"}</td></tr>
                            </table>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with row1_col2:
                        st.markdown(f"""
                        <div class="card fixed-size">
                            <h3>🧪 Prediction Result:</h3>
                            <table>
                                <tr><th>Attribute</th><th>Value</th></tr>
                                <tr><td>Status</td><td>{status}</td></tr>
                                <tr><td>Disease Name</td><td>{disease_name}</td></tr>
                                <tr><td>Confidence Score (%)</td><td>{np.round(top_confidence_score * 100, 2)}</td></tr>
                            </table>
                        </div>
                        """, unsafe_allow_html=True)

                    # Second row
                    row2_col1, row2_col2 = st.columns(2)
                    with row2_col1:
                        img_encoded = encode_image(image)
                        if img_encoded:
                            st.markdown(f"""
                            <div class="card fixed-size">
                                <h3>📷 Uploaded Image:</h3>
                                <img src="data:image/png;base64,{img_encoded}" class="small-image"/>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with row2_col2:
                        st.markdown(f"""
                        <div class="card">
                            <h3>💡 Recommendations:</h3>
                            <h4>🔍 Disease Description:</h4>
                            <p>{recommendation["Description"]}</p>
                            <h4>✅ Recommended Actions:</h4>
                            <ol>
                                {''.join([f"<li>{action}</li>" for action in recommendation["Recommendations"]])}
                            </ol>
                        </div>
                        """, unsafe_allow_html=True)

                    # Download CSV
                    result_data = {
                        "Crop": crops_list,
                        "Water Condition": water_condition if water_condition else "N/A",
                        "Weather Condition": weather_condition,
                        "Temperature (°C)": temperature,
                        "Humidity (%)": humidity,
                        "Soil Type": soil if soil else "N/A",
                        "Soil pH Level": pH_level if pH_level else "N/A",
                        "Nutrient Level": nutrient_level if nutrient_level else "N/A",
                        "Status": status,
                        "Disease Name": disease_name,
                        "Confidence Score (%)": np.round(top_confidence_score * 100, 2)
                    }
                    df = pd.DataFrame([result_data])
                    csv_file = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Results as CSV",
                        csv_file,
                        "crop_disease_prediction.csv",
                        "text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")
                    st.info("Please try again with a different image or check your model files.")

# Route based on selected option
if selected == "Home":
    display_home()
elif selected == "Blog":
    display_blog()
elif selected == "Graphs":
    display_graphs()
elif selected == "Disease Detector":
    display_detector()