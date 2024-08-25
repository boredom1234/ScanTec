import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image
import io

# Fixed dimensions for images
FIXED_WIDTH = 400  # You can adjust this value as needed
FIXED_HEIGHT = 300  # Optional: You can set height instead if needed

# Function to generate Grad-CAM heatmap
def get_gradcam(model, img_array, layer_name):
    grad_model = tf.keras.models.Model(inputs=[model.inputs], outputs=[model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predicted_class = np.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        raise ValueError(f"Gradient calculation returned None. Check the layer name: {layer_name}")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0].numpy()

    for i in range(conv_outputs.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i].numpy()

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))

    return heatmap

# Updated Function to overlay Grad-CAM on the image
def overlay_gradcam_on_image(img, heatmap, alpha=0.4):
    # Ensure the heatmap is in the same size as the original image
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Ensure both images are in the same size
    if img.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Convert the original image to RGB format if needed (in case it is in BGR)
    if len(img.shape) == 2 or img.shape[-1] != 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Overlay heatmap on the image
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return superimposed_img

# Function to draw the region of interest (ROI)
def draw_roi(img, heatmap, threshold=0.6):
    heatmap_bin = np.where(heatmap > threshold, 1, 0).astype(np.uint8)
    contours, _ = cv2.findContours(heatmap_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 50:
            hull = cv2.convexHull(contour)
            cv2.drawContours(img, [hull], -1, (0, 0, 255), 2)

    return img

# Streamlit app starts here
st.title("Grad-CAM with ROI Detection")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width=FIXED_WIDTH)
    img_array = np.expand_dims(image.img_to_array(img.resize((224, 224))), axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

    # Model selection
    model_name = st.selectbox("Choose a model", ["ResNet50_Elbow_frac", "ResNet50_Hand_frac", "ResNet50_Shoulder_frac"])
    model_path = f"weights/{model_name}.h5"
    
    # Load the chosen model
    model = tf.keras.models.load_model(model_path)

    # Layer selection
    gradcam_layer_name = st.text_input("Enter the Grad-CAM layer name", "conv5_block3_add")

    # Slider to adjust Grad-CAM overlay alpha
    alpha = st.slider("Overlay transparency (alpha)", min_value=0.0, max_value=1.0, value=0.4, step=0.05)

    # Generate Grad-CAM heatmap
    heatmap = get_gradcam(model, img_array, gradcam_layer_name)

    # Overlay Grad-CAM heatmap on original image
    img_array_disp = np.array(img.convert('RGB'))  # Convert PIL image to NumPy array for OpenCV
    superimposed_img = overlay_gradcam_on_image(img_array_disp.copy(), heatmap, alpha=alpha)

    st.image(superimposed_img, caption="Grad-CAM Superimposed", width=FIXED_WIDTH)

    # Slider to adjust ROI threshold
    threshold = st.slider("ROI Threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.05)

    # Draw ROI on the image
    roi_img = draw_roi(superimposed_img.copy(), heatmap, threshold=threshold)

    st.image(roi_img, caption="Image with ROI", width=FIXED_WIDTH)

    # Option to download the ROI image
    roi_img_pil = Image.fromarray(roi_img)
    buf = io.BytesIO()
    roi_img_pil.save(buf, format="JPEG")
    byte_im = buf.getvalue()

    st.download_button("Download ROI Image", byte_im, "roi_image.jpg", "image/jpeg")
