import os
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
from predictions import predict
import io

# Fixed dimensions for images
FIXED_WIDTH = 400

# Global variables
project_folder = os.path.dirname(os.path.abspath(__file__))
folder_path = project_folder + '/images/'

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

# Function to overlay Grad-CAM heatmap on the image
def overlay_gradcam_on_image(img, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    if img.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    if len(img.shape) == 2 or img.shape[-1] != 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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

# Function to save the result image
def save_result(image):
    temp_dir = st.text_input("Enter a file name to save the result (e.g., result.png):")
    if temp_dir:
        image.save(temp_dir)
        st.success(f"Image saved to {temp_dir}")

# Streamlit App
def main():
    st.title("Fracture Detection with Grad-CAM (ROI Detection)")

    # Compact layout: Upload and display image
    st.subheader("Step 1: Upload X-ray Image")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        img_resized = img.resize((int(256 / img.height * img.width), 256))
        st.image(img_resized, caption="Uploaded Image", use_column_width=False)

        # Step 2: Fracture Detection and Prediction
        if st.button("Analyze Fracture"):
            st.subheader("Step 2: Fracture Detection")
            bone_type_result = predict(uploaded_file)  # Predict body part (Elbow, Hand, Shoulder)
            result = predict(uploaded_file, bone_type_result)  # Predict fracture or not

            if result == 'fractured':
                st.error(f"Fracture Detected in {bone_type_result}")
                
                # Grad-CAM section (only shown if fracture is detected)
                st.subheader("Step 3: Grad-CAM Visualization")
                model_map = {
                    "Elbow": "ResNet50_Elbow_frac",
                    "Hand": "ResNet50_Hand_frac",
                    "Shoulder": "ResNet50_Shoulder_frac"
                }
                model_name = model_map.get(bone_type_result)

                if model_name:
                    model_path = f"weights/{model_name}.h5"
                    model = tf.keras.models.load_model(model_path)

                    # Grad-CAM layer selection (inside expander for optional controls)
                    with st.expander("Advanced Settings"):
                        gradcam_layer_name = st.text_input("Enter Grad-CAM layer", "conv5_block3_add")
                        alpha = st.slider("Overlay transparency (alpha)", 0.0, 1.0, 0.4, step=0.05)
                        threshold = st.slider("ROI Threshold", 0.0, 1.0, 0.6, step=0.05)
                    
                    # Process image for Grad-CAM
                    img_array = np.expand_dims(image.img_to_array(img.resize((224, 224))), axis=0)
                    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
                    heatmap = get_gradcam(model, img_array, gradcam_layer_name)

                    # Display Grad-CAM results
                    img_array_disp = np.array(img.convert('RGB'))
                    superimposed_img = overlay_gradcam_on_image(img_array_disp.copy(), heatmap, alpha)
                    st.image(superimposed_img, caption="Grad-CAM Heatmap", width=FIXED_WIDTH)

                    # Draw ROI
                    roi_img = draw_roi(superimposed_img.copy(), heatmap, threshold)
                    st.image(roi_img, caption="Region of Interest (ROI)", width=FIXED_WIDTH)

                    # Download ROI image
                    roi_img_pil = Image.fromarray(roi_img)
                    buf = io.BytesIO()
                    roi_img_pil.save(buf, format="JPEG")
                    byte_im = buf.getvalue()
                    st.download_button("Download ROI Image", byte_im, "roi_image.jpg", "image/jpeg")

            else:
                st.success(f"No Fracture Detected in {bone_type_result}")

        # Optional: Save results in a compact manner
        with st.expander("Save Results"):
            if st.button("Save Result"):
                save_result(img_resized)

if __name__ == "__main__":
    main()
