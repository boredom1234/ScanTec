import os
import streamlit as st
from PIL import Image
from predictions import predict
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Global variables
project_folder = os.path.dirname(os.path.abspath(__file__))
folder_path = project_folder + '/images/'

# Grad-CAM functions
def get_gradcam(model, img_array, layer_name):
    grad_model = tf.keras.models.Model(inputs=[model.inputs], outputs=[model.get_layer(layer_name).output, model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predicted_class = np.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)

    # Check for None gradients
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

def overlay_gradcam_on_image(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

    return superimposed_img

def draw_roi(img, heatmap, threshold=0.6):
    heatmap_bin = np.where(heatmap > threshold, 1, 0).astype(np.uint8)
    contours, _ = cv2.findContours(heatmap_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 50:
            hull = cv2.convexHull(contour)
            cv2.drawContours(img, [hull], -1, (255, 0, 0), 2)

    return img

# Load model once globally
model = tf.keras.models.load_model("weights/ResNet50_Elbow_frac.h5")

# Streamlit App
def main():
    st.title("Bone Fracture Detection")
    
    # Header
    st.image(os.path.join(folder_path, "info.png"), width=40)
    st.header("Bone Fracture Detection System")

    # Info Text
    st.write(
        """
        Bone fracture detection system, upload an X-ray image for fracture detection.
        """
    )

    # Upload Image
    uploaded_file = st.file_uploader("Upload an X-ray Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # Display the uploaded image
        img = Image.open(uploaded_file)
        img_resized = img.resize((int(256 / img.height * img.width), 256))
        st.image(img_resized, caption="Uploaded Image", use_column_width=False)

        # Save the uploaded file to a temporary location
        temp_img_path = os.path.join("temp", uploaded_file.name)
        img.save(temp_img_path)

        # Prediction button
        if st.button("Predict"):
            # Get prediction result
            bone_type_result = predict(temp_img_path)
            result = predict(temp_img_path, bone_type_result)

            # Display result
            if result == 'fractured':
                st.error("Result: Fractured")
            else:
                st.success("Result: Normal")

            # Display Bone Type
            st.info(f"Type: {bone_type_result}")

            # Grad-CAM visualization
            st.subheader("Grad-CAM Visualization")
            
            img_array = np.expand_dims(image.img_to_array(img_resized), axis=0)
            img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
            
            # Generate Grad-CAM heatmap
            gradcam_layer_name = "conv5_block3_out"  # Update with the correct layer name
            heatmap = get_gradcam(model, img_array, gradcam_layer_name)
            
            # Overlay Grad-CAM heatmap on the original image
            superimposed_img = overlay_gradcam_on_image(temp_img_path, heatmap)

            # Draw ROI on the image
            roi_img = draw_roi(superimposed_img.copy(), heatmap)

            # Display Grad-CAM image
            st.image(roi_img, caption="Grad-CAM with ROI", use_column_width=True)

    # Save Results
    if uploaded_file and st.button("Save Result"):
        save_result(img_resized)

def save_result(image):
    # Function to save the result image
    temp_dir = st.text_input("Enter a file name to save the result (e.g., result.png):")
    
    if temp_dir:
        image.save(temp_dir)
        st.success(f"Image saved to {temp_dir}")


if __name__ == "__main__":
    main()
