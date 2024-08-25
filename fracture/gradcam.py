import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

def get_gradcam(model, img_array, layer_name):
    # Create a model that maps the input image to the activations of the last conv layer
    grad_model = tf.keras.models.Model(inputs=[model.inputs], outputs=[model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        # Watch the input image
        conv_outputs, predictions = grad_model(img_array)
        predicted_class = np.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    # Compute the gradients with respect to the output feature map of the last convolutional layer
    grads = tape.gradient(loss, conv_outputs)

    # Check for None gradients
    if grads is None:
        raise ValueError(f"Gradient calculation returned None. Check the layer name: {layer_name}")

    # Calculate the pooled gradients across the feature maps
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Convert EagerTensor to NumPy array for in-place assignment
    conv_outputs = conv_outputs[0].numpy()

    # Multiply each channel in the feature map by the average gradient of that channel
    for i in range(conv_outputs.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i].numpy()

    # Generate the heatmap
    heatmap = np.mean(conv_outputs, axis=-1)
    
    # Apply ReLU to remove negative values
    heatmap = np.maximum(heatmap, 0)
    
    # Normalize the heatmap between 0 and 1 for visualization
    heatmap /= np.max(heatmap)

    # Resize the heatmap to the size of the input image
    heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))

    return heatmap

def overlay_gradcam_on_image(img_path, heatmap, alpha=0.4):
    # Load the original image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB if necessary

    # Ensure the heatmap has the same size as the original image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Convert heatmap to RGB (3 channels)
    heatmap = np.uint8(255 * heatmap)  # Scale between 0 and 255
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose the heatmap onto the original image
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

    return superimposed_img

def draw_roi(img, heatmap, threshold=0.6):
    # Convert the heatmap into binary based on a threshold
    heatmap_bin = np.where(heatmap > threshold, 1, 0).astype(np.uint8)

    # Find contours (areas of activation)
    contours, _ = cv2.findContours(heatmap_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw red triangles (or other shapes) around the highest activation regions
    for contour in contours:
        if cv2.contourArea(contour) > 50:  # Filter out small contours based on area
            hull = cv2.convexHull(contour)  # Use convex hull to generate a triangle-like shape
            cv2.drawContours(img, [hull], -1, (0, 0, 255), 2)  # Red color in RGB

    return img

# Load pre-trained model
model = tf.keras.models.load_model("weights\ResNet50_Hand_frac.h5")
# model = tf.keras.models.load_model("weights\ResNet50_Hand_frac.h5")
# model = tf.keras.models.load_model("weights\ResNet50_Shoulder_frac.h5")

# Load and preprocess test image
img_path = "test\\Hand\\fractured\\broken.jpg"
img = image.load_img(img_path, target_size=(224, 224))
img_array = np.expand_dims(image.img_to_array(img), axis=0)
img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

# Generate Grad-CAM heatmap
gradcam_layer_name = "conv5_block3_add"  # Adjust as needed
heatmap = get_gradcam(model, img_array, gradcam_layer_name)

# Overlay heatmap on image and save result
superimposed_img = overlay_gradcam_on_image(img_path, heatmap)

# Generate and overlay ROI
roi_img = draw_roi(superimposed_img.copy(), heatmap)

# Save and display the final image with both Grad-CAM and ROI
output_path = "output_path_to_save_image_with_roi.jpg"
plt.imsave(output_path, roi_img)
plt.imshow(roi_img)
plt.show()
