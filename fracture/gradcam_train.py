import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# 1. Dataset Loading Function (Your function)
def load_path(path, part):
    """
    Load X-ray dataset
    """
    dataset = []
    for folder in os.listdir(path):
        folder = path + '/' + str(folder)
        if os.path.isdir(folder):
            for body in os.listdir(folder):
                if body == part:
                    body_part = body
                    path_p = folder + '/' + str(body)
                    for id_p in os.listdir(path_p):
                        patient_id = id_p
                        path_id = path_p + '/' + str(id_p)
                        for lab in os.listdir(path_id):
                            if lab.split('_')[-1] == 'positive':
                                label = 'fractured'
                            elif lab.split('_')[-1] == 'negative':
                                label = 'normal'
                            path_l = path_id + '/' + str(lab)
                            for img in os.listdir(path_l):
                                img_path = path_l + '/' + str(img)
                                dataset.append(
                                    {
                                        'body_part': body_part,
                                        'patient_id': patient_id,
                                        'label': label,
                                        'image_path': img_path
                                    }
                                )
    return dataset

# 2. Load the dataset using the load_path function
dataset = load_path('Dataset', 'Shoulder')

# 3. Prepare image data and labels
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    return img_array

# Convert dataset to lists of images and labels
images = [preprocess_image(item['image_path']) for item in dataset]
labels = [1 if item['label'] == 'fractured' else 0 for item in dataset]  # 1 for fractured, 0 for normal

# Convert to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# 4. Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# 5. Build the Pre-trained ResNet50 Model
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model layers

# Build the full model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# 6. Train the Model
class_weights = {0: 1.0, 1: 2.0}  # Example of class weighting

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    class_weight=class_weights
)

# Fine-tuning (Unfreeze some layers of the base model)
base_model.trainable = True
for layer in base_model.layers[:-10]:  # Fine-tune last 10 layers
    layer.trainable = False

# Re-compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # Lower learning rate for fine-tuning
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# Fine-tune the model
history_fine = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    class_weight=class_weights
)

# 7. Grad-CAM Implementation
def get_gradcam(model, img_array, layer_name):
    grad_model = tf.keras.models.Model(inputs=[model.inputs], outputs=[model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predicted_class = np.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)
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
            cv2.drawContours(img, [hull], -1, (255, 0, 0), 2)  # Red ROI
    return img

# 8. Test Grad-CAM on a Test Image
img_path = "Dataset/test/Shoulder/patient11749/study1_positive/image1.png"
img = image.load_img(img_path, target_size=(224, 224))
img_array = np.expand_dims(image.img_to_array(img), axis=0)
img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

gradcam_layer_name = "conv5_block3_out"
heatmap = get_gradcam(model, img_array, gradcam_layer_name)
superimposed_img = overlay_gradcam_on_image(img_path, heatmap)

roi_img = draw_roi(superimposed_img.copy(), heatmap)

output_path = "output_image_with_roi.jpg"
plt.imsave(output_path, roi_img)
plt.imshow(roi_img)
plt.show()
