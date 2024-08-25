import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import pandas as pd
import os.path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def load_path(path, part):
    """
    load X-ray dataset
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

def get_gradcam(model, img_array, layer_name):
    # Create a model that maps the input image to the activations of the last conv layer
    grad_model = Model(inputs=[model.inputs], outputs=[model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predicted_class = np.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    # Compute gradients with respect to the last convolutional layer
    grads = tape.gradient(loss, conv_outputs)[0]
    if grads is None:
        raise ValueError("Gradients are None, possibly due to incorrect layer name or model input shape.")
    
    # Compute the pooled gradients
    pooled_grads = K.mean(grads, axis=(0, 1))

    conv_outputs = conv_outputs[0]

    # Multiply each channel in the feature map array by the average gradient for that channel
    for i in range(pooled_grads.shape[0]):  # Correct pooling shape
        conv_outputs[:, :, i] *= pooled_grads[i]

    # Compute the heatmap by averaging all the feature maps and apply ReLU
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # Resize heatmap to match the input image size
    heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))

    return heatmap

def overlay_gradcam_on_image(img_path, heatmap, alpha=0.4):
    # Load original image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert the heatmap to RGB and apply it to the original image
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Overlay the heatmap on the image
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    
    return superimposed_img


def trainPart(part):
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    image_dir = THIS_FOLDER + '/Dataset/'
    data = load_path(image_dir, part)
    labels = []
    filepaths = []

    for row in data:
        labels.append(row['label'])
        filepaths.append(row['image_path'])

    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')
    images = pd.concat([filepaths, labels], axis=1)

    train_df, test_df = train_test_split(images, train_size=0.9, shuffle=True, random_state=1)

    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,
                                                                      preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
                                                                      validation_split=0.2)
    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)

    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=64,
        shuffle=True,
        seed=42,
        subset='training'
    )

    val_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=64,
        shuffle=True,
        seed=42,
        subset='validation'
    )

    test_images = test_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False
    )

    pretrained_model = tf.keras.applications.resnet50.ResNet50(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg')

    pretrained_model.trainable = False

    inputs = pretrained_model.input
    x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
    x = tf.keras.layers.Dense(50, activation='relu')(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)

    print("-------Training " + part + "-------")
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(train_images, validation_data=val_images, epochs=25, callbacks=[callbacks])

    model.save(THIS_FOLDER + "/weights/ResNet50_" + part + "_frac.h5")
    results = model.evaluate(test_images, verbose=0)
    print(part + " Results:")
    print(results)
    print(f"Test Accuracy: {np.round(results[1] * 100, 2)}%")

    # Grad-CAM for test images
    gradcam_layer_name = "conv5_block3_out"  # Last convolutional layer in ResNet50
    for img_path in test_df['Filepath'][:5]:  # Generate Grad-CAM for first 5 test images
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = np.expand_dims(image.img_to_array(img), axis=0)
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

        # Get Grad-CAM heatmap and overlay it on the original image
        heatmap = get_gradcam(model, img_array, gradcam_layer_name)
        superimposed_img = overlay_gradcam_on_image(img_path, heatmap)

        # Save the image with the overlay
        output_path = os.path.join(THIS_FOLDER, f"./gradcam/{part}/gradcam_{os.path.basename(img_path)}")
        plt.imsave(output_path, superimposed_img)

    # Plot accuracy and loss
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    figAcc = plt.gcf()
    my_file = os.path.join(THIS_FOLDER, "./plots/FractureDetection/" + part + "/_Accuracy.jpeg")
    figAcc.savefig(my_file)
    plt.clf()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    figAcc = plt.gcf()
    my_file = os.path.join(THIS_FOLDER, "./plots/FractureDetection/" + part + "/_Loss.jpeg")
    figAcc.savefig(my_file)
    plt.clf()

# run the function and create model for each parts in the array
categories_parts = ["Elbow", "Hand", "Shoulder"]
for category in categories_parts:
    trainPart(category)