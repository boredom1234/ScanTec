import os
import streamlit as st
from PIL import Image
from predictions import predict

# Global variables
project_folder = os.path.dirname(os.path.abspath(__file__))
folder_path = project_folder + '/images/'

# Streamlit App
def main():
    st.title("Bone Fracture Detection")
    
    # Header
    st.image(os.path.join(folder_path, "info.png"), width=40)
    st.header("Bone Fracture Detection System")

    # Info Text
    st.write(
        """
        Bone fracture detection system, upload an x-ray image for fracture detection.
        """
    )

    # Upload Image
    uploaded_file = st.file_uploader("Upload an X-ray Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # Display the uploaded image
        img = Image.open(uploaded_file)
        img_resized = img.resize((int(256 / img.height * img.width), 256))
        st.image(img_resized, caption="Uploaded Image", use_column_width=False)

        # Prediction button
        if st.button("Predict"):
            # Simulating the predict function
            bone_type_result = predict(uploaded_file)
            result = predict(uploaded_file, bone_type_result)

            # Display result
            if result == 'fractured':
                st.error("Result: Fractured")
            else:
                st.success("Result: Normal")

            # Bone Type Result
            bone_type_result = predict(uploaded_file, "Parts")
            st.info(f"Type: {bone_type_result}")
    
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
