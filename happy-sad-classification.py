import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model with error handling
try:
    model = tf.keras.models.load_model('imageclassifier.h5')
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Define the Streamlit app
def app():
    # Set the app title and page layout
    st.set_page_config(
        page_title="Emotion Classifier",
        page_icon="😊",
        layout="wide"
    )

    # Set the app title with improved styling
    st.title('Happy 😄 or Sad 🥺 Classifier')
    
    # Add a file uploader widget with centered layout
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    
    # Add some space for a cleaner layout
    st.markdown("---")

    # Check if an image has been uploaded
    if uploaded_file is not None:
        # Read the image data
        image = Image.open(uploaded_file)

        # Resize the image to match the expected input shape of the model
        image = image.resize((256, 256))

        # Convert the image to a numpy array
        image_array = np.array(image)

        # Normalize the pixel values
        image_array = image_array / 255.0

        # Add a batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        # Make predictions on the image with error handling
        try:
            predictions = model.predict(image_array)
            threshold = 0.5
            predicted_class_name = 'sad🥺' if predictions[0, 0] >= threshold else 'happy😄'
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.stop()

        # Display the image and the predicted class name in a centered layout
        st.image(image, caption='', use_column_width=True)

        # Add some space for a cleaner layout
        st.markdown("---")

        # Display the predicted class name with improved styling
        st.markdown(f'## Prediction: {predicted_class_name.capitalize()}')

        # Display confidence score
        confidence_score = predictions[0, 0] if predicted_class_name == 'sad🥺' else 1 - predictions[0, 0]
        st.write(f"Confidence Score: {confidence_score:.2%}")

# Run the app
if __name__ == "__main__":
    app()
