import streamlit as st
import cv2
import easyocr
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Number Plate Recognition",
    page_icon="🚗"
)

st.title("🚗 Number Plate Recognition System")

st.write(
    "Upload a vehicle image to recognize its number plate."
)

@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'])

reader = load_reader()

uploaded_file = st.file_uploader(
    "Upload Vehicle Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    img_array = np.array(image)

    st.subheader("Original Image")
    st.image(img_array, use_container_width=True)

    if st.button("Detect Number Plate"):

        gray = cv2.cvtColor(
            img_array,
            cv2.COLOR_RGB2GRAY
        )

        results = reader.readtext(gray)

        detected = False

        output = img_array.copy()

        st.subheader("Recognition Results")

        for result in results:

            bbox = result[0]
            text = result[1]
            confidence = result[2]

            if confidence > 0.30:

                detected = True

                points = np.array(
                    bbox,
                    dtype=np.int32
                )

                cv2.polylines(
                    output,
                    [points],
                    True,
                    (0, 255, 0),
                    3
                )

                st.success(
                    f"Detected Text: {text}"
                )

                st.write(
                    f"Confidence: {confidence:.2f}"
                )

        if detected:

            st.image(
                output,
                caption="Detected Text Regions",
                use_container_width=True
            )

        else:

            st.warning(
                "No readable text detected. "
                "Try another clear vehicle image."
            )

st.sidebar.title("About Project")

st.sidebar.info(
    "Python Computer Vision Project "
    "using OpenCV and EasyOCR."
)
