import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

def calculate_histogram(image):
    """Calculate the histogram for an image."""
    color = ('b', 'g', 'r')
    histogram = {}
    if len(image.shape) == 2:  # Grayscale
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        histogram["gray"] = hist
    else:
        for i, col in enumerate(color):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            histogram[col] = hist
    return histogram

def apply_filter(image, filter_type, **params):
    """Apply the selected filter to the image."""
    if filter_type == "Blur":
        return cv2.GaussianBlur(image, (params['kernel'], params['kernel']), 0)
    elif filter_type == "Grayscale":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif filter_type == "Thresholding":
        _, thresh = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), params['threshold'], 255, cv2.THRESH_BINARY)
        return thresh
    elif filter_type == "Erosion":
        kernel = np.ones((params['kernel'], params['kernel']), np.uint8)
        return cv2.erode(image, kernel, iterations=params['iterations'])
    elif filter_type == "Dilation":
        kernel = np.ones((params['kernel'], params['kernel']), np.uint8)
        return cv2.dilate(image, kernel, iterations=params['iterations'])
    elif filter_type == "Morphological Transformations":
        kernel = np.ones((params['kernel'], params['kernel']), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)  # No iterations here
    elif filter_type == "Rotation":
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, params['angle'], 1.0)
        return cv2.warpAffine(image, matrix, (w, h))
    elif filter_type == "Scaling":
        return cv2.resize(image, None, fx=params['scale'], fy=params['scale'], interpolation=cv2.INTER_LINEAR)
    elif filter_type == "Gaussian Blur":
        return cv2.GaussianBlur(image, (params['kernel'], params['kernel']), 0)
    elif filter_type == "Median Blur":
        return cv2.medianBlur(image, params['kernel'])
    elif filter_type == "Canny Edge Detection":
        return cv2.Canny(image, params['threshold1'], params['threshold2'])
    elif filter_type == "Invert Colors":
        return cv2.bitwise_not(image)
    else:
        return image

def plot_histogram(histogram, title):
    """Plot the histogram."""
    fig, ax = plt.subplots()
    for color, hist in histogram.items():
        ax.plot(hist, color=color if color != "gray" else "black")
    ax.set_title(title)
    ax.set_xlim([0, 256])
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

st.title("Advanced Image Processing Filters")

# Display filter descriptions
filter_descriptions = {
    "Blur": "Smoothens the image by averaging the pixels in the kernel area, reducing noise.",
    "Grayscale": "Converts the image to grayscale, where each pixel is represented by a single intensity value.",
    "Thresholding": "Converts the image to binary format, turning pixels above a threshold to white and below to black.",
    "Erosion": "Shrinks bright areas in the image, removing small noise and boundaries.",
    "Dilation": "Expands bright areas, adding pixels to the boundaries of objects.",
    "Morphological Transformations": "Detects edges or changes between eroded and dilated versions of the image (morphological gradient).",
    "Rotation": "Rotates the image by the specified angle.",
    "Scaling": "Resizes the image by a scale factor, making it larger or smaller.",
    "Gaussian Blur": "Applies a smoothing filter to the image using a Gaussian kernel, reducing detail and noise.",
    "Median Blur": "Reduces noise by replacing each pixel with the median of its neighboring pixels.",
    "Canny Edge Detection": "Detects edges in the image based on intensity changes between adjacent pixels.",
    "Invert Colors": "Inverts the colors of the image, making light areas dark and vice versa."
}

uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    image = np.array(image)

    # Filter selection
    filter_type = st.selectbox(
        "Select Filter",
        [
            "Blur",
            "Grayscale",
            "Thresholding",
            "Erosion",
            "Dilation",
            "Morphological Transformations",
            "Rotation",
            "Scaling",
            "Gaussian Blur",
            "Median Blur",
            "Canny Edge Detection",
            "Invert Colors"
        ]
    )

    # Display filter description
    st.write(f"**Description**: {filter_descriptions[filter_type]}")

    # Set dynamic filter parameters
    filter_params = {}
    if filter_type == "Blur":
        filter_params['kernel'] = st.slider("Kernel Size", min_value=1, max_value=21, step=2)
    elif filter_type == "Thresholding":
        filter_params['threshold'] = st.slider("Threshold Value", min_value=0, max_value=255, step=1)
    elif filter_type == "Erosion" or filter_type == "Dilation" or filter_type == "Morphological Transformations":
        filter_params['kernel'] = st.slider("Kernel Size", min_value=1, max_value=21, step=2)
        if filter_type == "Erosion" or filter_type == "Dilation":
            filter_params['iterations'] = st.slider("Iterations", min_value=1, max_value=5, step=1)
    elif filter_type == "Rotation":
        filter_params['angle'] = st.slider("Angle (degrees)", min_value=-180, max_value=180, step=1)
    elif filter_type == "Scaling":
        filter_params['scale'] = st.slider("Scale Factor", min_value=0.1, max_value=3.0, step=0.1)
    elif filter_type == "Gaussian Blur":
        filter_params['kernel'] = st.slider("Kernel Size", min_value=1, max_value=21, step=2)
    elif filter_type == "Median Blur":
        filter_params['kernel'] = st.slider("Kernel Size", min_value=1, max_value=21, step=2)
    elif filter_type == "Canny Edge Detection":
        filter_params['threshold1'] = st.slider("Threshold1", min_value=0, max_value=255, step=1)
        filter_params['threshold2'] = st.slider("Threshold2", min_value=0, max_value=255, step=1)
    elif filter_type == "Invert Colors":
        filter_params = {}

    filtered_image = apply_filter(image, filter_type, **filter_params)

    # Display Images Side by Side
    st.subheader("Image Comparison")
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Image", use_container_width=True)

    with col2:
        st.image(filtered_image, caption="Filtered Image", use_container_width=True)

    # Display Histograms Side by Side
    st.subheader("Histogram Comparison")
    col1, col2 = st.columns(2)

    with col1:
        original_histogram = calculate_histogram(image)
        plot_histogram(original_histogram, "Original Image Histogram")

    with col2:
        filtered_histogram = calculate_histogram(filtered_image)
        plot_histogram(filtered_histogram, "Filtered Image Histogram")

    # Convert filtered image to PIL format for download
    filtered_image_pil = Image.fromarray(filtered_image)

    # Create a download button for the filtered image
    buf = io.BytesIO()
    filtered_image_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="Download Filtered Image",
        data=byte_im,
        file_name="filtered_image.png",
        mime="image/png"
    )
