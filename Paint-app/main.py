import torch
import clip
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import base64
import io

# Function to compute similarity leaderboard
def similarity_score(image_data, prompt):
    # Load the CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Preprocess the drawn image
    image = Image.fromarray(image_data.astype("uint8"), "RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    # Encode prompt
    text = clip.tokenize([prompt]).to(device)

    # Calculate similarity score
    image_feature = model.encode_image(image_tensor)
    text_feature = model.encode_text(text)
    similarity_score = (image_feature @ text_feature.T).item()

    return similarity_score


# Streamlit App
st.title("Image Similarity")

# Drawing and configuration options
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:",
    ("freedraw", "line", "rect", "circle", "transform", "polygon", "point"),
)
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
if drawing_mode == "point":
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    update_streamlit=realtime_update,
    height=500,
    width=500,
    drawing_mode=drawing_mode,
    point_display_radius=point_display_radius if drawing_mode == "point" else 0,
    display_toolbar=st.sidebar.checkbox("Display toolbar", True),
    key="full_app",
)

# Prompt for similarity comparison
prompt = st.text_input("Enter prompt:", "A scenic landscape with mountains")

# Display the drawn image
if canvas_result.image_data is not None:
    st.image(canvas_result.image_data, use_column_width=True, caption="Drawn Image")


# # Compute similarity score and display
# similarity = similarity_score(canvas_result.image_data, prompt)
# st.write(f"Similarity Score with Prompt: {similarity:.4f}")
