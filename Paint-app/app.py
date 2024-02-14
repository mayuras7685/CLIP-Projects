import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import clip


drawing_mode = st.sidebar.selectbox(
    "Drawing tool:",
    ("freedraw", "line", "rect", "circle", "transform", "polygon", "point"),
)
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
if drawing_mode == "point":
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
fill_color = st.sidebar.color_picker("Fill color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Prompt for similarity comparison
prompt = st.text_input("Enter prompt:", "A scenic landscape with mountains")

# Create a canvas component
canvas_result = st_canvas(
    fill_color= fill_color,  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    update_streamlit=realtime_update,
    height=500,
    width= 500,
    drawing_mode=drawing_mode,
    point_display_radius=point_display_radius if drawing_mode == "point" else 0,
    display_toolbar=st.sidebar.checkbox("Display toolbar", True),
    key="full_app",
)

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

# Do something interesting with the image data and paths

if canvas_result.image_data is not None and prompt is not None:
    data = st.image(canvas_result.image_data)
    print(type(data))


    


