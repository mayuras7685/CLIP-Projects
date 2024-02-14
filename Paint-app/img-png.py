import base64
import os
import re
import time
import uuid
from io import BytesIO
from pathlib import Path

import torch
import clip

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas



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

if "button_id" not in st.session_state:
   st.session_state["button_id"] = ""
if "color_to_label" not in st.session_state: 
   st.session_state["color_to_label"] = {}

st.markdown(
        """
    Realtime update is disabled for this demo. 
    Press the 'Download' button at the bottom of canvas to update exported image.
    """
    )
# Prompt for similarity comparison
prompt = st.text_input("Enter prompt:", "A scenic landscape with mountains")

try:
  Path("tmp/").mkdir()
except FileExistsError:
  pass

# Regular deletion of tmp files
# Hopefully callback makes this better
now = time.time()
N_HOURS_BEFORE_DELETION = 1
for f in Path("tmp/").glob("*.png"):
  st.write(f, os.stat(f).st_mtime, now)
  if os.stat(f).st_mtime < now - N_HOURS_BEFORE_DELETION * 3600:
    Path.unlink(f)


if st.session_state["button_id"] == "":
    st.session_state["button_id"] = re.sub(
        "\d+", "", str(uuid.uuid4()).replace("-", "")
    )


button_id = st.session_state["button_id"]
file_path = f"tmp/{button_id}.png"

custom_css = f""" 
    <style>
        #{button_id} {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background-color: rgb(255, 255, 255);
            color: rgb(38, 39, 48);
            padding: .25rem .75rem;
            position: relative;
            text-decoration: none;
            border-radius: 4px;
            border-width: 1px;
            border-style: solid;
            border-color: rgb(230, 234, 241);
            border-image: initial;
        }} 
        #{button_id}:hover {{
            border-color: rgb(246, 51, 102);
            color: rgb(246, 51, 102);
        }}
        #{button_id}:active {{
            box-shadow: none;
            background-color: rgb(246, 51, 102);
            color: white;
            }}
    </style> """

# Specify canvas parameters in application
drawing_mode = st.sidebar.selectbox(
            "Drawing tool:",
            ("freedraw", "line", "rect", "circle", "transform", "polygon", "point"),
        )
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
if drawing_mode == "point":
  point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)

stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
fill_color = st.sidebar.color_picker("Fill color hex: ", "#eee")
realtime_update = st.sidebar.checkbox("Update in realtime", True)


data = st_canvas(
   fill_color= fill_color, 
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            height=500,
            width=500,
            drawing_mode=drawing_mode,
            point_display_radius=point_display_radius if drawing_mode == "point" else 0,
            display_toolbar=st.sidebar.checkbox("Display toolbar", True),
   update_streamlit=False, key="png_export")

if data is not None and data.image_data is not None:
    img_data = data.image_data
    im = Image.fromarray(img_data.astype("uint8"), mode="RGBA")
    im.save(file_path, "png")
    buffered = BytesIO()
    im.save(buffered, format="png")
    img_data = buffered.getvalue()
    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(img_data.encode()).decode()
    except AttributeError:
        b64 = base64.b64encode(img_data).decode()
    dl_link = (
        custom_css
        + f'<a download="{file_path}" id="{button_id}" href="data:file/txbase64,{b64}">Export PNG</a><br></br>'
    )
    st.markdown(dl_link, unsafe_allow_html=True)
