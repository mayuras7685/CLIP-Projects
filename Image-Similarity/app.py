import clip
import torch
from PIL import Image 
from itertools import chain

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, transform = clip.load("ViT-B/32", device=device)

# Prepare input data
image_paths = ["Data/test1.png", "Data/test2.png", "Data/test3.png"]
images = torch.stack([transform(Image.open(image_path)).to(device) for image_path in image_paths])

text = clip.tokenize(["Upside down Dinosaur"]).to(device)

# Compute embeddings
image_features = model.encode_image(images)
text_features = model.encode_text(text)

# Calculate similarity scores and convert to percentages
similarity_scores = (text_features @ image_features.T).softmax(dim=1)
percentage_list = [[value.item() * 100 for value in row] for row in similarity_scores]
scores = list(chain(*percentage_list))

# Combine image names with their respective scores
result = zip(image_paths, scores)

# Display the result
for image_name, score in result:
    formatted_result = f"{image_name}: {score:.4f}"
    print(formatted_result)
