import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("Data/dog_Cat.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["a dog", "a cat", "a cow"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# Convert the NumPy array to a Python list
probs_list = probs.tolist()

# Access the elements and format them
formatted_probs = [f"{100 * prob:.2f}%" for prob in probs_list[0]]

# Print the formatted probabilities
print(f"Label probs: {formatted_probs}")


