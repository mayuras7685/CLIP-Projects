import torch
import clip
from PIL import Image

def similarity_leaderboard(images, prompt):
    # Load the CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Load and preprocess images
    image_tensors = [preprocess(Image.open(image_path)).unsqueeze(0).to(device) for image_path in images]

    # Encode prompt
    text = clip.tokenize([prompt]).to(device)

    # Calculate similarity scores
    image_features = model.encode_image(torch.cat(image_tensors))
    text_features = model.encode_text(text)
    similarity_scores = (image_features @ text_features.T).squeeze(1)

    # Rank images based on similarity scores
    ranked_indices = similarity_scores.argsort(descending=True)
    ranked_images = [{"image_path": images[idx], "similarity_score": similarity_scores[idx].item()} for idx in ranked_indices]

    return ranked_images

# Example usage:
images = ["Data/test1.png", "Data/test2.png", "Data/test3.png"]  # Paths to your images
prompt = "Upside down dinosaur"

ranked_leaderboard = similarity_leaderboard(images, prompt)
print("Similarity Leaderboard:")
for rank, entry in enumerate(ranked_leaderboard, start=1):
    print(f"Rank {rank}: Image: {entry['image_path']}, Similarity Score: {entry['similarity_score']:.4f}")
