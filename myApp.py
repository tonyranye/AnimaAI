import torch
from torchvision import models, transforms
from PIL import Image
import gradio as gr

# --- Device ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load checkpoint saved by your training script ---
checkpoint = torch.load("animal_model_local.pth", map_location=DEVICE)
class_names = checkpoint["class_names"]           # e.g. ["cat", "dog", "spider"]
label_to_idx = checkpoint["label_to_idx"]         # e.g. {"cat": 0, "dog": 1, "spider": 2}

# --- Rebuild the same model architecture ---
model = models.resnet18(weights=None)             # start empty
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

# --- Same preprocessing as your val_transform ---
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def predict(image: Image.Image):
    """
    image: PIL image from Gradio upload
    returns: dict {class_name: probability} for Gradio to display nicely
    """
    img = preprocess(image).unsqueeze(0).to(DEVICE)   # [1, 3, 224, 224]

    with torch.no_grad():
        outputs = model(img)                          # logits
        probs = torch.softmax(outputs, dim=1)[0]      # [num_classes]

    # Map to {class_name: prob}
    result = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    return result

# --- Gradio UI with an upload button ---
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an animal image"),
    outputs=gr.Label(num_top_classes=3, label="Predicted class"),
    title="AnimAI – Animal Classifier",
    description="Upload a cat/dog/spider image and see what the model predicts.",
)

print(torch.cuda.is_available())
print(DEVICE)

if __name__ == "__main__":
    demo.launch(share = True)
    