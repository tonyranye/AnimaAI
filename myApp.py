import torch
from torchvision import models, transforms
from PIL import Image
import gradio as gr

# --- Device ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load checkpoint saved by your training script ---
checkpoint = torch.load("animal_model_local.pth", map_location=DEVICE)
class_names = checkpoint["class_names"]           # list of raw class labels (folder names)
label_to_idx = checkpoint["label_to_idx"]

# --- Rebuild the same model architecture ---
model = models.resnet18(weights=None)             # weights are loaded from the checkpoint
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

# --- Pretty display names for classes (you can expand this anytime) ---
PRETTY_LABELS = {
    # generic classes
    "cats": "Cat",
    "dogs": "Dog",
    "elephant": "Elephant",
    "butterfly": "Butterfly",
    "Lions":"Loin",
    "Cheetahs":"Cheetah",

    # cat breeds (folders under Animals/cat-breeds/)
    "Bengal": "Bengal cat",
    "Birman": "Birman cat",
    "Bombay": "Bombay cat",
    "Russian Blue": "Russian Blue cat",
    "Siamese": "Siamese cat",
    "Sphynx": "Sphynx cat",

}

def pretty_name(raw_label: str) -> str:
    """Map raw folder name -> nice display name."""
    return PRETTY_LABELS.get(raw_label, raw_label)

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
    returns: dict {pretty_class_name: probability} for Gradio to display nicely
    """
    img = preprocess(image).unsqueeze(0).to(DEVICE)   # [1, 3, 224, 224]

    with torch.no_grad():
        outputs = model(img)                          # logits
        probs = torch.softmax(outputs, dim=1)[0]      # [num_classes]

    # Map to {pretty_class_name: prob}
    result = {
        pretty_name(class_names[i]): float(probs[i])
        for i in range(len(class_names))
    }
    return result

# --- Gradio UI with an upload button ---
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an animal image"),
    outputs=gr.Label(num_top_classes=3, label="Predicted breed / species"),
    title="AnimAI – Animal & Breed Classifier",
    description=(
        "Upload an image of a cat, dog, or other animal. "
        "The model predicts both the animal and (for supported cats) the breed."
    ),
)

if __name__ == "__main__":
    demo.launch()
