import torch
from torchvision import models, transforms
from PIL import Image
import gradio as gr


'''
List of all 74 classes trained in the butterfly-BREED model:
'''

[
    'ADONIS',
    'AFRICAN GIANT SWALLOWTAIL',
    'AMERICAN SNOOT',
    'AN 88',
    'APPOLLO',
    'ATALA',
    'BANDED ORANGE HELICONIAN',
    'BANDED PEACOCK',
    'BECKERS WHITE',
    'BLACK HAIRSTREAK',
    'BLUE MORPHO',
    'BLUE SPOTTED CROW',
    'BROWN SIPROETA',
    'CABBAGE WHITE',
    'CAIRNS BIRDWING',
    'CHECQUERED SKIPPER',
    'CLAROON',
    'CLEOPATRA',
    'CLODIUS PARNASSIAN',
    'CLOUDED SULPHUR',
    'COMMON BANDED AWL',
    'COMMON WOOD-NYMPH',
    'COPPER TAIL',
    'CRECENT',
    'CRIMSON PATCH',
    'DANAID EGGFLY',
    'EASTERN COMA',
    'EASTERN DAPPLE WHITE',
    'EASTERN PINE ELFIN',
    'ELBOWED PIERROT',
    'EMERALD SWALLOWTAIL',
    'GOLD BANDED',
    'GREAT EGGFLY',
    'GREAT JAY',
    'GREEN CELLED CATTLEHEART',
    'GREEN HAIRSTREAK',
    'GREY HAIRSTREAK',
    'INDRA SWALLOW',
    'IPHICLUS SISTER',
    'JULIA',
    'LARGE MARBLE',
    'MALACHITE',
    'MANGROVE SKIPPER',
    'METALMARK',
    'MONARCH',
    'MOURNING CLOAK',
    'ORANGE OAKLEAF',
    'ORANGE TIP',
    'ORCHARD SWALLOW',
    'PAINTED LADY',
    'PAPER KITE',
    'PEACOCK',
    'PINE WHITE',
    'PINK ROSETTE',
    'PIPEVINE SWALLOW',
    'POPINJAY',
    'PURPLE HAIRSTREAK',
    'RED ADMIRAL',
    'RED CRACKER',
    'RED POSTMAN',
    'RED SPOTTED PURPLE',
    'SCARCE SWALLOW',
    'SILVER SPOT SKIPPER',
    'SLEEPY ORANGE',
    'SOOTYWING',
    'SOUTHERN DOGFACE',
    'STRAITED QUEEN',
    'TROPICAL LEAFWING',
    'TWO BARRED FLASHER',
    'ULYSES',
    'VICEROY',
    'WOOD SATYR',
    'YELLOW SWALLOW TAIL',
    'ZEBRA LONG WING'
]

# --- Device ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ---------------------------
# Helpers to rebuild a ResNet18
# ---------------------------
def build_resnet18(num_classes: int):
    model = models.resnet18(weights=None)  # architecture only
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, num_classes)
    return model

# ---------------------------
# Load GENERAL animal model
#   (cats, dogs, elephant, butterfly, etc.)
# ---------------------------
gen_checkpoint = torch.load("animal_model_local.pth", map_location=DEVICE)
gen_class_names = gen_checkpoint["class_names"]      # e.g. ["butterfly", "cats", "dogs", "elephant"]

gen_model = build_resnet18(num_classes=len(gen_class_names))
gen_model.load_state_dict(gen_checkpoint["model_state_dict"])
gen_model.to(DEVICE)
gen_model.eval()

# ---------------------------
# Load butterfly-BREED model
#   (70+ butterfly breeds)
# ---------------------------
breed_checkpoint = torch.load("butterfly_model_local.pth", map_location=DEVICE)
breed_class_names = breed_checkpoint["class_names"]  # e.g. ["husky", "pug", "german_shepherd", ...]

breed_model = build_resnet18(num_classes=len(breed_class_names))
breed_model.load_state_dict(breed_checkpoint["model_state_dict"])
breed_model.to(DEVICE)
breed_model.eval()

# If your general model uses "butterfly" vs "butterfliess", adjust this set
BUTTERFLY_LABEL = {"butterfly", "butterflies"}

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
    returns: dict {class_name: probability}
    - If general model says NOT butterfly: return general animal probs
    - If general model says butterfly: return butterfly class probs only
    """
    img = preprocess(image).unsqueeze(0).to(DEVICE)   # [1, 3, 224, 224]

    with torch.no_grad():
        # ----- Stage 1: General animal classifier -----
        gen_outputs = gen_model(img)                  # logits
        gen_probs = torch.softmax(gen_outputs, dim=1)[0]
        gen_idx = int(torch.argmax(gen_probs))
        general_label = gen_class_names[gen_idx]

    # If it's not a butterfly → just return general animal probabilities
    if general_label not in BUTTERFLY_LABEL:
        result = {gen_class_names[i]: float(gen_probs[i]) for i in range(len(gen_class_names))}
        return result

    # ----- Stage 2: Dog-breed classifier -----
    with torch.no_grad():
        breed_outputs = breed_model(img)
        breed_probs = torch.softmax(breed_outputs, dim=1)[0]

    # Map to {breed_name: prob}
    # (we're only returning butterfly-class solutions as you requested)
    result = {breed_class_names[i]: float(breed_probs[i]) for i in range(len(breed_class_names))}
    return result

# --- Gradio UI with an upload button ---
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an animal image"),
    outputs=gr.Label(num_top_classes=3, label="Predicted class"),
    title="AnimAI – Hierarchical Animal & Butterfly class Classifier",
    description=(
        "Step 1: General model predicts cat/dog/elephant/butterfly, etc. "
        "If it's a dog, a second model predicts the specific dog breed."
    ),
)

if __name__ == "__main__":
    demo.launch(share=True) 