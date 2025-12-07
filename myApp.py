import os
import torch
from torchvision import models, transforms
from PIL import Image
import gradio as gr

# --- Base directories ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "Models")

print("BASE_DIR:", BASE_DIR)
print("MODEL_DIR:", MODEL_DIR)
print("Files in MODEL_DIR:", os.listdir(MODEL_DIR))

# ----------------- Device -----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------- Loader helper -----------------

def load_resnet18_from_checkpoint(ckpt_path: str):
    """
    Load a ResNet18 from one of your training checkpoints.
    Expects keys: 'class_names', 'label_to_idx', 'model_state_dict'.
    """
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    class_names = ckpt["class_names"]
    label_to_idx = ckpt.get(
        "label_to_idx",
        {name: i for i, name in enumerate(class_names)}
    )

    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, len(class_names))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, class_names, label_to_idx


def safe_load(path):
    """Load a checkpoint if it exists, otherwise return (None, None, None)."""
    if os.path.exists(path):
        return load_resnet18_from_checkpoint(path)
    print(f"[WARN] Specialist checkpoint not found: {path}")
    return None, None, None


# ----------------- Load all models -----------------

# router (coarse animal classifier)
ROUTER_CKPT = os.path.join(MODEL_DIR, "animal_model_local.pth")
router_model, router_classes, _ = load_resnet18_from_checkpoint(ROUTER_CKPT)

# specialist models
cats_dogs_model, cats_dogs_classes, _ = safe_load(
    os.path.join(MODEL_DIR, "cats-dogs_model_local.pth")
)
cat_breed_model, cat_breed_classes, _ = safe_load(
    os.path.join(MODEL_DIR, "cat-breed_model_local.pth")
)
dog_breed_model, dog_breed_classes, _ = safe_load(
    os.path.join(MODEL_DIR, "dog-breed_model_local.pth")
)
butterfly_model, butterfly_classes, _ = safe_load(
    os.path.join(MODEL_DIR, "butterfly_model_local.pth")
)
birds_model, birds_classes, _ = safe_load(
    os.path.join(MODEL_DIR, "bird_breed_model_local.pth")
)
birds_butter_model, birds_butter_classes, _ = safe_load(
    os.path.join(MODEL_DIR, "birds-butterflies_model_local.pth")
)
feline_model, feline_classes, _ = safe_load(
    os.path.join(MODEL_DIR, "feline_model_local.pth")
)
snake_gecko_chameleon_model, snake_gecko_chameleon_classes, _ = safe_load(
    os.path.join(MODEL_DIR, "snake_gecko_chameleon_local.pth")
)
birds_chemeleon_model, birds_chemeleon_classes, _ = safe_load(
    os.path.join(MODEL_DIR, "bird_chameleon_model_local.pth")
)




# ----------------- Pretty labels -----------------

PRETTY_LABELS = {
    # router-level classes (folder names)
    "birds": "Bird",
    "butterfly": "Butterfly",
    "cats": "Cat",
    "dogs": "Dog",
    "Cheetahs": "Cheetah",
    "Crocodile-Alligator": "Crocodile / Alligator",
    "elephant": "Elephant",
    "Frog": "Frog",
    "Gecko": "Gecko",
    "horse": "Horse",
    "snake": "Snake",
    "spider": "Spider",
    "wolves": "Wolf",
}


def pretty_name(raw: str) -> str:
    return PRETTY_LABELS.get(raw, raw.replace("_", " ").title())


# ----------------- Preprocessing -----------------

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ----------------- Small helpers -----------------

def run_model_to_dict(model, class_names, img_tensor):
    """Run model and return {raw_label: prob}."""
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)[0]

    return {
        class_names[i]: float(probs[i])
        for i in range(len(class_names))
    }


def top_k_from_dict(scores, k=2):
    """Return list of (label, prob) sorted desc."""
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]


def pretty_scores(scores_dict):
    """Map raw labels -> pretty display labels."""
    return {pretty_name(lbl): prob for lbl, prob in scores_dict.items()}


# ----------------- Main hierarchical predict() -----------------

def predict(image: Image.Image):
    x = preprocess(image).unsqueeze(0).to(DEVICE)

    # 1) Router: coarse prediction across all 14 folders
    router_scores = run_model_to_dict(router_model, router_classes, x)
    router_top2 = top_k_from_dict(router_scores, k=2)
    router_top1_label, router_top1_prob = router_top2[0]
    router_top_labels = [lbl for lbl, _ in router_top2]

    # ----- Special case: cat vs dog ambiguity -> cats-dogs specialist -----
    if ("cats" in router_top_labels and "dogs" in router_top_labels
            and cats_dogs_model is not None):
        cd_scores = run_model_to_dict(cats_dogs_model, cats_dogs_classes, x)
        cd_top1_label, cd_top1_prob = top_k_from_dict(cd_scores, k=1)[0]

        # If that says "cats" -> cat breed model
        if cd_top1_label == "cats" and cat_breed_model is not None:
            cat_scores = run_model_to_dict(cat_breed_model, cat_breed_classes, x)
            return pretty_scores(cat_scores)

        # If that says "dogs" -> dog breed model
        if cd_top1_label == "dogs" and dog_breed_model is not None:
            dog_scores = run_model_to_dict(dog_breed_model, dog_breed_classes, x)
            return pretty_scores(dog_scores)

        # otherwise just show cats-dogs output
        return pretty_scores(cd_scores)

    # ----- Birds vs Butterfly ambiguity -> birds-butterflies specialist -----
    if ("birds" in router_top_labels and "butterfly" in router_top_labels
            and birds_butter_model is not None):
        bb_scores = run_model_to_dict(birds_butter_model, birds_butter_classes, x)
        return pretty_scores(bb_scores)

    # ----- Otherwise, route by router top-1 label --------------------------
    label = router_top1_label

    # cats -> cat breeds if available
    if label == "cats":
        if cat_breed_model is not None:
            cat_scores = run_model_to_dict(cat_breed_model, cat_breed_classes, x)
            return pretty_scores(cat_scores)
        # fallback to cats-dogs if cat-breed missing
        if cats_dogs_model is not None:
            cd_scores = run_model_to_dict(cats_dogs_model, cats_dogs_classes, x)
            return pretty_scores(cd_scores)

    # dogs -> dog breeds if available
    if label == "dogs":
        if dog_breed_model is not None:
            dog_scores = run_model_to_dict(dog_breed_model, dog_breed_classes, x)
            return pretty_scores(dog_scores)
        if cats_dogs_model is not None:
            cd_scores = run_model_to_dict(cats_dogs_model, cats_dogs_classes, x)
            return pretty_scores(cd_scores)
    # butterfly -> butterfly breeds if available
    if label == "butterfly":
        if butterfly_model is not None:
            butterfly_scores = run_model_to_dict(butterfly_model, butterfly_classes, x)
            return pretty_scores(butterfly_scores)
        if birds_butter_model is not None:
            bb_scores = run_model_to_dict(birds_butter_model, birds_butter_classes, x)
            return pretty_scores(bb_scores)
    # birds -> bird breeds if available
    if label == "birds":
        if birds_model is not None:
            bird_scores = run_model_to_dict(birds_model, birds_classes, x)
            return pretty_scores(bird_scores)
        if birds_butter_model is not None:
            bb_scores = run_model_to_dict(birds_butter_model, birds_butter_classes, x)
            return pretty_scores(bb_scores)

    # birds / butterfly -> specialist
    if label in {"birds", "butterfly"} and birds_butter_model is not None:
        bb_scores = run_model_to_dict(birds_butter_model, birds_butter_classes, x)
        return pretty_scores(bb_scores)

    # cheetahs / other big-feline style labels -> feline specialist
    if label in {"Cheetahs","cats"} and feline_model is not None:
        fe_scores = run_model_to_dict(feline_model, feline_classes, x)
        return pretty_scores(fe_scores)

    #  snake / gecko / chameleon -> specialist
    if label in {"snake","Gecko","Chameleon"} and feline_model is not None:
        sgc_scores = run_model_to_dict(snake_gecko_chameleon_model, snake_gecko_chameleon_classes, x)
        return pretty_scores(sgc_scores)
    
    # birds / chameleon -> specialist
    if label in {"birds", "Chameleon"} and birds_chemeleon_model is not None:
        bc_scores = run_model_to_dict(birds_chemeleon_model, birds_chemeleon_classes, x)
        return pretty_scores(bc_scores)


    # For all other animals (elephant, snake, spider, horse, etc.)
    # we only have the router, so just show its probabilities.
    return pretty_scores(router_scores)


# ----------------- Gradio UI -----------------

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an animal image"),
    outputs=gr.Label(num_top_classes=3, label="Predicted animal / breed"),
    title="AnimAI – Hierarchical Animal Classifier",
    description=(
        "Step 1: a router model guesses the general animal type.\n"
        "Step 2: for cats/dogs/birds/butterflies/cheetahs, "
        "specialist models refine the prediction."
    ),
)

if __name__ == "__main__":
    demo.launch()
