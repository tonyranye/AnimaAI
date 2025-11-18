<!-- .github/copilot-instructions.md for AnimaAI -->
# Copilot / AI Agent Instructions — AnimaAI

This file gives targeted, actionable guidance so an AI coding assistant can be immediately productive working in this repository.

1) Big picture
- Purpose: trains an image classifier from local folders of animal photos and provides a simple Streamlit front-end.
- Key components:
  - `train_animals.py`: main training script. Scans `Animals/` for class subfolders, builds datasets, trains a ResNet18, and saves `animal_model_local.pth`.
  - `animal_dataset.py`: lightweight `Dataset` wrapper. Constructor: `(image_paths, labels, label_to_idx, transform=None)` and `__getitem__` returns `(image_tensor, label_idx)`.
  - `myApp.py`: Streamlit app skeleton that imports `mymodel` (expected to provide inference helpers). `mymodel.py` is not present — look for or implement inference/load utilities before running the app.

2) Data layout and conventions
- Images live under `Animals/<class_name>/*.jpg|.jpeg|.png`. Class names are inferred from subfolder names and are sorted when building `label_to_idx`.
- Only files matching `(.jpg, .jpeg, .png)` are considered (see `list_image_paths_and_labels` in `train_animals.py`).
- The training code normalizes images with ImageNet mean/std and resizes to `224x224` (compatible with pretrained ResNet18).

3) Runtime / workflows
- Install dependencies (as in `README.md`): `torch`, `torchvision`, `Pillow`, `matplotlib`. Streamlit required to run `myApp.py` UI: `pip install streamlit`.
- Train locally:
  - Command: `python train_animals.py`
  - Output: `animal_model_local.pth` (a checkpoint dict containing `model_state_dict`, `label_to_idx`, `class_names`).
- Run UI:
  - Command: `streamlit run myApp.py`
  - Note: `myApp.py` imports `mymodel as m`; ensure that module exists and can load `animal_model_local.pth`.

4) Important code patterns & examples
- Label mapping: created in `create_dataloaders()` as `label_to_idx = {name: i for i, name in enumerate(class_names)}`. Use the saved `label_to_idx` to decode predictions.
- Loading saved checkpoint example (how to load programmatically):
  ```py
  import torch
  ckpt = torch.load('animal_model_local.pth', map_location='cpu')
  model.load_state_dict(ckpt['model_state_dict'])
  label_to_idx = ckpt['label_to_idx']
  class_names = ckpt['class_names']
  ```
- Dataset API: `AnimalDataset` returns `(img_tensor, label_idx)`; DataLoader wraps it with `batch_size` and `shuffle`.

5) Hyperparameters & small gotchas
- Defaults in `train_animals.py`: `BATCH_SIZE=32`, `NUM_EPOCHS=5`, `LR=1e-4`, `VAL_SPLIT=0.2`, `num_workers=2` in DataLoader.
- Platform note (Windows): child-process DataLoader `num_workers>0` can cause issues; when debugging on Windows set `num_workers=0`.
- Device: script auto-detects `cuda` via `torch.cuda.is_available()`; make sure drivers/CUDA setup is correct before expecting GPU speedups.

6) What to change or add when requested
- If asked to add inference utilities: add `mymodel.py` that exposes `load_model(path) -> model, class_names` and `predict(model, pil_image) -> (class_name, prob)`.
- If asked to add tests: write small unit tests for `list_image_paths_and_labels()` and `AnimalDataset.__getitem__()` using a temporary `Animals/` fixture with a couple of tiny images.

7) Where to look for examples in repo
- `train_animals.py`: overall flow and constants — always the primary reference for training behavior.
- `animal_dataset.py`: dataset shape and expectations for image loading and returned labels.
- `README.md`: user-facing run instructions and dependency list.

8) Questions an agent should ask the user before making changes
- Do you want the Streamlit app to be implemented now? If yes, should inference use CPU-only or target GPU?
- Should we add a `mymodel.py` inference helper and example UI integration in `myApp.py`?

If anything here is unclear or you want more/less detail, tell me which area to expand (training, inference, or Streamlit UI). 
