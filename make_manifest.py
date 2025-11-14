# step2_make_manifest.py

import os
from google.cloud import storage

# --- CONFIG ---  (change only if your project/bucket name changes)
PROJECT_ID = "poised-gateway-478017-a4"
BUCKET_NAME = "animal-ai-images"
GCS_PREFIX = ""                 # top-level folders = labels (cats/, dogs/, ...)
LOCAL_CACHE_DIR = "images_cache"
MAX_IMAGES = 2000               # cap while testing; raise later if you want


def build_manifest():
    """
    Downloads images from GCS (if not already cached locally) and
    returns two lists: image_paths and labels.
    """
    os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)

    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)

    image_paths = []
    labels = []

    print("Listing blobs from GCS...")
    for blob in client.list_blobs(bucket, prefix=GCS_PREFIX):
        # Expect: "<label>/<filename>"
        parts = blob.name.split("/")

        # Skip 'directory' placeholders or weird names
        if len(parts) < 2 or parts[-1] == "":
            continue

        label = parts[0]         # e.g. "cats"
        filename = parts[-1]     # e.g. "img001.jpg"

        # Only use typical image extensions
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        local_path = os.path.join(LOCAL_CACHE_DIR, f"{label}_{filename}")

        if not os.path.exists(local_path):
            blob.download_to_filename(local_path)
            print("Downloaded:", local_path)

        image_paths.append(local_path)
        labels.append(label)

        if len(image_paths) >= MAX_IMAGES:
            break

    print("Total images:", len(image_paths))
    print("Total labels:", len(labels))

    # show a few examples
    for i in range(min(5, len(image_paths))):
        print(" ", image_paths[i], "->", labels[i])

    return image_paths, labels


if __name__ == "__main__":
    build_manifest()
