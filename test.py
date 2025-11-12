from google.cloud import storage
from google.cloud import storage

client = storage.Client(project="poised-gateway-478017-a4")
bucket = client.bucket("animal-ai-images")

for i, blob in enumerate(client.list_blobs(bucket, prefix="images/")):
    if i >= 50:  # only first 50 images
        break
    blob.download_to_filename(f"test_images/{blob.name.split('/')[-1]}")
