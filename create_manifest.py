from google.cloud import storage
import csv

BUCKET = "animal-ai-images"
PREFIX = "poised-gateway-478017-a4/"  # root folder inside bucket
MANIFEST = "manifest.csv"

client = storage.Client()

bucket = client.bucket(BUCKET)
blobs = bucket.list_blobs(prefix=PREFIX)

with open(MANIFEST, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["gcs_path", "label"])
    
    for blob in blobs:
        if blob.name.endswith("/"):  # skip folder placeholders
            continue
        # label = the folder name (2nd part after prefix)
        label = blob.name[len(PREFIX):].split("/")[0]
        writer.writerow([f"gs://{BUCKET}/{blob.name}", label])

print(f"Manifest created: {MANIFEST}")
