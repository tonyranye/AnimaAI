from google.cloud import storage
from pathlib import Path

def sync_gcs_prefix(bucket_name, prefix, local_dir):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    for blob in client.list_blobs(bucket, prefix=prefix):
        rel = Path(blob.name[len(prefix):]).name if blob.name.endswith('/') else Path(blob.name[len(prefix):])
        if not rel:
            continue
        dest = local_dir / rel
        if not dest.exists() or dest.stat().st_size != blob.size:
            dest.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(dest.as_posix())
    return local_dir

# Usage:
# cache_dir = sync_gcs_prefix("my-bucket", "images/", "./data_cache/images")
# Now point your PyTorch/TensorFlow dataloader to ./data_cache/images
