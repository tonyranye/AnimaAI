# 🧠 Google Cloud Storage ML Project — Setup & Access Guide

This guide explains how anyone can access, run, and modify this machine learning project that uses images stored in **Google Cloud Storage (GCS)**.  
By following these steps, you’ll be able to authenticate, download a few sample images, and start training or experimenting locally — without having to repeat the one-time configuration that was originally required.

---

## 1️⃣ Overview

This project stores its dataset in a shared **Google Cloud Storage bucket**.  
The Python code connects directly to that bucket to retrieve images for machine learning experiments.

You will:
- Authenticate with Google Cloud  
- Download or view images from the shared bucket  
- Run `test.py`
- Modify the project on your own computer  

---

## 2️⃣ Prerequisites

Before starting, make sure you have:

- A **Google account** (the same one that will be granted bucket access)
- **Python 3.9 or higher**
- **pip** (Python package manager)
- The following Python packages installed:

```bash
pip install google-cloud-storage pillow matplotlib
```

Optional but recommended:
- The **Google Cloud SDK** (for authentication):  
  https://cloud.google.com/sdk/docs/install

---

## 3️⃣ Access to the Dataset

The project’s dataset is stored in:

```
Bucket name: animal-ai-images
Project ID:  poised-gateway-478017
Path:        images/
```

You need permission to access the bucket.  
You can get this in one of two ways:

---

### 🅰️ Option A — Direct Access via Google Account (Recommended)

1. Someone grants you access in the **Google Cloud Console**:  
   - Go to **Storage → Buckets → animal-ai-images → Permissions**  
   - Click **“Grant Access”**  
   - Enter your **Google email address**  
   - Assign the role: **Storage Object Viewer**  
   - Click **Save**

2. Once added, you automatically gain permission to read and download the dataset.

3. Authenticate locally (only once):

```bash
gcloud auth application-default login
```

---

### 🅱️ Option B — Service Account JSON Key

If direct access isn’t possible, someone can send you a JSON key file that grants access.

1. Save the file (for example):

```
C:\Users\<YourName>\Downloads\service-key.json
```

2. Set the environment variable so your code can use it:

**Windows (PowerShell):**
```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\Users\<YourName>\Downloads\service-key.json"
```

**macOS/Linux (bash):**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/Users/<YourName>/Downloads/service-key.json"
```

---

## 4️⃣ Configure the Project ID

You can set your project ID globally or pass it in the Python script.

**Option 1 — Environment Variable**
```bash
set GOOGLE_CLOUD_PROJECT=poised-gateway-478017        # Windows
export GOOGLE_CLOUD_PROJECT=poised-gateway-478017     # macOS/Linux
```

**Option 2 — Hardcoded in Python**
```python
client = storage.Client(project="poised-gateway-478017")
```

---

## 5️⃣ Running the Example Script

After access and dependencies are ready, test the setup:

```bash
python testdisplay.py
```

If it worked, you should get an image of a cat!

## 6️⃣ Troubleshooting

| Problem | Likely Cause | Solution |
|----------|---------------|-----------|
| `OSError: Project was not passed...` | No project ID found | Add `project="poised-gateway-478017"` in `storage.Client()` or set the environment variable. |
| `403 Forbidden` | You don’t have permission to the bucket | Ask someone to add your email as a **Storage Object Viewer**. |
| `google.auth.exceptions.DefaultCredentialsError` | No valid login or JSON key | Run `gcloud auth application-default login` or set `GOOGLE_APPLICATION_CREDENTIALS`. |
| No images downloaded | Wrong bucket name or prefix | Verify `BUCKET_NAME` and `PREFIX` values. |

---

## 7️⃣ Editing and Local Use

Once you can access the images:
- You can edit any script or notebook locally.
- Training and experimentation will run faster once images are cached on disk.
- No additional cloud credits are consumed unless you re-download data.

---

## 8️⃣ Optional: Uploading Results or New Files

If you’re granted **write access**, you can upload new files to the same bucket.

```python
blob = client.bucket("animal-ai-images").blob("images/new_image.jpg")
blob.upload_from_filename("local_image.jpg")
```

If you only have read access, uploading will be blocked — this is normal for most collaborators.

---

## 9️⃣ Summary

| Step | Action |
|------|--------|
| 1 | Install Python and dependencies |
| 2 | Get bucket access or a JSON key |
| 3 | Authenticate (`gcloud` or environment variable) |
| 4 | Set project ID |
| 5 | Run `python test_gcs.py` |
| 6 | See sample images downloaded and displayed |

---

## 🔒 Notes on Safety

Deleting or modifying anything in **Google Cloud Storage** only affects the bucket itself — it does **not** delete your personal Google Drive or Google account data.

---

## 👤 Project Info

**Project ID:** `poised-gateway-478017`  
**Bucket:** `animal-ai-images`  
**Dataset Path:** `images/`  

---

*End of README*
