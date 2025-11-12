from google.cloud import storage
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

client = storage.Client(project="poised-gateway-478017")  # use your project ID
bucket = client.bucket("animal-ai-images")

blob = bucket.blob("poised-gateway-478017-a4/cats/0_0001.jpg")  # replace with a valid path in your bucket
img_bytes = blob.download_as_bytes()

img = Image.open(BytesIO(img_bytes))
plt.imshow(img)
plt.axis('off')
plt.show()
