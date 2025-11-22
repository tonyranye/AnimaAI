from google.cloud import storage
client = storage.Client()
print("Success! Authenticated as:", client.project)
