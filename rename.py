from pathlib import Path

# Base "images" directory (relative to where you run this script)
BASE_DIR = Path("images")

if not BASE_DIR.is_dir():
    print(f"Directory '{BASE_DIR}' not found!")
    exit(1)

for folder in BASE_DIR.iterdir():
    if not folder.is_dir():
        continue

    orig_name = folder.name

    # Expect something like "001.Black_Footed_Albatross"
    # Remove the first 4 characters: "001." -> "Black_Footed_Albatross"
    if len(orig_name) > 4 and orig_name[3] == ".":
        clean_name = orig_name[4:]
    else:
        clean_name = orig_name  # fallback: leave name as-is

    # Rename folder if needed
    target_folder = folder.with_name(clean_name)
    if folder != target_folder:
        print(f"Renaming folder: {folder.name} -> {target_folder.name}")
        folder.rename(target_folder)
        folder = target_folder  # update reference

    # Get all files in this folder
    files = sorted(p for p in folder.iterdir() if p.is_file())

    # Rename files to 001.ext, 002.ext, ...
    for idx, file_path in enumerate(files, start=1):
        new_name = f"{idx:03d}{file_path.suffix.lower()}"
        new_path = folder / new_name
        print(f"  {file_path.name} -> {new_name}")
        file_path.rename(new_path)

print("Renaming complete.")
