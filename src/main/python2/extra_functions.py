import os


def list_image_dir(image_base_path):
    files = os.listdir(image_base_path)
    if files[0] == 'Thumbs.db':
        files = files[1:]
    files = [elem for elem in files if os.path.isfile(os.path.join(image_base_path, elem))]
    return files
