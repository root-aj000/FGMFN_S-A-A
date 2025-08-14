import os

def create_structure_from_list(structure_file_path):
    with open(structure_file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    for path in lines:
        path = path.replace("\\", "/")  # Ensure forward slashes
        if path.endswith("/"):
            # It's a directory
            os.makedirs(path, exist_ok=True)
        else:
            # It's a file
            dir_path = os.path.dirname(path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(path, 'a', encoding='utf-8'):
                pass  # Just touch the file

    print("All files and folders created successfully.")

if __name__ == "__main__":
    create_structure_from_list("structure.txt")
