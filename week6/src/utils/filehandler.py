import shutil
import os

def move_file(filename, source_dir, destination_dir):
    # Ensure source directory exists
    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' does not exist.")
        return
    
    # Ensure destination directory exists, if not create it
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    
    source_file = os.path.join(source_dir, filename)
    destination_file = os.path.join(destination_dir, filename)
    
    # Move the file
    shutil.move(source_file, destination_file)
    print(f"Moved '{source_file}' to '{destination_file}'")

def copy_files_containing_50(source_dir, destination_dir):
    """
    Copy files containing '50' in their filenames from source directory to destination directory.
    
    Args:
        source_dir (str): Path to the source directory.
        destination_dir (str): Path to the destination directory.
    """
    # Create destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    
    # Iterate over files in the source directory
    for filename in os.listdir(source_dir):
        # Check if the filename contains '50'
        if '50' in filename:
            # Construct paths for source and destination files
            source_file_path = os.path.join(source_dir, filename)
            destination_file_path = os.path.join(destination_dir, filename)
            
            # Copy the file to the destination directory
            shutil.copy2(source_file_path, destination_file_path)
            print(f"File '{filename}' copied to '{destination_dir}'.")


