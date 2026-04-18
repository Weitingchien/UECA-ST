import shutil
import os
import sys

# Define directories relative to current working directory
source_dir = "ep_split10_home_t1_te1_v1_u7_disjoint"
dest_dir = "ep_split10_home_t1te1v1_u7_disjoint"

# Print absolute paths for verification
cwd = os.getcwd()
abs_source = os.path.join(cwd, source_dir)
abs_dest = os.path.join(cwd, dest_dir)

print(f"Current Working Directory: {cwd}")
print(f"Source: {abs_source}")
print(f"Destination: {abs_dest}")

if not os.path.exists(source_dir):
    print(f"Error: Source directory '{source_dir}' does not exist.")
    sys.exit(1)

if not os.path.exists(dest_dir):
    print(f"Error: Destination directory '{dest_dir}' does not exist.")
    sys.exit(1)

# List items to move
items = os.listdir(source_dir)
if not items:
    print("Source directory is empty. Nothing to move.")
    # Attempt remove if empty
    try:
        os.rmdir(source_dir)
        print(f"Removed empty source directory: {source_dir}")
    except OSError as e:
        print(f"Could not remove empty source directory: {e}")
    sys.exit(0)

print(f"Found {len(items)} items to move.")

for item in items:
    src_path = os.path.join(source_dir, item)
    dst_path = os.path.join(dest_dir, item)
    
    if os.path.exists(dst_path):
        print(f"Skipping '{item}': Already exists in destination.")
    else:
        try:
            print(f"Moving '{item}'...")
            shutil.move(src_path, dst_path)
        except Exception as e:
            print(f"Failed to move '{item}': {e}")

# Check if source is empty now and remove it
remaining_items = os.listdir(source_dir)
if not remaining_items:
    print(f"All items moved. Removing source directory: {source_dir}")
    try:
        os.rmdir(source_dir)
        print("Success.")
    except OSError as e:
        print(f"Could not remove source directory: {e}")
else:
    print(f"Source directory not empty ({len(remaining_items)} items remain). Not removing.")
