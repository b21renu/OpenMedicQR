# rename_labels.py (Corrected Version)
import os
import re

# IMPORTANT: Make sure this path exactly matches your folder structure.
# From your error message, it looks like this is the correct path.
labels_dir = 'data/demo_images/train_labels'

print(f"Starting rename in: {labels_dir}")

# This is the line we are fixing.
# The new pattern accounts for the '_jpg' part in the filename.
# It now looks for (imgXXX)_jpg.rf.longstring.txt

renamed_count = 0
skipped_count = 0

for filename in os.listdir(labels_dir):
    if filename.endswith('.txt'):
        # The corrected regex pattern
        match = re.match(r'(img\d{3})_jpg\.rf\.[\d\w]+\.txt', filename)
        
        if match:
            base_name = match.group(1)  # This will be 'img001'
            new_filename = f'{base_name}.txt'
            
            old_path = os.path.join(labels_dir, filename)
            new_path = os.path.join(labels_dir, new_filename)
            
            if os.path.exists(old_path):
                os.rename(old_path, new_path)
                renamed_count += 1
        else:
            # Let's check for already renamed files so we don't call them an error
            if not re.match(r'img\d{3}\.txt', filename):
                print(f"Skipping: {filename} (Pattern mismatch)")
                skipped_count += 1

print("-" * 30)
print(f"Renaming complete.")
print(f"Successfully renamed: {renamed_count} files.")
if skipped_count > 0:
    print(f"Skipped {skipped_count} files due to a name mismatch.")
print("Please check your 'data/train_labels' folder for the new filenames.")