#!/bin/bash

# Set the root folder path and target size in bytes
root_folder="/data/harish/rtma"  # Update this to the actual root folder path
target_size=134827360

# Loop through each subfolder in the root folder
for folder in "$root_folder"/*; do
    if [ -d "$folder" ]; then
        # Calculate the size of the folder (in bytes)
        folder_size=$(du -sb "$folder" | awk '{print $1}')
        
        # Check if the folder size is not equal to the target size
        if [ "$folder_size" -ne "$target_size" ]; then
            echo "Folder: $(basename "$folder") | Size: $folder_size bytes"
        fi
    fi
done

