cd /data/harish/rtma

for folder in */; do
    echo "Searching in folder: $folder"
    
    # Find files matching *4d__* and loop over them
    find "$folder" -type f -name "*4d__*" | while read -r file; do
        # Display the file being processed
        echo "Found file: $file"

        # Copy the file to the remote location with the same folder structure
        #rclone copy -P --transfers 1 "$file" "harish_hal:/data/harish/rtma/i10fg/$folder"
        rm -f "$file"
    done
done

