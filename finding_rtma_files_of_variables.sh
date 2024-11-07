root_dir=$(pwd)

VARIABLES=("UGRD" "HGT" "TMP") # "v10"
wild_keys=("ed__" "5e__" "75__") # "d5__"
for i in "${!VARIABLES[@]}"; do
	VARIABLE="${VARIABLES[i]}"
    wild_key="${wild_keys[i]}"

    # Specify the output text file
    output_file="$root_dir/$VARIABLE.txt"

    # Clear the file if it already exists
    > "$output_file"

    cd /data/harish/rtma
    for folder in */; do
        echo "Searching in folder: $folder"
        
        # Find files matching *4d__* and loop over them
        find "$folder" -type f -name "*$wild_key*" | while read -r file; do
            # Display the file being processed
            echo $file >> "$output_file"
        done
    done
    cd /data/harish
    mkdir -p $VARIABLE
    rclone move --progress --transfers 12 --files-from "$output_file" /data/harish/rtma "$VARIABLE"
done

echo $root_dir