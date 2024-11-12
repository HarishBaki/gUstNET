VARIABLES=("WDIR" "DPT" "SPFH" "PRES") 
for i in "${!VARIABLES[@]}"; do
	VARIABLE="${VARIABLES[i]}"
    rclone copy --progress --transfers 12 "/data/harish/RTMA/$VARIABLE" "harish_hal:/data/harish/RTMA/$VARIABLE"
done
echo "All transfers have been finished."