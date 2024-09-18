#!/bin/bash

# Variables to be passed to Python script
VARIABLE="GUST"
LEVEL="10 m"
DOWNLOAD_PATH="/data/harish"

# Loop over the years 2018 to 2023
for year in {2018..2023}; do
  # Loop over the months 1 to 12
  for month in {11..12}; do
    # Determine the number of days in the month, accounting for leap years
    if [[ "$month" == "04" || "$month" == "06" || "$month" == "09" || "$month" == "11" ]]; then
      days_in_month=30
    elif [[ "$month" == "02" ]]; then
      # Check for leap year (divisible by 4 and not 100 unless also divisible by 400)
      if (( (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0) )); then
        days_in_month=29
      else
        days_in_month=28
      fi
    else
      days_in_month=31
    fi
    
    # Loop over the days of the month
    for day in $(seq -w 1 $days_in_month); do
      # Loop over the hours of the day
      for hour in {00..23}; do
        # Call the Python script with the arguments
        python3 RTMA_download_variable_and_instancewise.py "$VARIABLE" "$LEVEL" "$year" "$month" "$day" "$hour" "$DOWNLOAD_PATH" &
      done
      wait
    done
  done
done

echo "All downloads have been finished."
