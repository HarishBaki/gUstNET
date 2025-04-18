#!/bin/bash
variables=("PRES")
max_concurrent_runs=48  # Maximum concurrent runs
run_count=0             # Counter for tracking concurrent runs

# Loop over variables and wild_keys simultaneously
for i in "${!variables[@]}"; do
    variable="${variables[i]}"

    # Loop over the years 2018 to 2023
    for year in {2021..2021}; do
        # Loop over the months 1 to 12
        for month in {01..12}; do
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
                time="$year$month$day"
                echo "Processing $time with variable=$variable"
                
                # Run the Python script with the arguments
                python processing_daily_rtma.py "$time" "$variable" &

                ((run_count++))

                # Check if we've reached the maximum number of concurrent runs
                if (( run_count >= max_concurrent_runs )); then
                    # Wait for all background jobs to complete
                    wait
                    run_count=0
                fi
            done
        done
        wait
        # Combine yearly data
        python processing_yearly_rtma.py "$variable" "$year"
    done
done