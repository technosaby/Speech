#!/bin/bash

output_file="merged.csv"
first_file=true

while IFS= read -r csv_file; do
    if [ -f "$csv_file" ]; then
        if $first_file; then
            # Copy header and data for the first file
            cat "$csv_file" > "$output_file"
            first_file=false
        else
            # Skip header (first line) for subsequent files
            tail -n +2 "$csv_file" >> "$output_file"
        fi
    else
        echo "File $csv_file not found!"
    fi
done < meta

echo "Merged CSV saved as $output_file"

