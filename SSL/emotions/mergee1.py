import pandas as pd
import os

# File paths
merged_csv_path = "merged.csv"  # CSV containing filename and transcription
paths_csv_path = "paths"      # CSV containing wavpath
output_csv_path = "updated.csv"  # Output CSV

# Load merged CSV with pipe delimiter and strip spaces
merged_df = pd.read_csv(merged_csv_path, delimiter="|", skipinitialspace=True)
merged_df.columns = merged_df.columns.str.strip()  # Strip column headers
merged_df['filename'] = merged_df['filename'].str.strip()  # Strip filename column values

# Load paths CSV with pipe delimiter and strip spaces
paths_df = pd.read_csv(paths_csv_path, delimiter="|", skipinitialspace=True)
paths_df.columns = paths_df.columns.str.strip()  # Strip column headers
paths_df['wavpath'] = paths_df['wavpath'].str.strip()  # Strip wavpath column values

# Extract filename from full path and create a mapping
paths_df['filename'] = paths_df['wavpath'].apply(lambda x: os.path.basename(x).strip())
path_mapping = dict(zip(paths_df['filename'], paths_df['wavpath']))

# Ensure filename is stripped before mapping
merged_df['wavpath'] = merged_df['filename'].map(lambda x: path_mapping.get(x.strip(), "N/A"))

# Reorder columns
merged_df = merged_df[['filename', 'wavpath', 'transcription']]

# Save the updated CSV with pipe delimiter
merged_df.to_csv(output_csv_path, sep="|", index=False)

print(f"Merged CSV saved as {output_csv_path}")

