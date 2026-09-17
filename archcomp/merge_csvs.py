import pandas as pd
import argparse
import sys
import os

def merge_csvs(input_files, output_file):
    dfs = []
    
    for i, file_path in enumerate(input_files, 1):
        try:
            df = pd.read_csv(file_path)
            # Check if 'specification' column exists
            # Handle case sensitivity if needed, but assuming strictly 'specification' based on prompt
            if 'specification' not in df.columns:
                # Try stripping whitespace from columns
                df.columns = df.columns.str.strip()
                if 'specification' not in df.columns:
                    print(f"Error: 'specification' column not found in {file_path}", file=sys.stderr)
                    # List available columns to help debug
                    print(f"Available columns: {list(df.columns)}", file=sys.stderr)
                    return

            # Set specification as index to facilitate joining
            df = df.set_index('specification')
            
            # Rename columns with suffix
            df = df.add_suffix(f'_{i}')
            
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}", file=sys.stderr)
            return

    if not dfs:
        print("No valid CSV files to merge.", file=sys.stderr)
        return

    # Merge all dataframes horizontally
    # outer join ensures we keep all specifications even if they don't appear in all files
    merged_df = pd.concat(dfs, axis=1, join='outer')
    
    # Reset index to make specification a column again
    merged_df = merged_df.reset_index()
    
    # Save to output file
    try:
        merged_df.to_csv(output_file, index=False)
        print(f"Successfully merged {len(input_files)} files into {output_file}")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple CSV files horizontally based on 'specification' column.")
    parser.add_argument("files", nargs='+', help="Input CSV files")
    parser.add_argument("-o", "--output", default="merged_all.csv", help="Output CSV file path (default: merged_all.csv)")
    
    args = parser.parse_args()
    
    merge_csvs(args.files, args.output)

