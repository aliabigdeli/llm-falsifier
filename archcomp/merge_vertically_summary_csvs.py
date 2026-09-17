import argparse
import pandas as pd
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Merge CSV files vertically.")
    parser.add_argument("files", nargs='+', help="List of CSV files to merge")
    parser.add_argument("-o", "--output", help="Output file path", default="merged_output.csv")
    
    args = parser.parse_args()
    
    if len(args.files) < 2:
        print("Warning: merging fewer than 2 files.")

    dataframes = []
    for file_path in args.files:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}", file=sys.stderr)
            continue
            
        try:
            df = pd.read_csv(file_path)
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}", file=sys.stderr)
    
    if not dataframes:
        print("No valid CSV files to merge.", file=sys.stderr)
        sys.exit(1)
        
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    try:
        merged_df.to_csv(args.output, index=False)
        print(f"Successfully merged {len(dataframes)} files into {args.output}")
    except Exception as e:
        print(f"Error writing to output file {args.output}: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
