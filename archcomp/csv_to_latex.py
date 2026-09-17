import pandas as pd
import argparse
import sys
import re

def generate_latex(input_file, include_median=False, highlight=False, precision=1, output_file=None):
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}", file=sys.stderr)
        sys.exit(1)

    # Filter suffixes
    suffixes = set()
    for col in df.columns:
        m = re.search(r'_(\d+)$', col)
        if m:
            suffixes.add(int(m.group(1)))
    
    sorted_suffixes = sorted(list(suffixes))
    
    if not sorted_suffixes:
        print("No numbered suffixes found in column names.", file=sys.stderr)
        return

    # Determine output stream
    if output_file:
        try:
            out_stream = open(output_file, 'w')
        except Exception as e:
            print(f"Error opening output file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        out_stream = sys.stdout

    try:
        if highlight:
            print(r"\newcommand{\hl}{\cellcolor{green!25}}", file=out_stream)

        # Table preamble
        print(r"\begin{table}[t]", file=out_stream)
        print(r"\centering", file=out_stream)
        print(r"\caption{Write caption here.}", file=out_stream)
        print(r"\label{tab:table-label}", file=out_stream)
        print(r"\setlength{\tabcolsep}{3.9pt} % Adjust this value to increase/decrease spacing", file=out_stream)
        
        # Tabular def
        cols_per_group = 2 + (1 if include_median else 0)
        
        col_def = "l|"
        for i, _ in enumerate(sorted_suffixes):
            col_def += "r" * cols_per_group
            if i < len(sorted_suffixes) - 1:
                col_def += "|"
                
        print(r"\begin{tabular}{" + col_def + "}", file=out_stream)
        print(r"\toprule", file=out_stream)
        
        # Header 1
        h1_parts = ["~"]
        for i, s in enumerate(sorted_suffixes):
            sep = "|" if i < len(sorted_suffixes) - 1 else ""
            h1_parts.append(r"\multicolumn{" + str(cols_per_group) + r"}{c" + sep + r"}{" + str(s) + r"}")
            
        print(" & ".join(h1_parts) + r" \\", file=out_stream)
        
        # Header 2
        h2_parts = ["Spec"]
        group_headers = ["FR", r"$\overline{S}$"]
        if include_median:
            group_headers.append(r"$\tilde{S}$")
            
        for _ in sorted_suffixes:
            h2_parts.extend(group_headers)
            
        print(" & ".join(h2_parts) + r" \\", file=out_stream)
        print(r"\midrule", file=out_stream)
        
        # Data
        last_prefix = None
        
        if 'specification' not in df.columns:
             print("Error: 'specification' column not found.", file=sys.stderr)
             return

        for idx, row in df.iterrows():
            spec = str(row['specification'])
            
            # Determine prefix for midrule
            m = re.match(r'^([A-Z]+)', spec)
            prefix = m.group(1) if m else spec
            
            if last_prefix is not None and prefix != last_prefix:
                print(r"\midrule", file=out_stream)
                
            last_prefix = prefix
            
            # Determine winners if highlight is enabled
            winners = set()
            if highlight:
                candidates = []
                for s in sorted_suffixes:
                    # FR
                    fr_col = f"falsification_ratio_{s}"
                    fr_val = row.get(fr_col)
                    try:
                        fr = float(fr_val) if pd.notna(fr_val) and str(fr_val).strip() != "" else 0.0
                    except ValueError:
                        fr = 0.0
                    
                    # Avg
                    avg_col = f"nfev_avg_falsified_{s}"
                    avg_val = row.get(avg_col)
                    try:
                        avg = float(avg_val) if pd.notna(avg_val) and str(avg_val).strip() != "" else float('inf')
                    except ValueError:
                        avg = float('inf')
                    
                    if fr > 0:
                        candidates.append({'s': s, 'fr': fr, 'avg': avg})
                
                if candidates:
                    # Sort by FR desc, then Avg asc
                    candidates.sort(key=lambda x: (-x['fr'], x['avg']))
                    
                    best = candidates[0]
                    # Find all that are close enough to best
                    for c in candidates:
                        if abs(c['fr'] - best['fr']) < 1e-9 and abs(c['avg'] - best['avg']) < 1e-9:
                            winners.add(c['s'])

            row_parts = [spec]
            
            for s in sorted_suffixes:
                is_winner = s in winners
                hl_prefix = r"\hl " if is_winner else ""

                # FR
                fr_col = f"falsification_ratio_{s}"
                fr_val = row.get(fr_col)
                
                if pd.isna(fr_val) or str(fr_val).strip() == "":
                    row_parts.append("-")
                else:
                    try:
                        val = float(fr_val) * 10
                        # Check if integer
                        s_val = str(int(round(val)))
                        row_parts.append(f"{hl_prefix}{s_val}")
                    except ValueError:
                        row_parts.append(str(fr_val))
                
                # Avg
                avg_col = f"nfev_avg_falsified_{s}"
                avg_val = row.get(avg_col)
                
                if pd.isna(avg_val) or str(avg_val).strip() == "":
                    row_parts.append("-")
                else:
                    try:
                        val = float(avg_val)
                        s_val = f"{val:.{precision}f}"
                        row_parts.append(f"{hl_prefix}{s_val}")
                    except ValueError:
                        row_parts.append(str(avg_val))

                # Median
                if include_median:
                    med_col = f"nfev_median_falsified_{s}"
                    med_val = row.get(med_col)
                    if pd.isna(med_val) or str(med_val).strip() == "":
                        row_parts.append("-")
                    else:
                        try:
                            val = float(med_val)
                            s_val = f"{val:.{precision}f}"
                            row_parts.append(s_val)
                        except ValueError:
                            row_parts.append(str(med_val))
                            
            print(" & ".join(row_parts) + r" \\", file=out_stream)
            
        print(r"\bottomrule", file=out_stream)
        print(r"\end{tabular}", file=out_stream)
        print(r"\end{table}", file=out_stream)
    finally:
        if output_file:
            out_stream.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV to LaTeX table", add_help=False)
    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument("--median", action="store_true", help="Include median columns")
    parser.add_argument("-h", "--highlight", action="store_true", help="Highlight best results")
    parser.add_argument("-p", "--precision", type=int, default=1, help="Precision for the table")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output file path (default: print to stdout)")
    parser.add_argument("--help", action="help", help="show this help message and exit")
    
    args = parser.parse_args()
    
    generate_latex(args.input_file, args.median, args.highlight, args.precision, args.output)
