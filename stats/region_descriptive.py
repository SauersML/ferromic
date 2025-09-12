import csv
import numpy as np
import io
import os

def calculate_inversion_stats(input_filename="inv_info.tsv", output_filename="inversion_stats.csv"):
    """
    Reads inversion data from the input TSV file, calculates descriptive
    statistics (mean and standard deviation) for frequency, size, and
    flanking repeat identity, grouped by recurrence status (excluding Y
    chromosome), and saves the results to the specified output CSV file
    with human-readable headers (using spaces).

    Args:
        input_filename (str): The path to the input TSV file (e.g., "inv_info.tsv").
        output_filename (str): The path to save the output CSV file (e.g., "inversion_stats.csv").

    Returns:
        str: A success or error message indicating the outcome.
    """
    if not os.path.exists(input_filename):
        return f"Error: Input file not found at '{input_filename}'."

    recurrent_data = {'freq': [], 'size': [], 'identity': []}
    single_event_data = {'freq': [], 'size': [], 'identity': []}

    try:
        with open(input_filename, 'r', newline='', encoding='utf-8') as csvfile: # encoding
            # Read the first line to get headers and clean them
            first_line = csvfile.readline()
            if first_line.startswith('\ufeff'): # Handle BOM
                first_line = first_line[1:]
            header = [h.strip() for h in first_line.strip().split('\t')]

            # --- Identify column indices robustly ---
            column_mapping = {
                'chrom': 'Chromosome',
                'recur': '0_single_1_recur',
                'size': 'Size_.kbp.',
                'freq': 'Inverted_AF',
                'identity': 'Flanking_Inverted_repeat_identity'
            }
            col_indices = {}
            missing_cols = []
            for key, expected_name in column_mapping.items():
                try:
                    col_indices[key] = header.index(expected_name)
                except ValueError:
                    # Try finding case-insensitive or with slightly different spacing just in case
                    found = False
                    for i, h in enumerate(header):
                        if h.lower() == expected_name.lower():
                           col_indices[key] = i
                           found = True
                           break
                    if not found:
                        missing_cols.append(expected_name)


            if missing_cols:
                return f"Error: Missing expected column(s) in TSV header: {', '.join(missing_cols)}"

            reader = csv.reader(csvfile, delimiter='\t')

            for row in reader:
                if not row: continue
                if len(row) <= max(col_indices.values()): continue

                if row[col_indices['chrom']].strip().lower() == 'chry': continue

                recur_status = row[col_indices['recur']].strip()
                if recur_status not in ['0', '1']: continue

                # --- Extract and clean data ---
                try:
                    freq_str = row[col_indices['freq']].strip()
                    freq = float(freq_str) if freq_str else np.nan
                except (ValueError, IndexError): freq = np.nan

                try:
                    size_str = row[col_indices['size']].strip()
                    size = float(size_str) if size_str else np.nan
                except (ValueError, IndexError): size = np.nan

                try:
                    identity_str = row[col_indices['identity']].strip().replace('%', '')
                    if identity_str.upper() == 'NA' or not identity_str:
                        identity = np.nan
                    else:
                        identity = float(identity_str)
                except (ValueError, IndexError): identity = np.nan

                target_data = recurrent_data if recur_status == '1' else single_event_data
                target_data['freq'].append(freq)
                target_data['size'].append(size)
                target_data['identity'].append(identity)

    except FileNotFoundError:
        return f"Error: Input file not found at '{input_filename}'."
    except Exception as e:
        return f"An unexpected error occurred while processing the input file: {e}"

    # --- Calculate Statistics ---
    stats = {}
    categories = {
        'Recurrent': recurrent_data,
        'Single-event': single_event_data
    }

    for category_name, data in categories.items():
        n = len(data['freq'])
        freq_arr = np.array(data['freq'], dtype=float)
        size_arr = np.array(data['size'], dtype=float)
        identity_arr = np.array(data['identity'], dtype=float)

        n_freq = np.count_nonzero(~np.isnan(freq_arr))
        n_size = np.count_nonzero(~np.isnan(size_arr))
        n_identity = np.count_nonzero(~np.isnan(identity_arr))

        stats[category_name] = {
            'n': n,
            'freq': {
                'mean': np.nanmean(freq_arr) if n_freq > 0 else np.nan,
                'std_dev': np.nanstd(freq_arr, ddof=1) if n_freq > 1 else np.nan
            },
            'size': {
                'mean': np.nanmean(size_arr) if n_size > 0 else np.nan,
                'std_dev': np.nanstd(size_arr, ddof=1) if n_size > 1 else np.nan
            },
            'identity': {
                'mean': np.nanmean(identity_arr) if n_identity > 0 else np.nan,
                'std_dev': np.nanstd(identity_arr, ddof=1) if n_identity > 1 else np.nan
            }
        }

    # --- Format Output CSV String ---
    output = io.StringIO()
    # Use comma delimiter for standard CSV, quoting fields with spaces/commas
    writer = csv.writer(output, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')

    # Write Header with Spaces
    writer.writerow([
        "Category",
        "Frequency AF Mean (SD)", # Use spaces in headers
        "Size kbp Mean (SD)",
        "Flanking Repeat Identity % Mean (SD)"
    ])

    # Helper function to format Mean (SD) string, handling NaNs
    def format_mean_sd(mean_val, sd_val, mean_prec, sd_prec):
        if np.isnan(mean_val): mean_str = "NA"
        else: mean_str = f"{mean_val:.{mean_prec}f}"

        if np.isnan(sd_val): sd_str = "NA"
        else: sd_str = f"{max(0, sd_val):.{sd_prec}f}" # SD is non-negative

        return f"{mean_str} ({sd_str})"

    # Write Data Rows
    for category_name, category_stats in stats.items():
        n_val = category_stats['n']
        cat_label = f"{category_name} (n={n_val})"

        freq_str = format_mean_sd(
            category_stats['freq']['mean'], category_stats['freq']['std_dev'], 4, 4
        )
        size_str = format_mean_sd(
            category_stats['size']['mean'], category_stats['size']['std_dev'], 3, 3
        )
        identity_str = format_mean_sd(
            category_stats['identity']['mean'], category_stats['identity']['std_dev'], 2, 2
        )

        writer.writerow([cat_label, freq_str, size_str, identity_str])

    # --- Write to Output File ---
    try:
        # writing with UTF-8 encoding
        with open(output_filename, 'w', newline='', encoding='utf-8') as outfile:
            outfile.write(output.getvalue())
        return f"Successfully calculated statistics and saved to '{output_filename}'."
    except Exception as e:
        return f"Error: Could not write to output file '{output_filename}'. Reason: {e}"

# --- Main execution block ---
if __name__ == "__main__":
    result_message = calculate_inversion_stats("inv_info.tsv", "inversion_stats.csv")
    print(result_message)
