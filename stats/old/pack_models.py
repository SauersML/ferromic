import os
import zipfile
import argparse
from tqdm import tqdm

def package_models_for_inference(input_dir: str, output_zip_path: str):
    """
    Finds all valid model/SNP pairs in a directory and packages them into a single zip file.

    A valid pair consists of a '.model.joblib' file and a corresponding '.snps.json' file
    with the same base name (inversion ID).

    Args:
        input_dir (str): The directory containing the trained model files.
        output_zip_path (str): The full path for the output .zip file.
    """
    print(f"--- Model Packaging Utility ---")
    print(f"Searching for models in: {os.path.abspath(input_dir)}")

    # 1. Validate that the input directory exists
    if not os.path.isdir(input_dir):
        print(f"\n[ERROR] Input directory not found: '{input_dir}'")
        print("Please specify the correct directory where your models are saved.")
        return

    # 2. Find all model files and identify their corresponding SNP metadata files
    all_files_in_dir = os.listdir(input_dir)
    model_files = sorted([f for f in all_files_in_dir if f.endswith(".model.joblib")])
    
    if not model_files:
        print("\n[ERROR] No '.model.joblib' files found in the specified directory.")
        print("Please ensure you have run the training script first.")
        return

    print(f"Found {len(model_files)} potential model files. Verifying pairs...")
    
    files_to_package = []
    skipped_count = 0

    for model_filename in model_files:
        # Derive the base name (e.g., 'HsInv0001') from the model filename
        inversion_id = model_filename.replace(".model.joblib", "")
        
        # Construct the expected SNP metadata filename
        snp_filename = f"{inversion_id}.snps.json"
        
        # Get the full paths to the files
        model_full_path = os.path.join(input_dir, model_filename)
        snp_full_path = os.path.join(input_dir, snp_filename)
        
        # Check if the corresponding SNP file exists
        if os.path.exists(snp_full_path):
            files_to_package.append({
                "model_path": model_full_path,
                "snp_path": snp_full_path,
                "model_arcname": model_filename, # Name inside the zip
                "snp_arcname": snp_filename     # Name inside the zip
            })
        else:
            print(f"  [WARNING] Skipping '{model_filename}': Corresponding SNP file '{snp_filename}' not found.")
            skipped_count += 1
            
    if not files_to_package:
        print("\n[ERROR] No valid model-SNP pairs were found to package.")
        return

    # 3. Create the zip archive and add the verified file pairs
    print(f"\nFound {len(files_to_package)} valid model pairs to package.")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} models due to missing SNP files.")
    
    print(f"Creating archive: {os.path.abspath(output_zip_path)}")

    try:
        # Use ZIP_DEFLATED for compression
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # The tqdm wrapper creates a nice progress bar
            for pair in tqdm(files_to_package, desc="Packaging models", unit="pair"):
                # Add the model file to the zip
                zipf.write(pair["model_path"], arcname=pair["model_arcname"])
                # Add the SNP metadata file to the zip
                zipf.write(pair["snp_path"], arcname=pair["snp_arcname"])
    except Exception as e:
        print(f"\n[FATAL ERROR] An error occurred while creating the zip file: {e}")
        return

    # 4. Final summary report
    final_size_mb = os.path.getsize(output_zip_path) / (1024 * 1024)
    print("\n" + "="*50)
    print("  Packaging Complete!")
    print("="*50)
    print(f"  - Total models packaged: {len(files_to_package)}")
    print(f"  - Total files in archive: {len(files_to_package) * 2}")
    print(f"  - Output file: {os.path.abspath(output_zip_path)}")
    print(f"  - Final size: {final_size_mb:.2f} MB")
    print("\nThis zip file contains everything needed for inference.")


def main():
    """Main function to parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Package trained imputation models and their SNP metadata into a single zip file for inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-i", "--input-dir",
        default="final_imputation_models",
        help="Path to the directory containing the model (.joblib) and SNP (.json) files."
    )
    
    parser.add_argument(
        "-o", "--output-file",
        default="imputation_models_package.zip",
        help="Path for the final output zip archive."
    )
    
    args = parser.parse_args()
    package_models_for_inference(args.input_dir, args.output_file)


if __name__ == "__main__":
    main()
