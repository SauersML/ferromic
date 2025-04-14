import os
import subprocess
import shutil
import sys
import platform
from pathlib import Path
from typing import Dict, List

# --- Configuration ---

# Map script names to their expected output image file names
# This makes the relationship explicit and drives the execution flow.
SCRIPT_IMAGE_MAP: Dict[str, str] = {
    'recur_diversity.py': 'inversion_pi_violins.png',
    'overall_fst_by_type.py': 'fst_recurrent_vs_single_event.png',
    'dist_diversity_by_type.py': 'pi_flanking_regions_mean_bar_plot.png',
}

# Name of the directory to store the final plots
OUTPUT_DIR_NAME: str = "final_plots"

# --- Helper Functions ---

def run_script(script_path: Path) -> bool:
    """
    Executes a Python script using the same interpreter and captures output.
    Returns True on success, False on failure.
    """
    print(f"--- Running script: {script_path.name} ---")
    try:
        # Use sys.executable to ensure the same Python environment
        process = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,         # Raise CalledProcessError if script fails
            capture_output=True,
            text=True,
            cwd=script_path.parent # Run from the script's directory
        )
        # Log stdout only if it's not empty, stderr always if not empty
        if process.stdout.strip():
            print(f"Stdout from {script_path.name}:\n{process.stdout.strip()}")
        if process.stderr.strip():
            print(f"Stderr from {script_path.name}:\n{process.stderr.strip()}")
        print(f"--- Finished script: {script_path.name} successfully ---")
        return True
    except FileNotFoundError:
        print(f"ERROR: Script not found: {script_path}")
        return False
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Script {script_path.name} failed (exit code {e.returncode}).")
        # Print captured output on error for debugging
        if e.stdout.strip(): print(f"Stdout:\n{e.stdout.strip()}")
        if e.stderr.strip(): print(f"Stderr:\n{e.stderr.strip()}")
        return False
    except Exception as e:
        print(f"ERROR: An unexpected error occurred running {script_path.name}: {e}")
        return False

def open_file(file_path: Path):
    """Opens a file using the default system application."""
    try:
        print(f"Attempting to open: {file_path.name}")
        system = platform.system()
        if system == "Windows":
            os.startfile(file_path)
        elif system == "Darwin":  # macOS
            subprocess.run(['open', str(file_path)], check=True)
        else:  # Linux and other POSIX
            subprocess.run(['xdg-open', str(file_path)], check=True)
    except FileNotFoundError:
        print(f"ERROR: Could not find application to open '{file_path.name}'. "
              f"Is a default viewer installed for '{file_path.suffix}' files?")
    except Exception as e:
        print(f"ERROR: Failed to open '{file_path.name}': {e}")

# --- Main Execution ---

def main():
    """Orchestrates running analysis scripts, copying, and opening images."""
    start_time = Path().cwd() # Use Path object for current directory
    output_dir = start_time / OUTPUT_DIR_NAME

    print(f"Working Directory: {start_time}")
    print(f"Target Plot Directory: {output_dir}")

    # 1. Ensure Output Directory Exists
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Ensured output directory '{output_dir.name}' exists.")
    except OSError as e:
        print(f"FATAL ERROR: Could not create output directory '{output_dir}': {e}")
        sys.exit(1) # Exit if we can't create the directory

    copied_image_paths: List[Path] = []
    all_scripts_succeeded: bool = True

    # 2. Run Scripts and Collect/Copy Images
    print("\n--- Running Analysis Scripts ---")
    for script_name, image_name in SCRIPT_IMAGE_MAP.items():
        script_path = start_time / script_name
        source_image_path = start_time / image_name
        dest_image_path = output_dir / image_name

        if not script_path.is_file():
            print(f"ERROR: Script '{script_name}' not found. Skipping.")
            all_scripts_succeeded = False
            continue

        # Run the script
        success = run_script(script_path)
        if not success:
            all_scripts_succeeded = False
            print(f"WARNING: Script '{script_name}' failed. Check logs above.")
            # Continue processing other scripts and checking for images

        # Check for and copy the expected image AFTER running the script
        print(f"Checking for output: {image_name}")
        if source_image_path.is_file():
            try:
                shutil.copy2(source_image_path, dest_image_path)
                print(f"  Copied '{image_name}' to '{output_dir.name}/'")
                copied_image_paths.append(dest_image_path)
            except Exception as e:
                print(f"  ERROR: Failed to copy '{image_name}': {e}")
        else:
            # Only warn if the script itself succeeded but the image is missing
            if success:
                 print(f"  WARNING: Expected image '{image_name}' not found after running '{script_name}'.")
            else:
                 print(f"  INFO: Expected image '{image_name}' not found (script '{script_name}' failed).")


    # 3. Report Overall Status and Open Images
    print("\n--- Script Execution Summary ---")
    if not all_scripts_succeeded:
        print("WARNING: One or more analysis scripts reported errors.")
    else:
        print("All analysis scripts completed without reported errors.")

    if copied_image_paths:
        print(f"\n--- Opening {len(copied_image_paths)} Copied Images ---")
        for img_path in copied_image_paths:
            open_file(img_path)
    else:
        print("\nNo images were found or successfully copied. Cannot open images.")

    print("\n--- Master Runner Script Finished ---")

if __name__ == "__main__":
    main()
