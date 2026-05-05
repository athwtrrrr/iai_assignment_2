import subprocess
import os
from pathlib import Path

# Configuration
TEST_DIR = "test_cases"          # Folder containing .txt test files
METHODS = ["DFS", "BFS", "GBFS", "AS", "UCS", "IDA", "MULTI"]
OUTPUT_FILE = "test_results.txt" # Where to save results

def run_all_tests():
    test_files = sorted(Path(TEST_DIR).glob("*.txt"))
    if not test_files:
        print(f"No .txt files found in '{TEST_DIR}/'")
        return

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for test_file in test_files:
            out.write(f"\n{'='*60}\n")
            out.write(f"Test File: {test_file.name}\n")
            out.write(f"{'='*60}\n")
            print(f"Running tests on {test_file.name}...")

            for method in METHODS:
                # Build command: python -m src.main <test_file> <method>
                cmd = ["python3", "-m", "src.main", str(test_file), method]
                result = subprocess.run(cmd, capture_output=True, text=True)

                # Write the standard output
                out.write(f" {result.stdout.strip()}\n")
                # If there was an error, write it as well
                if result.stderr:
                    out.write(f"      ERR: {result.stderr.strip()}\n")

    print(f"All tests completed. Results saved to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    run_all_tests()