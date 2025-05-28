import argparse
import pandas as pd
import numpy as np # For np.nan, though pd.isna handles it
import re
import json
from pathlib import Path
from typing import Iterator, Dict, Any

# --- Core Processing Functions ---
def load_csv(path: Path) -> pd.DataFrame:
    """Loads a CSV file into a pandas DataFrame."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    return df

def clean_text(text: Any) -> str:
    """Cleans the input text."""
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.replace("___", "").strip()

def chunk_text_with_overlap(text: str, chunk_size: int, overlap: int) -> Iterator[str]:
    """Chunks text into overlapping segments."""
    words = text.split()
    if not words:
        return
    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size
    for start in range(0, len(words), step):
        yield " ".join(words[start : start + chunk_size])

def process_notes(
    df: pd.DataFrame, chunk_size: int, overlap: int
) -> Iterator[Dict[str, Any]]:
    """Cleans, filters (cardiac), and chunks notes from the DataFrame."""
    df["clean_text"] = df["text"].apply(clean_text)

    cardiac_keywords = ["cardiac", "heart", "myocardial", "coronary", "ECG", "heart failure"]
    df["clean_text"] = df["clean_text"].astype(str)
    original_rows = len(df)
    df_filtered = df[df["clean_text"].str.contains("|".join(cardiac_keywords), case=False, na=False)]

    if not df_filtered.empty:
        print(f"Filtered by cardiac keywords: {len(df_filtered)} rows from {original_rows}")
        df_to_process = df_filtered
    else:
        print(f"No rows matched cardiac keywords. Original {original_rows} rows (post-cleaning) will be considered if 'df_to_process = df' is used, otherwise no rows processed from this cardiac filter path.")
        df_to_process = df_filtered # Keeps original behavior: if filter empty, process empty

    # Define all columns you expect in the output, even if they might be missing in source
    # These will be pulled from the row, and if missing or NaN, will be set to None (null in JSON)
    output_metadata_cols = ["note_id", "subject_id", "hadm_id", "note_type", "note_seq", "charttime", "storetime"]

    for _, row in df_to_process.iterrows():
        base_record = {}
        
        # Populate base_record with values from row, converting NaN/missing to None
        for col_name in output_metadata_cols:
            val = row.get(col_name) # Use .get() in case column itself is missing from DataFrame row
            if pd.isna(val):
                base_record[col_name] = None
            elif col_name in ["charttime", "storetime"] and val is not None: # Ensure timestamps are strings
                base_record[col_name] = str(val)
            else:
                base_record[col_name] = val
        
        # Critical check: if note_id is None after this, skip row as it's a primary key for the chunk.
        if base_record["note_id"] is None:
            print(f"Warning: 'note_id' is missing or null for a row. Skipping this row's text processing.")
            continue

        row_clean_text = row.get("clean_text", "") # Get cleaned text safely
        for idx, chunk in enumerate(
            chunk_text_with_overlap(row_clean_text, chunk_size, overlap)
        ):
            chunk_record = base_record.copy() # Start with metadata
            chunk_record["chunk_index"] = idx
            chunk_record["chunk_text"] = chunk
            yield chunk_record

# --- Batch Processing Logic ---
def parse_batch_args():
    parser = argparse.ArgumentParser(
        description="Batch load CSV files, clean/chunk text, dump to JSON."
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("../data/raw/"),
        help="Directory containing raw CSV files."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("../data/processed/json/"),
        help="Directory for processed JSON files."
    )
    parser.add_argument(
        "--chunk-size", type=int, default=250, help="Words per chunk."
    )
    parser.add_argument(
        "--overlap", type=int, default=50, help="Overlapping words between chunks."
    )
    parser.add_argument(
        "--max-docs-per-file", type=int, default=1000,
        help="Max processed documents (chunks) to save per input CSV. 0 or negative for no limit. Default: 1000."
    )
    # --- New Argument for limiting input rows ---
    parser.add_argument(
        "--max-input-rows", type=int, default=1000,
        help="Max rows to read from start of each input CSV. 0 or negative for no limit. Default: 1000."
    )
    return parser.parse_args()

def process_single_csv(
    input_csv_path: Path, output_json_path: Path, 
    chunk_size: int, overlap: int, 
    max_docs: int, max_input_r: int # <-- Renamed for clarity
):
    print(f"Processing: {input_csv_path.name}")
    try:
        df_full = load_csv(input_csv_path)

        # --- Limit input rows ---
        if max_input_r > 0 and len(df_full) > max_input_r:
            print(f"INFO: Limiting input from {input_csv_path.name} to first {max_input_r} rows (out of {len(df_full)}).")
            df = df_full.head(max_input_r).copy() # Use .copy() to avoid SettingWithCopyWarning later
        else:
            df = df_full # Process all rows if no limit or limit not exceeded

        if "text" not in df.columns:
            print(f"SKIPPING: Column 'text' not found in {input_csv_path}. Required for processing.")
            return

        # Check for metadata columns (informational)
        expected_metadata_cols = ["note_id", "subject_id", "hadm_id", "note_type", "note_seq", "charttime", "storetime"]
        missing_cols = [col for col in expected_metadata_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Metadata columns missing in {input_csv_path.name}: {', '.join(missing_cols)}. They will be 'null' in output if not present.")

        documents = []
        processed_doc_count = 0
        limit_hit = False
        for doc in process_notes(df, chunk_size, overlap):
            if max_docs > 0 and processed_doc_count >= max_docs:
                limit_hit = True
                break
            documents.append(doc)
            processed_doc_count += 1
        
        if limit_hit:
            print(f"INFO: Reached document limit of {max_docs} for {input_csv_path.name}. Processed {processed_doc_count} documents from the (potentially limited) input rows.")

        if not documents:
            print(f"No processable documents generated for {input_csv_path.name}.")
            return

        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with output_json_path.open("w") as f:
            json.dump(documents, f, indent=2)
        print(f"SUCCESS: Saved {len(documents)} chunked documents to {output_json_path}")

    except FileNotFoundError:
        print(f"ERROR: Input file not found: {input_csv_path}")
    except pd.errors.EmptyDataError:
        print(f"ERROR: Input file is empty: {input_csv_path}")
    except Exception as e:
        print(f"ERROR: Failed processing {input_csv_path.name}. Reason: {type(e).__name__} - {e}")

def main_batch_processing():
    args = parse_batch_args()
    print(f"Starting batch processing...")
    print(f"Input directory: {args.input_dir.resolve()}")
    print(f"Output directory: {args.output_dir.resolve()}")
    print(f"Chunk size: {args.chunk_size}, Overlap: {args.overlap}")
    if args.max_input_rows > 0:
        print(f"Max input rows per CSV: {args.max_input_rows}")
    else:
        print(f"Max input rows per CSV: No limit")
    if args.max_docs_per_file > 0:
        print(f"Max processed documents per file: {args.max_docs_per_file}")
    else:
        print(f"Max processed documents per file: No limit")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_files_found = list(args.input_dir.glob("*.csv"))

    if not csv_files_found:
        print(f"No CSV files found in {args.input_dir}. Check path.")
        return

    print(f"Found {len(csv_files_found)} CSV file(s): {[f.name for f in csv_files_found]}")
    processed_count = 0
    for input_csv_path in csv_files_found:
        print(f"\n--- Processing file: {input_csv_path.name} ---")
        output_json_filename = f"{input_csv_path.stem}_processed.json"
        output_json_path = args.output_dir / output_json_filename
        
        process_single_csv(
            input_csv_path, output_json_path, 
            args.chunk_size, args.overlap,
            args.max_docs_per_file, args.max_input_rows # Pass new arg
        )
        processed_count += 1
    
    print(f"\n--- Batch processing finished. Attempted {processed_count} CSV file(s). ---")

if __name__ == "__main__":
    main_batch_processing()