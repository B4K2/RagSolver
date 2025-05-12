import os
import json
import chromadb
import logging
from tqdm import tqdm
import copy
import numpy as np # Sentence Transformers often returns numpy arrays
import time # For basic timing

# --- Image Handling ---
try:
    from PIL import Image
except ImportError:
    print("Pillow library not found. Install it: pip install Pillow")
    exit(1)

# --- Use standard libraries ---
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Transformer/Captioning Libraries ---
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
except ImportError:
    print("Transformers library not found. Install it: pip install transformers torch torchvision torchaudio")
    # Or: pip install transformers tensorflow Pillow (if using TensorFlow)
    exit(1)

# --- Configuration ---
OUTPUT_FOLDER_PATH = "./OUTPUT"
CHROMA_DB_PATH = "./chroma_db_standard"
COLLECTION_NAME = "my_documents_collection_standard"
DEVICE = "cuda" # Use "cuda" if GPU is available, otherwise "cpu"

# --- Captioning Model Configuration ---
# Using a smaller base model for potentially faster processing
CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-base"
# CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-large" # Alternative larger model

# --- Image File Extensions to Process ---
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif") # Add more if needed

# --- Helper Functions ---
def sanitize_metadata(metadata_dict):
    """
    Recursively sanitizes a dictionary to ensure all values are ChromaDB-compatible
    (str, int, float, bool). Converts lists/dicts to JSON strings. Non-serializable objects are converted to string.
    """
    sanitized = {}
    if not isinstance(metadata_dict, dict):
        return {}
    for key, value in metadata_dict.items():
        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, (list, dict)):
            try:
                # Attempt standard JSON serialization
                sanitized[key] = json.dumps(value, ensure_ascii=False)
            except TypeError:
                # Fallback: Convert elements to string if direct serialization fails
                try:
                   sanitized_value = json.dumps(str(value), ensure_ascii=False)
                   sanitized[key] = sanitized_value
                except Exception:
                     sanitized[key] = "[Unserializable Data]" # Final fallback
        elif value is None:
             sanitized[key] = "" # Or None, but string is safer for Chroma
        else:
             # Attempt to convert other types to string as a fallback
             try:
                  sanitized[key] = str(value)
             except Exception:
                  sanitized[key] = "[Unrepresentable Value]"
    return sanitized

# --- Chunking Configuration ---
try:
    chunker = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    print(f"Using Langchain RecursiveCharacterTextSplitter: size={chunker._chunk_size}, overlap={chunker._chunk_overlap}")
except ImportError:
    print("langchain library not found. Install it: pip install langchain")
    exit(1)
except Exception as e:
    print(f"Error initializing Langchain text splitter: {e}")
    exit(1)

# --- Embedder Configuration (Gemini) ---
try:
    from google import genai
    from google.genai import types as genai_types
    from dotenv import load_dotenv

    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_EMBEDDER_MODEL_ID = "embedding-001"

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set. Cannot initialize Gemini client.")

    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    print(f"Using Gemini Embedder: {GEMINI_EMBEDDER_MODEL_ID}")

    def get_gemini_embeddings(texts):
        response = gemini_client.models.embed_content(
            model=GEMINI_EMBEDDER_MODEL_ID,
            contents=texts,
            config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )
        if hasattr(response, 'embeddings') and response.embeddings:
            return [list(emb.values) for emb in response.embeddings]
        else:
            raise ValueError("Unexpected response structure from Gemini embed_content.")

except ImportError:
    print("google-generativeai library not found. Install it: pip install google-generativeai")
    exit(1)
except Exception as e:
    print(f"Error initializing Gemini Embedder: {e}")
    exit(1)

# --- Image Captioning Model Initialization ---
try:
    print(f"Loading Image Captioning model: {CAPTION_MODEL_NAME}...")
    caption_processor = BlipProcessor.from_pretrained(CAPTION_MODEL_NAME)
    caption_model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL_NAME).to(DEVICE)
    print(f"Image Captioning model ready on device {DEVICE}.")
except ImportError:
    print("transformers/torch library not found or import error.")
    print("Install it: pip install transformers torch torchvision torchaudio Pillow")
    exit(1)
except Exception as e:
    print(f"Error initializing Image Captioning model ({CAPTION_MODEL_NAME}): {e}")
    print("Ensure you have an internet connection to download the model.")
    exit(1)


# --- Initialize ChromaDB ---
print(f"Initializing ChromaDB client at path: {CHROMA_DB_PATH}")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# --- !! IMPORTANT: Collection Deletion !! ---
# This script DELETES the collection on each run.
# Comment out the try/except block below if you want to ADD to an existing collection.
# Be cautious about duplicates if you remove deletion and run multiple times on the same data.
try:
    client.delete_collection(COLLECTION_NAME)
    print(f"Attempted to delete existing collection (if any): {COLLECTION_NAME}")
except Exception as e:
    # Collection might not exist, or other error (like permissions)
    print(f"Note: Could not delete collection '{COLLECTION_NAME}' (might not exist): {e}")
    pass # Continue anyway, get_or_create handles non-existence

# Get embedding dimension (important for some Chroma setups, good practice)
try:
    embedding_dim = 768  # Default dimension for Gemini embedding-001
    print(f"Using embedding dimension: {embedding_dim}")
except Exception as e:
    print(f"Error determining embedding dimension: {e}")
    exit(1)

# Get or create the collection
try:
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine",  # Good default for sentence transformers
                  "embedding_model": GEMINI_EMBEDDER_MODEL_ID,
                  "captioning_model": CAPTION_MODEL_NAME}
        # embedding_function=None # We provide embeddings manually
    )
    print(f"ChromaDB collection '{COLLECTION_NAME}' ready.")
except Exception as e:
     print(f"Fatal Error: Could not get or create Chroma collection '{COLLECTION_NAME}': {e}")
     exit(1)


# --- Data Preparation (Text Chunks + Image Captions) ---
all_texts_to_embed = [] # Combined list for both text chunks and image captions
all_metadatas = []
all_ids = []

print(f"\nScanning directory: {os.path.abspath(OUTPUT_FOLDER_PATH)}")

subfolders_processed = 0
subfolders_skipped = 0
images_processed = 0
images_failed = 0

# Ensure the OUTPUT folder exists
if not os.path.isdir(OUTPUT_FOLDER_PATH):
    print(f"Error: Output directory not found at '{os.path.abspath(OUTPUT_FOLDER_PATH)}'")
    exit(1)

# Iterate through items in the output folder
total_items = len(os.listdir(OUTPUT_FOLDER_PATH))
for item_name in tqdm(os.listdir(OUTPUT_FOLDER_PATH), desc="Processing Folders", total=total_items):
    item_path = os.path.join(OUTPUT_FOLDER_PATH, item_name)

    if os.path.isdir(item_path):
        folder_path = item_path
        md_file_path = None
        json_file_path = None
        image_files = [] # List to store full paths of found image files
        original_doc_id = item_name # Use folder name as base ID

        # --- Stage 1: Find MD, JSON, and Image files ---
        try:
            found_md = False
            found_json = False
            for file_name in os.listdir(folder_path):
                file_path_full = os.path.join(folder_path, file_name)
                if not os.path.isfile(file_path_full):
                    continue # Skip sub-directories

                file_lower = file_name.lower()

                # Check for Markdown file
                if (file_lower.endswith((".md", ".markdown"))) and not found_md:
                     md_file_path = file_path_full
                     found_md = True
                # Check for JSON meta file
                elif file_lower.endswith("_meta.json") and not found_json:
                    json_file_path = file_path_full
                    found_json = True
                # Check for Image Files
                elif file_lower.endswith(IMAGE_EXTENSIONS):
                    image_files.append(file_path_full) # Store full path

        except FileNotFoundError:
            print(f"\nWarning: Folder '{item_name}' became inaccessible? Skipping.")
            subfolders_skipped += 1
            continue
        except Exception as e:
            print(f"\nError listing files in folder {item_name} ({folder_path}): {e}. Skipping.")
            subfolders_skipped += 1
            continue

        # --- Stage 2: Process if MD and JSON are found ---
        if md_file_path and json_file_path:
            try:
                # --- Process Text Content ---
                with open(md_file_path, 'r', encoding='utf-8') as f:
                    md_content = f.read()

                # Load and sanitize the primary metadata
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    raw_metadata_orig = json.load(f)

                if not isinstance(raw_metadata_orig, dict):
                     print(f"\nWarning: Metadata in {os.path.basename(json_file_path)} (folder: {item_name}) is not a dictionary. Skipping text processing.")
                     # Still attempt to process images below if found
                else:
                     metadata_orig_sanitized = sanitize_metadata(raw_metadata_orig) # Sanitize ONCE

                     if md_content and not md_content.isspace():
                         chunks = chunker.split_text(md_content)
                         if not chunks:
                              print(f"\nWarning: Chunker returned no text chunks for {os.path.basename(md_file_path)} (folder: {item_name}).")
                         else:
                             # Add text chunks to the lists
                             for i, chunk_text in enumerate(chunks):
                                 chunk_id = f"{original_doc_id}_chunk_{i}"
                                 chunk_metadata = copy.deepcopy(metadata_orig_sanitized) # Use sanitized copy
                                 chunk_metadata['original_doc_id'] = original_doc_id
                                 chunk_metadata['chunk_index'] = i
                                 chunk_metadata['source_markdown_file'] = os.path.basename(md_file_path)
                                 chunk_metadata['source_json_file'] = os.path.basename(json_file_path)
                                 chunk_metadata['content_type'] = 'text_chunk' # Add type identifier

                                 all_texts_to_embed.append(chunk_text)
                                 all_metadatas.append(sanitize_metadata(chunk_metadata)) # Sanitize again just in case
                                 all_ids.append(chunk_id)
                     else:
                         print(f"\nSkipping text processing for folder {item_name}: Markdown file '{os.path.basename(md_file_path)}' is empty.")


                # --- Process Images (if any were found) ---
                if image_files:
                    print(f"  Processing {len(image_files)} images for folder {item_name}...")
                    img_start_time = time.time()
                    for img_idx, img_path in enumerate(image_files):
                         img_filename = os.path.basename(img_path)
                         caption_id = f"{original_doc_id}_image_{img_idx}_{img_filename}"

                         try:
                             raw_image = Image.open(img_path).convert('RGB') # Load and ensure RGB

                             # Generate caption
                             inputs = caption_processor(images=raw_image, return_tensors="pt").to(DEVICE)
                             outputs = caption_model.generate(**inputs, max_new_tokens=50) # Increase max_new_tokens if needed
                             caption_text = caption_processor.decode(outputs[0], skip_special_tokens=True).strip()

                             if caption_text:
                                 # Create metadata for the caption
                                 # Start with a copy of the original *sanitized* doc metadata (if valid)
                                 if isinstance(raw_metadata_orig, dict): # Check if original metadata was valid
                                     caption_metadata = copy.deepcopy(metadata_orig_sanitized)
                                 else:
                                     caption_metadata = {} # Start fresh if original metadata was invalid

                                 caption_metadata['original_doc_id'] = original_doc_id
                                 caption_metadata['source_image_file'] = img_filename
                                 caption_metadata['source_markdown_file'] = os.path.basename(md_file_path) # Link back
                                 caption_metadata['source_json_file'] = os.path.basename(json_file_path) # Link back
                                 caption_metadata['content_type'] = 'image_caption' # Add type identifier

                                 all_texts_to_embed.append(caption_text)
                                 all_metadatas.append(sanitize_metadata(caption_metadata)) # Sanitize final metadata
                                 all_ids.append(caption_id)
                                 images_processed += 1
                             else:
                                 print(f"\n  Warning: Empty caption generated for {img_filename} in {item_name}. Skipping.")
                                 images_failed += 1

                         except FileNotFoundError:
                              print(f"\n  Error: Image file not found during processing: {img_path}. Skipping.")
                              images_failed += 1
                         except Exception as img_e:
                              print(f"\n  Error processing image {img_filename} in {item_name}: {img_e}. Skipping.")
                              images_failed += 1
                    img_end_time = time.time()
                    print(f"  Finished processing images for {item_name} in {img_end_time - img_start_time:.2f}s")

                subfolders_processed += 1 # Count as processed if MD/JSON existed

            except json.JSONDecodeError:
                print(f"\nError: Invalid JSON in file: {os.path.basename(json_file_path)} (folder: {item_name}). Skipping folder.")
                subfolders_skipped += 1
            except Exception as e:
                print(f"\nError processing folder {item_name} (path: {folder_path}): {e}. Skipping folder.")
                subfolders_skipped += 1
        else:
            # This case means either MD or JSON (or both) were missing
            missing_files = []
            if not md_file_path: missing_files.append("Markdown (.md/.markdown)")
            if not json_file_path: missing_files.append("Metadata (_meta.json)")
            # Don't print warning if folder is simply empty
            try:
                if any(os.scandir(folder_path)): # Check if directory is not empty
                     print(f"\nSkipping folder {item_name}: Missing required file(s): {', '.join(missing_files)}")
            except Exception:
                pass # Ignore errors listing dir content here
            subfolders_skipped += 1


print(f"\nScan complete.")
print(f"--> Processed {subfolders_processed} folders for text/metadata.")
print(f"--> Found and attempted to process {images_processed + images_failed} images.")
print(f"    - Successfully generated captions for: {images_processed} images.")
print(f"    - Failed to process/caption: {images_failed} images.")
print(f"--> Skipped {subfolders_skipped} items (non-directories or folders missing required files/content/valid JSON).")
print(f"--> Total items prepared for embedding (text chunks + image captions): {len(all_texts_to_embed)}")


# --- Embedding and Adding to ChromaDB ---
if all_texts_to_embed:
    print(f"\nEmbedding {len(all_texts_to_embed)} text chunks and image captions using Gemini model...")
    try:
        # Embed ALL texts (chunks and captions) together
        all_embeddings = get_gemini_embeddings(all_texts_to_embed)
        print("Generated embeddings using Gemini model.")

        # Verify list lengths
        if not (len(all_ids) == len(all_embeddings) == len(all_metadatas) == len(all_texts_to_embed)):
             print("\nCRITICAL ERROR: Mismatch in list lengths before adding to ChromaDB!")
             print(f"IDs: {len(all_ids)}, Embeddings: {len(all_embeddings)}, Metadatas: {len(all_metadatas)}, Documents: {len(all_texts_to_embed)}")
             exit(1)

    except Exception as e:
        print(f"Error generating embeddings with Gemini: {e}")
        exit(1)

    print(f"\nAdding {len(all_ids)} items to ChromaDB collection '{COLLECTION_NAME}'...")
    try:
        batch_size = 100 # Keep ChromaDB batch size reasonable
        for i in tqdm(range(0, len(all_ids), batch_size), desc="Adding batches to ChromaDB"):
             start_idx = i
             end_idx = min(i + batch_size, len(all_ids))
             ids_batch = all_ids[start_idx:end_idx]
             embeddings_batch = all_embeddings[start_idx:end_idx]
             metadatas_batch = all_metadatas[start_idx:end_idx]
             documents_batch = all_texts_to_embed[start_idx:end_idx] # Use the combined text list

             # Final check for batch consistency (optional but helpful for debugging)
             if not (len(ids_batch) == len(embeddings_batch) == len(metadatas_batch) == len(documents_batch)):
                 print(f"\nERROR: Batch inconsistency at index {start_idx}!")
                 continue # Skip this batch

             collection.add(
                 ids=ids_batch,
                 embeddings=embeddings_batch,
                 metadatas=metadatas_batch,
                 documents=documents_batch # Add the text chunk or caption text as the document
             )
        print("\nData added to ChromaDB successfully!")
        print(f"Collection '{COLLECTION_NAME}' now contains approximately {collection.count()} items.")

    except Exception as e:
        print(f"\nError adding data to ChromaDB: {e}")
        # You might want more detailed error logging here, e.g., print the failing batch data
        # for idx, item_id in enumerate(ids_batch):
        #    print(f"  Problematic Item?: ID={item_id}, Metadata={metadatas_batch[idx]}")


else:
    print("\nNo valid text chunks or image captions found to add to the database.")

print("\nScript finished.")