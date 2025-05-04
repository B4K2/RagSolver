# RagSolver: Visual PDF RAG Agent

RagSolver is an experimental Retrieval-Augmented Generation (RAG) system designed to answer procedural "how-to" questions. It uniquely leverages PDF documents that primarily consist of step-by-step screenshots, making it suitable for visual guides.

The system processes these PDFs, extracts text, captions the images using an AI model, creates vector embeddings for both text and image captions, stores them in a vector database, and uses a Large Language Model (LLM) agent to retrieve relevant information and answer user queries.

## Features

*   **Visual PDF Processing:** Uses the [Marker](https://github.com/VikParuchuri/marker) tool to parse PDFs, extracting text, metadata, and images.
*   **Image Captioning:** Employs the BLIP model (via Hugging Face Transformers) to generate descriptive text captions for extracted images.
*   **Text Chunking:** Uses LangChain's `RecursiveCharacterTextSplitter` for effective text segmentation.
*   **Vector Embeddings:** Generates embeddings for both text chunks and image captions using Sentence Transformers.
*   **Vector Storage:** Stores embeddings, documents (text/captions), and metadata in a persistent ChromaDB database.
*   **RAG Agent:** Implements a basic agent using Google's ADK Agents framework (powered by Gemini models) to query the ChromaDB database and synthesize answers based on retrieved context.

## Workflow

The process for building and using the RAG knowledge base involves several steps:

1.  **Document Preparation (External - Marker):**
    *   **Input:** Source PDF documents containing visual step-by-step guides (e.g., located in the `Dataset/Visual pdf` directory).
    *   **Processing:** The **Marker tool** converts each PDF into a dedicated output folder (typically placed within `./OUTPUT`).
    *   **Output:** Each Marker output folder contains:
        *   Markdown (`.md`) file: Extracted text content.
        *   JSON (`_meta.json`) file: Extracted metadata.
        *   Image files (`.png`, `.jpg`, etc.): Extracted images.

2.  **Data Ingestion & Processing (`vector_db.py`):**
    *   The script initializes the Sentence Transformer (embedding), BLIP (captioning) models, and connects to ChromaDB.
    *   **Important:** By default, this script *deletes and recreates* the ChromaDB collection (`my_documents_collection_standard`) on each run.
    *   It iterates through Marker's output folders in `./OUTPUT`.
    *   Reads Markdown text and splits it into chunks using LangChain.
    *   Loads metadata from the JSON file.
    *   Generates descriptive captions for all images within the folder using the **BLIP model**.
    *   Aggregates text chunks and image captions along with their metadata.

3.  **Vectorization & Storage (`vector_db.py`):**
    *   Uses the **Sentence Transformer model** to create vector embeddings for *both* text chunks and image captions.
    *   Upserts the embeddings, original text (chunk/caption), metadata, and unique IDs into the **ChromaDB** collection.

4.  **Querying (`agent.py`):**
    *   Initializes the Sentence Transformer model and connects to the existing ChromaDB collection.
    *   Defines a Google ADK Agent (`DocumentRAGAgent`) powered by a Gemini model.
    *   Provides the agent with a tool (`retrieve_document_chunks_tool`) that:
        *   Takes a user query.
        *   Embeds the query using the Sentence Transformer.
        *   Queries ChromaDB for relevant text chunks or image captions based on vector similarity.
    *   The agent uses the retrieved information (or lack thereof) to formulate an answer based on its instructions.

## Technology Stack

*   **Python 3.x**
*   **PDF Processing:** [Marker](https://github.com/VikParuchuri/marker)
*   **Vector Database:** [ChromaDB](https://www.trychroma.com/)
*   **Embeddings:** [Sentence Transformers](https://www.sbert.net/) (`paraphrase-multilingual-mpnet-base-v2`)
*   **Image Captioning:** [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) (using `Salesforce/blip-image-captioning-base`)
*   **Text Chunking:** [LangChain](https://python.langchain.com/docs/get_started/introduction)
*   **LLM Agent:** [Google AI Developer Kit (ADK) Agents](https://developers.google.com/google-ai/adk) (using `gemini-1.5-flash-latest` or similar - adjust `agent.py`)
*   **Image Handling:** Pillow
*   **Package Management:** uv / pip

## Dataset

The primary dataset consists of visually-oriented PDF files containing step-by-step guides with screenshots. A small, self-created sample dataset used for development can be found here:

*   [Dataset/Visual pdf](https://github.com/B4K2/RagSolver/tree/main/Dataset/Visual%20pdf)

This dataset is currently limited and serves as a proof-of-concept.

## Setup & Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/B4K2/RagSolver.git # Replace with your actual repo URL if different
    cd RagSolver
    ```

2.  **Create Virtual Environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    *   Ensure you have Python installed.
    *   Install Marker following its official instructions: [https://github.com/VikParuchuri/marker](https://github.com/VikParuchuri/marker) (Make sure the `marker` command is available in your PATH).
    *   Install Python packages using `uv` (recommended, based on `uv.lock`) or `pip`:
        ```bash
        # Using uv (recommended)
        uv pip sync

        # Or using pip (generate requirements.txt first if needed)
        # uv pip freeze > requirements.txt # If you want to generate requirements.txt
        # pip install -r requirements.txt
        ```

4.  **Environment Variables:**
    *   You may need to set up environment variables, especially for the Google ADK Agent. Create a `.env` file in the root directory:
        ```dotenv
        # Example .env file
        GOOGLE_API_KEY=YOUR_GOOGLE_AI_API_KEY
        ```
    *   The `agent.py` script might need modification to load this key (e.g., using `python-dotenv`).

5.  **CUDA Setup (Recommended):**
    *   For significantly better performance with embedding and captioning models, ensure you have an NVIDIA GPU and the appropriate CUDA toolkit installed. The scripts are configured to use `"cuda"` if available.

6.  **Model Downloads:**
    *   The Sentence Transformer and BLIP models will be downloaded automatically by the respective libraries on their first use. Ensure you have an internet connection.

## Usage

1.  **Prepare Data:**
    *   Place your source PDF files in a suitable input directory.
    *   Run the Marker tool on your PDFs to generate the output folders containing `.md`, `_meta.json`, and image files. Ensure these output folders are placed inside the `./OUTPUT` directory (or modify `OUTPUT_FOLDER_PATH` in `vector_db.py`).
    *   *(Alternatively, use the sample data provided in the dataset link).*

2.  **Create/Update Vector Database:**
    *   Run the vector database creation script. **Warning:** This script deletes the existing collection by default. Comment out the `client.delete_collection` line in `vector_db.py` if you want to add to the existing DB instead (be mindful of potential duplicates if run multiple times on the same data without `upsert`).
    ```bash
    python vector_db.py
    ```
    *   Monitor the output for processing details and embedding progress.

3.  **Run the RAG Agent:**
    *   Start the agent script:
    ```bash
    python agent.py
    ```
    *   The script will initialize models and connect to ChromaDB.
    *   Interact with the agent by typing questions in the terminal. The agent will use the `retrieve_document_chunks_tool` to query the database when necessary. Enter 'quit' to exit.

## Future Work & Improvements

*   **Refine Agent Instructions:** Improve the prompt engineering for `agent.py` for more accurate and reliable RAG behavior.
*   **Background Tasks:** Implement background processing (e.g., using Celery, RQ) for the web upload feature (`app.py`) to handle large files and prevent request timeouts.
*   **Error Handling:** Add more robust error handling and user feedback throughout the pipeline.
*   **Multi-Modal Models:** Experiment with true multi-modal embedding models (like CLIP) that can embed images directly, potentially improving retrieval based on visual content.
*   **UI Development:** Create a more polished user interface for interaction and document management.
*   **Duplicate Handling:** Implement more robust duplicate detection/handling if not relying solely on the default collection deletion behavior.
*   **Scalability:** Evaluate and potentially switch to more scalable vector database solutions if handling very large datasets.
*   **Marker Configuration:** Allow finer control over Marker parameters via configuration files or script arguments.
