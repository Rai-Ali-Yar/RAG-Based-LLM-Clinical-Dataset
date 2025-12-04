# RAG-Based-LLM-Clinical-Dataset

# 🩺 AI Doctor: Diagnostic Reasoning with RAG & Llama-3

An AI-powered Clinical Diagnostic Assistant that uses **Retrieval-Augmented Generation (RAG)** to analyze patient symptoms and medical history. By retrieving similar past cases from the MIMIC-IV database and leveraging **Meta Llama-3**, this system provides grounded, evidence-based diagnostic reasoning.

## 🚀 Project Overview

This project simulates a clinical decision support system. Instead of relying solely on the internal knowledge of a Large Language Model (which can hallucinate), this system:
1.  **Retrieves** real, anonymized clinical notes from a vector database that match the current patient's symptoms.
2.  **Augments** the prompt with these "ground truth" similar cases.
3.  **Generates** a diagnosis, reasoning, and key findings using the Llama-3-8B model.

It includes a **Jupyter Notebook** for the backend pipeline and a **Streamlit Web App** for an interactive user interface.

## 🛠️ Tech Stack

* **LLM:** Meta Llama-3-8B-Instruct (4-bit Quantized via `bitsandbytes`)
* **Orchestration:** LangChain
* **Embeddings:** `sentence-transformers/all-mpnet-base-v2`
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Interface:** Streamlit (tunneled via Cloudflare for Colab access)
* **Environment:** Google Colab (T4 GPU recommended)

## 📂 Dataset Structure

The system is designed to ingest JSON-based clinical notes (derived from MIMIC-IV).
* **File Structure:** Folder names represent the **Diagnosis** (Ground Truth).
* **File Content:** JSON files with keys `input1` through `input9` representing different sections of a clinical note (Chief Complaint, History, Meds, Labs, etc.).

## ⚙️ Pipeline Architecture

1.  **Data Ingestion:** recursive loading of JSON files and reconstruction of full clinical texts.
2.  **Chunking:** Text is split using `RecursiveCharacterTextSplitter` (1000 char size, 200 overlap) to preserve context.
3.  **Indexing:** Chunks are embedded into vectors using HuggingFace Embeddings and stored in a local FAISS index.
4.  **Retrieval:** When a user queries symptoms, the system performs a similarity search to find the top 3 most relevant historical cases.
5.  **Generation:** A custom prompt injects the retrieved context into Llama-3 to generate a structured medical assessment.

## 📥 Installation & Setup

### Prerequisites
* A Google Colab account (or local GPU environment).
* A **Hugging Face Access Token** (with access granted to Llama-3 models).

### Steps
1.  **Clone the Repository** (or upload the notebook to Colab).
2.  **Install Dependencies:**
    ```python
    !pip install langchain langchain-community langchain-huggingface faiss-cpu sentence-transformers transformers accelerate bitsandbytes streamlit
    ```
3.  **Authenticate Hugging Face:**
    You must log in to download Llama-3.
    ```python
    from huggingface_hub import notebook_login
    notebook_login()
    ```
4.  **Run the Pipeline:** Execute the cells in the notebook to build the vector database.

## 🖥️ How to Run the App

The project includes a Streamlit app that runs directly from Google Colab using a Cloudflare tunnel.

1.  Run the final cell in the notebook labeled **"Run the App"**.
2.  Wait for the output to generate a URL ending in `.trycloudflare.com`.
3.  Click the link to open the AI Doctor interface in your browser.

## 📊 Evaluation (LLM-as-a-Judge)

The project includes an automated evaluation module. It uses the LLM itself to critique its own diagnoses based on three metrics:
* **Relevance:** Did it address the specific symptoms?
* **Coherence:** Is the logic sound?
* **Grounding:** Is the diagnosis supported by the retrieved context? (Crucial for medical safety).

## ⚠️ Medical Disclaimer

**This project is for educational and research purposes only.**
* It is **not** a licensed medical device.
* It should **not** be used for real-life medical diagnosis or treatment.
* The system analyzes anonymized historical data and may produce inaccuracies. Always consult a qualified healthcare professional for medical advice.

#
