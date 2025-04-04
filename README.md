# Testing Mistral OCR 2503 Model

## Project Summary

This project aims to test the newly launched Mistral OCR 2503 model, which boasts strong character recognition capabilities with an in-built LLM base. The model first understands the content of the uploaded document and then extracts the information.

## File Structure

- `data/`: Contains PDF files used for testing.
  - `example_1.pdf`
  - `example_2.pdf`
  - `example_3.pdf`
- `mistralai/`: Contains the Mistral client library.
- `notebook.ipynb`: Jupyter notebook demonstrating the usage of the Mistral OCR model.
- `README.md`: Project documentation.

## Usage Instructions

1. **Set up the environment**:
   - Ensure you have the necessary libraries installed (`os`, `json`, `io`, `base64`, `pandas`, `fitz`, `PIL`, `dotenv`, `IPython`, `mistralai`, `openai`).
   - Load environment variables using `dotenv`.

