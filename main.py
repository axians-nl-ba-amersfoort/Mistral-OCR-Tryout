import os
import io
import base64
import argparse
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from PIL import Image
import fitz

from mistralai import Mistral, DocumentURLChunk, TextChunk
from openai import AzureOpenAI


# ------------------------------
# Setup
# ------------------------------
load_dotenv(find_dotenv())

mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY_AXIANS"))
gpt_client = AzureOpenAI(
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-05-01-preview",
)
gpt_model = os.getenv("DEPLOYMENT_NAME")


# ------------------------------
# Mistral OCR
# ------------------------------
def extract_page_and_ocr(pdf_path: Path, page_number: int) -> str:
    doc = fitz.open(pdf_path)
    page_index = page_number - 1
    if page_index >= len(doc):
        raise IndexError(f"PDF has only {len(doc)} pages.")

    new_doc = fitz.open()
    new_doc.insert_pdf(doc, from_page=page_index, to_page=page_index)
    output_path = pdf_path.parent / f"page_{page_number}_only.pdf"
    new_doc.save(output_path)

    uploaded = mistral_client.files.upload(
        file={"file_name": output_path.name, "content": output_path.read_bytes()},
        purpose="ocr",
    )
    signed_url = mistral_client.files.get_signed_url(file_id=uploaded.id, expiry=1)

    result = mistral_client.ocr.process(
        document=DocumentURLChunk(document_url=signed_url.url),
        model="mistral-ocr-latest",
        include_image_base64=True,
    )

    return "\n\n".join(
        page.markdown for page in result.pages
    )


# ------------------------------
# GPT-4 Vision
# ------------------------------
def extract_pdf_page_as_base64_image(pdf_path: Path, page_number: int) -> str:
    doc = fitz.open(pdf_path)
    pix = doc[page_number - 1].get_pixmap(dpi=300)
    image = Image.open(io.BytesIO(pix.tobytes()))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def analyze_pdf_page_with_gpt_vision(pdf_path: Path, page_number: int, prompt: str) -> str:
    base64_image = extract_pdf_page_as_base64_image(pdf_path, page_number)

    response = gpt_client.chat.completions.create(
        model=gpt_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": "high"
                    }},
                ],
            }
        ],
        max_tokens=1500,
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()


# ------------------------------
# CLI Entry Point
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="OCR & GPT-4 Vision on PDF pages")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ocr_parser = subparsers.add_parser("ocr")
    ocr_parser.add_argument("--pdf", required=True, type=Path)
    ocr_parser.add_argument("--page", required=True, type=int)

    gpt_parser = subparsers.add_parser("gpt")
    gpt_parser.add_argument("--pdf", required=True, type=Path)
    gpt_parser.add_argument("--page", required=True, type=int)
    gpt_parser.add_argument("--prompt", required=True, type=str)

    args = parser.parse_args()

    if args.command == "ocr":
        result = extract_page_and_ocr(args.pdf, args.page)
    elif args.command == "gpt":
        result = analyze_pdf_page_with_gpt_vision(args.pdf, args.page, args.prompt)
    else:
        raise ValueError("Unknown command")

    print("\n" + "="*40 + "\n")
    print(result)
    print("\n" + "="*40 + "\n")


if __name__ == "__main__":
    main()
