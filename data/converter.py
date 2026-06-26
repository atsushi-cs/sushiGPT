from pypdf import PdfReader
from pathlib import Path

# converts pdf -> txt
DIRECTORY = "/Users/allantsay/Desktop/houston/llm/readings"

target_dir = Path(DIRECTORY)

for pdf_path in target_dir.glob("*.pdf"):
    reader = PdfReader(pdf_path)
    number_of_pages = len(reader.pages)
    output_path = pdf_path.with_suffix(".txt")
    with open(output_path, "w", encoding="utf-8") as file:
        for i in range(number_of_pages):
            page = reader.pages[i]
            text = page.extract_text()
            file.write(text)
            file.write("\n")