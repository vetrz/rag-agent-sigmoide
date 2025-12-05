from bs4 import BeautifulSoup
import os
import re
import sys

def cleaned_text(text: str) -> str:
    text = re.sub(r'C ∫ Σ', "", text)
    text = re.sub(r'>>>', "", text)
    text = re.sub(r'\s+', " ", text)
    text = re.sub(r'\s-\s', "", text)

    return text

def extract_text_from_svg(directory:str) -> str:
    index_data: int = next((i for i, path in enumerate(sys.path) if path.endswith('/agent')))
    complete_text: str = ""
    data_directory: str = sys.path[index_data]+directory

    for current_path, _, files in os.walk(data_directory):
        for filename in files:
            if filename.lower().endswith(".svg"): 
                complet_path = os.path.join(current_path, filename) 
                content_svg = ""
                try:
                    with open(complet_path, "r", encoding="utf-8") as f:
                        content_svg = f.read()
                    soup = BeautifulSoup(content_svg, "xml")

                    text_elements = soup.find_all("text")
                    for text_element in text_elements:
                        text = text_element.get_text()
                        text = cleaned_text(text)
                        complete_text += text + " "
                    complete_text += "\n"

                except Exception as e:
                        print(f"error when reading {complet_path}: {e}")  

    return complete_text