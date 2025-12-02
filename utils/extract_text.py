from bs4 import BeautifulSoup
import os
import re

def cleaned_text(text: str) -> str:
    text = re.sub(r'C ∫ Σ', "", text)
    text = re.sub(r'>>>', "", text)
    text = re.sub(r'\s-\s', "", text)
    text = re.sub(r'^\s*$\n', "", text, flags=re.MULTILINE)

    return text

def extract_text_from_svg() -> list:
    text: str = None
    complete_text: list = []
    root_directory = ".."

    for current_path, _, files in os.walk(root_directory):
        for filename in files:
            if filename.lower().endswith(".svg"):
                complet_path = os.path.join(current_path, filename)        
                content_svg = ""
                if  not "./trash/" in complet_path:
                    try:
                        with open(complet_path, "r", encoding="utf-8") as f:
                            content_svg = f.read()
                        soup = BeautifulSoup(content_svg, "xml")

                        text_elements = soup.find_all("text")
                        for text_element in text_elements:
                            text = text_element.get_text()
                            text = cleaned_text(text)
                            complete_text.append(text)

                    except Exception as e:
                        print(f"error when reading {complet_path}: {e}")
                        
    return complete_text