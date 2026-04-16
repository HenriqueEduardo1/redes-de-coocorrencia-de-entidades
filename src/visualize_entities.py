import os
import spacy
from spacy import displacy

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clean_text_path = os.path.join(project_root, "data", "processed", "7xTGNNLPyMI_clean.txt")
    output_html_path = os.path.join(project_root, "assets", "entities_visualization.html")

    if not os.path.exists(clean_text_path):
        print(f"Arquivo limpo não encontrado: {clean_text_path}")
        return
    
    nlp = spacy.load("en_core_web_trf")
    
    with open(clean_text_path, 'r', encoding='utf-8') as f:
        text = f.read()

    doc = nlp(text)
    html = displacy.render(doc, style="ent", page=True)
    
    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
    with open(output_html_path, 'w', encoding='utf-8') as f:f.write(html)

if __name__ == "__main__":
    main()
