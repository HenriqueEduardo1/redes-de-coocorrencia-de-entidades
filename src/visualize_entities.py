import os
import spacy
from spacy import displacy

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clean_text_path = os.path.join(
        project_root, "data", "processed", "7xTGNNLPyMI_clean.txt"
    )
    output_html_path = os.path.join(
        project_root, "assets", "visualize_entities.html"
    )

    if not os.path.exists(clean_text_path):
        print(f"Arquivo limpo não encontrado: {clean_text_path}")
        return

    # Adicionado bloco try-except para manter o padrão de segurança do script anterior
    try:
        print("Carregando o modelo Transformer...")
        nlp = spacy.load("en_core_web_trf")
    except OSError:
        print("Modelo do spaCy não encontrado. Execute 'python -m spacy download en_core_web_trf'")
        return

    print("Lendo arquivo de texto...")
    with open(clean_text_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Separa em parágrafos usando a quebra que criamos no text_processing.py
    paragraphs = text.split("\n\n")
    
    # Define um limite de parágrafos para não travar a RAM nem o Navegador
    # 50 parágrafos é um excelente tamanho para você validar a performance do modelo
    SAMPLE_SIZE = 50
    sample_paragraphs = paragraphs[:SAMPLE_SIZE]
    
    print(f"Processando amostra de {len(sample_paragraphs)} parágrafos para validação visual...")
    
    # Processa a amostra em lotes de forma eficiente
    docs = list(nlp.pipe(sample_paragraphs, disable=["textcat"]))

    print("Gerando HTML com displacy...")
    # O displacy.render aceita nativamente uma lista de objetos 'Doc'
    html = displacy.render(docs, style="ent", page=True)

    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html)
        
    print(f"Sucesso! Visualização HTML salva em: {output_html_path}")
    print("Abra o arquivo no seu navegador para verificar as marcações.")

if __name__ == "__main__":
    main()