import os
import re

def clean_text(text):
    """
    Higienização do texto pré-NER para melhorar a performance do modelo Transformer.
    - Remove pausas ou marcadores de fala ("uh", "um", "like", etc).
    - Trata pontuações e quebras isoladas do YouTube.
    - Formata espaçamentos para ajudar o Transformer a captar melhor o contexto.
    """
    if not text:
        return ""
    
    # 1. Remove stopwords de fala/hesitação comuns em oralidade
    text = re.sub(r'\b(uh|um|hmm|ah|like|you know|so yeah|i mean)\b', '', text, flags=re.IGNORECASE)
    
    # 2. Consolida espaços e pontos duplicados
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\.{2,}', '.', text)
    
    # 3. Capitaliza o Início de Frases que foram quebradas se estiverem logo após pontuações
    text = re.sub(r'(?<=\. )([a-z])', lambda match: match.group(1).upper(), text)
    
    return text.strip()

def structure_paragraphs(text, words_per_sentence=20, sents_per_paragraph=5):
    """
    Como a transcrição bruta do YouTube não possui pontuação,
    o SBD (Sentence Boundary Detection) do spaCy falha em separar frases.
    Aqui estruturamos parágrafos agrupando um número fixo de palavras.
    """
    words = text.split()
    paragraphs = []
    
    words_per_paragraph = words_per_sentence * sents_per_paragraph
    
    for i in range(0, len(words), words_per_paragraph):
        chunk = words[i:i + words_per_paragraph]
        # Monta a "frase" com primeira letra maiúscula e ponto final
        paragraph_text = " ".join(chunk).capitalize() + "."
        paragraphs.append(paragraph_text)
        
    return "\n\n".join(paragraphs)

def main():
    video_id = "7xTGNNLPyMI"
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    raw_path = os.path.join(project_root, "data", "raw", f"{video_id}.txt")
    processed_dir = os.path.join(project_root, "data", "processed")
    
    if not os.path.exists(raw_path):
        print(f"Arquivo bruto não encontrado: {raw_path}")
        return

    print("Lendo arquivo bruto...")
    with open(raw_path, 'r', encoding='utf-8') as f:
        raw_text = f.read().replace("\n", " ")

    print("Higienizando marcações de oralidade e arrumando pontuações...")
    clean_content = clean_text(raw_text)

    print("Identificando limites de frases e estruturando parágrafos...")
    structured_content = structure_paragraphs(clean_content)

    os.makedirs(processed_dir, exist_ok=True)
    clean_path = os.path.join(processed_dir, f"{video_id}_clean.txt")
    
    with open(clean_path, 'w', encoding='utf-8') as f:
        f.write(structured_content)
        
    print(f"Sucesso! Texto limpo e processado salvo em: {clean_path}")

if __name__ == "__main__":
    main()