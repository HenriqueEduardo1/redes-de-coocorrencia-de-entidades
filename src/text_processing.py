import os
import re
from deepmultilingualpunctuation import PunctuationModel

# Dicionário para correções manuais (Regex -> Substituição)
# Útil para corrigir erros comuns de transcrição do YouTube.
# Dicionário para correções manuais (Regex -> Substituição)
# Resolve a fragmentação de entidades gerada pela transcrição do áudio.
CUSTOM_CORRECTIONS = {
    # 1. Variações e erros de ChatGPT
    r'\b[Cc]hachi\s*[Pp][Tt]\b': 'ChatGPT',
    r'\b[Cc]hat\s*[Gg]pt\b': 'ChatGPT',
    r'\b[Cc][Hh]\s*[Gg][Pp][Tt]\b': 'ChatGPT',       # Corrige "CH GPT"
    r'\b[Cc]hash\s*[Aa]pt\b': 'ChatGPT',             # Corrige "Chash Apt"

    # 2. Variações e erros de OpenAI
    r'\b[Oo]pen\s*[Aa][Aa]?[Ii]-?\b': 'OpenAI',      # Corrige "Open Aai-" e "Open Ai"
    r'\b[Oo]peni\b': 'OpenAI',                       # Corrige "Openi"
    r'\b[Oo]pening\*s[Ee]y\b': 'OpenAI',             # Corrige "Opening*ey"

    # 3. Variações de Modelos GPT
    r'\b[Gg][Pp][Dd]\s*2\b|\b[Gg][Pp][Dd]2\b': 'GPT-2', # Corrige "Gpd2"
    r'\b[Gg][Pp][Tt]\s*2\b|\b[Gg][Pp][Tt]2\b': 'GPT-2', # Padroniza "Gpt2" e "GPT 2"
    r'\b[Gg][Pp][Tt]\s*4\b|\b[Gg][Pp][Tt]4\b': 'GPT-4', # Padroniza "GPT 4"
    r'\b[Gg][Pp][Tt]\s*3\b|\b[Gg][Pp][Tt]3\b': 'GPT-3', # Padroniza "Gpt3" e "GPT 3",

    # 4. Variações do Dataset Common Crawl
    r'\b[Cc]ommon\s*[Cc]raw\b': 'Common Crawl',      # Corrige "Common Craw"
    r'\b[Cc]ommon\s*[Cc]w\b': 'Common Crawl',        # Corrige "Common Cw"

    # 5. Agrupamento de Nomes Próprios
    # Junta as palavras para garantir que o spaCy entenda como uma entidade única
    r'\b[Ff]ine\s*[Ww]eb\b': 'FineWeb',              # Corrige "Fine Web"
    r'\b[Hh]ugging\s*[Ff]ace\b': 'HuggingFace',      # Corrige "Hugging Face"
    r'\b[Gg]ithub\b': 'GitHub',                      # Padronização de capitalização
}

punct_model = PunctuationModel()

def clean_text(text):
    """
    Higienização do texto pré-NER para melhorar a performance do modelo Transformer.
    - Remove pausas ou marcadores de fala ("uh", "um", "like", etc).
    - Trata pontuações e quebras isoladas do YouTube.
    - Formata espaçamentos para ajudar o Transformer a captar melhor o contexto.
    - Corrige termos específicos de domínio (ex: erros de transcrição).
    """
    if not text:
        return ""
    
    # 1. Remove stopwords de fala/hesitação comuns em oralidade
    # text = re.sub(r'\b(uh|um|hmm|ah|like|you know|so yeah|i mean)\b', '', text, flags=re.IGNORECASE)
    
    # 1.5. Aplica correções manuais de domínio
    for pattern, replacement in CUSTOM_CORRECTIONS.items():
        text = re.sub(pattern, replacement, text)
    
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
    print("  -> Rodando inferência para restaurar pontuação (isso pode levar alguns segundos)...")
    punctuated_text = punct_model.restore_punctuation(text)
    
    # 2. Separa o texto pontuado em frases reais baseando-se nos pontos finais, exclamações ou interrogações
    # O regex abaixo divide o texto mantendo a pontuação final na frase
    sentences = re.split(r'(?<=[.!?]) +', punctuated_text)
    
    paragraphs = []
    
    # 3. Agrupa um número fixo de FRASES REAIS para formar um parágrafo
    for i in range(0, len(sentences), sents_per_paragraph):
        chunk_sents = sentences[i:i + sents_per_paragraph]
        paragraph_text = " ".join(chunk_sents)
        paragraphs.append(paragraph_text)
        
    # Retorna o texto formatado com quebras de linha duplas entre os parágrafos
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
        # Pega todo o texto e remove quebras de linha pré-existentes
        raw_text = f.read().replace("\n", " ")

    print("Higienizando marcações de oralidade...")
    clean_content = clean_text(raw_text)

    print("Restaurando pontuação e estruturando parágrafos lógicos...")
    structured_content = structure_paragraphs(clean_content)

    os.makedirs(processed_dir, exist_ok=True)
    clean_path = os.path.join(processed_dir, f"{video_id}_clean.txt")
    
    with open(clean_path, 'w', encoding='utf-8') as f:
        f.write(structured_content)
        
    print(f"Sucesso! Texto limpo, pontuado e processado salvo em: {clean_path}")

if __name__ == "__main__":
    main()