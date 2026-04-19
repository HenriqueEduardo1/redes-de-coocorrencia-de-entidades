import os
import re
from deepmultilingualpunctuation import PunctuationModel

# Dicionário para correções manuais (Regex -> Substituição)
# Útil para corrigir erros comuns de transcrição do YouTube.
# Dicionário para correções manuais (Regex -> Substituição)
# Resolve a fragmentação de entidades gerada pela transcrição do áudio.
CUSTOM_CORRECTIONS = {
    # 1. Família ChatGPT e Erros Grotescos de STT
    r'\b[Cc]hachi\s*[Pp][Tt]\s*(\d+)\b': r'ChatGPT \1',
    r'\b[Cc]hachi\s*[Pp][Tt]\b': 'ChatGPT',
    r'\b[Cc]hachi\s*[Pp]\b': 'ChatGPT',
    r'\b[Cc]hach[ty]?\b': 'ChatGPT',                 # Cobre "Chach", "Chacht", "Chachy"
    r'\b[Cc]hasht?\b': 'ChatGPT',                    # Cobre "Chash" e o nó "Chasht" do grafo
    r'\b[Cc]hash\s*[Aa]pt\b': 'ChatGPT',
    r'\b[Cc]hash\s*[Pp][Tt]\w*\b': 'ChatGPT',
    r'\b[Cc]hat\s*[Gg]pt\b': 'ChatGPT',
    r'\b[Cc][Hh]\s*[Gg][Pp][Tt]\b': 'ChatGPT',
    r'\bchpt\b': 'ChatGPT',
    r'\b[Cc]hbt\b': 'ChatGPT',                       # Nó "Chbt" solto no grafo

    # 2. Família GPT (Padronização de versões e capitalização)
    r'\b[Gg]bt\b': 'GPT',                            # Nó "Gbt" (áudio de GPT)
    r'\b[Gg][Pp][Dd]\s*2\b|\b[Gg][Pp][Dd]2\b': 'GPT-2', 
    r'\b[Gg][Pp][Tt]\s*2\b|\b[Gg][Pp][Tt]2\b': 'GPT-2',
    r'\b[Gg][Pp][Tt]\s*3\b|\b[Gg][Pp][Tt]3\b': 'GPT-3',
    r'\b[Gg][Pp][Tt]\s*4\b|\b[Gg][Pp][Tt]4\b': 'GPT-4',
    r'\b[Gg][Pp][Tt]\s*4[Oo]\b|\b[Gg][Pp][Tt]4[Oo]\b': 'GPT-4o', # Nó "GPT4O" / "Gpt4o"
    r'\b[Gg][Pp][Tt]ini\b': 'Gemini',                # Nó "GPTini" (confusão de áudio)
    r'\bgpt\b': 'GPT',                               # Força maiúscula para o standardize_entity

    # 3. Empresas e Concorrentes
    r'\b[Oo]pen\s*[Aa]?[Ii]-?\b': 'OpenAI',          
    r'\b[Oo]peni\b': 'OpenAI',                       
    r'\b[Oo]pening\s*[Ee]y\b': 'OpenAI',             # Corrigido o regex (antes com asterisco)
    r'\b[Aa]nthropic\b': 'Anthropic',            
    r'\b[Dd]eep\s*[Ss]eek\b': 'DeepSeek',            # Junta o nó "Deep Seek"
    r'\b[Dd]eep\s*[Mm]ind\b': 'DeepMind',            # Junta o nó "Deep Mind"
    r'\b[Gg]ooglecom\b': 'Google',                   # Nó "Googlecom"
    r'\b[Hh]ugging\s*[Ff]ace\b': 'HuggingFace',      

    # 4. Conceitos Técnicos e Modelos Específicos
    r'\b[Aa]lpha[Oo]\b|\b[Aa]lphag\b': 'AlphaGo',    # Nós "Alphao" e "Alphag" da imagem
    r'\b[Ff]ine\s*[Ww]eb\b': 'FineWeb',              
    r'\b[Ff]ine\s*[Tt]uning\b|\b[Ff]ine-[Tt]uning\b': 'Fine-Tuning', # Une "Fine Tuning" e "Fine-Tuning"
    r'\b[Rr]hf\b': 'RLHF',                           # Nó "Rhf" perto do cluster de RL
    r'\b[Rr]einforcement\s*[Ll]earning\b': 'RL',     # Opcional: Funde os dois nós gigantes
    r'\b[Cc]ommon\s*[Cc]raw\b|\b[Cc]ommon\s*[Cc][Ww]\b': 'Common Crawl',
    
    # 5. Forçando Capitalização de Siglas para o spaCy
    r'\bllm\b': 'LLM',                               # Força "LLM" maiúsculo para o seu padronizador
    r'\bllms\b': 'LLMs',                             # Mantém plural, mas capitaliza
    r'\brl\b': 'RL',                                 # Força "RL" maiúsculo
    r'\bterabyt\b': 'terabytes',

    # 6. Limpeza de Nomes Próprios Específicos do Grafo
    r'\b[Jj]ane\s*[Aa]ustin\'[Ss]?\b': 'Jane Austen', # Nó "Jane Austin'S"
    r'\b[Aa]llen[,]?\s*[Ii]nstitute\s*[Oo]f\s*[Aa]rtificial\s*[Ii]ntelligence\b': 'Allen Institute for AI', # Encurta o nó gigante
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