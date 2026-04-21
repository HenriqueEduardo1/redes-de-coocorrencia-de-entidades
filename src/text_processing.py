import os
import re
from deepmultilingualpunctuation import PunctuationModel

# Dicionário para correções manuais (Regex -> Substituição)
# Útil para corrigir erros comuns de transcrição do YouTube.
# Dicionário para correções manuais (Regex -> Substituição)
# Resolve a fragmentação de entidades gerada pela transcrição do áudio.
CUSTOM_CORRECTIONS = {
    # 1. Família ChatGPT e Erros Extremos de STT
    r'\b[Cc]hachi\s*[Pp][Tt]\s*(\d+)\b': r'ChatGPT \1',
    r'\b[Cc]hachi\s*[Pp][Tt]\b': 'ChatGPT',
    r'\b[Cc]hachi\s*[Pp]\b': 'ChatGPT',
    r'\b[Cc]hach[ty]?\b': 'ChatGPT',                 # Cobre "Chach", "Chacht", "Chachy"
    r'\b[Cc]hasht?\b': 'ChatGPT',                    # Cobre "Chash" e "Chasht"
    r'\b[Cc]hash\s*[Aa]pt\b': 'ChatGPT',
    r'\b[Cc]hash\s*[Pp][Tt]\w*\b': 'ChatGPT',
    r'\b[Cc]hashi\s*[Pp]t40\b': 'ChatGPT 4o',        # Corrige "Chashi Pt40"
    r'\b[Cc]hat\s*[Gg]pt\b': 'ChatGPT',
    r'\b[Cc][Hh]\s*[Gg][Pp][Tt]\b': 'ChatGPT',
    r'\bchpt\b': 'ChatGPT',
    r'\b[Cc]hbt\b': 'ChatGPT',                       # Nó "Chbt"

    # 2. Família GPT e Variações (Corrigindo o zero pelo "O")
    r'\b[Gg]bt\b': 'GPT',                            # Nó "Gbt"
    r'\b[Gg]bt2\b': 'GPT-2',                         # Nó "Gbt2"
    r'\b[Gg][Pp][Dd]\s*2\b|\b[Gg][Pp][Dd]2\b': 'GPT-2', 
    r'\b[Gg][Pp][Tt]\s*2\b|\b[Gg][Pp][Tt]2\b': 'GPT-2',
    r'\b[Gg][Pp][Tt]\s*3\b|\b[Gg][Pp][Tt]3\b': 'GPT-3',
    r'\b[Gg][Pp][Tt]\s*4\b|\b[Gg][Pp][Tt]4\b': 'GPT-4',
    r'\b[Gg][Pp][Tt]\s*40\b|\b[Gg][Pp][Tt]40\b': 'GPT-4o', # Transcrição leu "40" em vez de "4o"
    r'\b[Gg]pt-4\s*40\s*[Mm]ini\b': 'GPT-4o Mini',   # Nó "Gpt-4 40 Mini"
    r'\b[Gg][Pp][Tt]ini\b': 'Gemini',                
    r'\bgpt\b': 'GPT',
    r'\b[Oo]03\s*[Mm]ini\b': 'o3-mini',              # Nó "O03 Mini"

    # 3. Empresas, Laboratórios e Hardware
    r'\b[Oo]pen\s*[Aa]?[Ii]-?\b': 'OpenAI',          
    r'\b[Oo]peni\b': 'OpenAI',                       
    r'\b[Oo]pening\s*[Ee]y\b|\b[Oo]pena\b': 'OpenAI', # Corrigido "Opening Ey" e "Opena"
    r'\b[Aa]nthropic\b': 'Anthropic',            
    r'\b[Dd]eep\s*[Ss]eek\b': 'DeepSeek',            
    r'\b[Dd]eccom\b|\b[Dd]c\s*[Kk]ai\b': 'DeepSeek', # "Deccom" e "Dc Kai" no contexto do DeepSeek
    r'\b[Dd]eep\s*[Mm]ind\b': 'DeepMind',            
    r'\b[Gg]ooglecom\b': 'Google',                   
    r'\b[Hh]ugging\s*[Ff]ace\b|\b[Hh]uging\s*[Pp]hase\b': 'HuggingFace', # Corrigido "Huging Phase"
    r'\b[Hh]100[Ss]\b|\b[Hh]100\s*[Gg]pu-?\b': 'H100', # Limpa variações da GPU H100
    r'\b[Bb]f16\b': 'BF16',
    r'\b[Ff]p8\b': 'FP8',

    # 4. Conceitos Técnicos e Modelos
    r'\b[Aa]lpha[Oo]\b|\b[Aa]lphag\b': 'AlphaGo',    
    r'\b[Ff]ine\s*[Ww]eb\b': 'FineWeb',              
    r'\b[Ff]ine\s*[Tt]uning\b|\b[Ff]ine-[Tt]uning\b': 'Fine-Tuning', 
    r'\b[Rr]hf\b': 'RLHF',                           
    r'\b[Rr]einforcement\s*[Ll]earning\b': 'RL',     # Centraliza o nó gigante para "RL"
    r'\b[Cc]ommon\s*[Cc]raw\b|\b[Cc]ommon\s*[Cc][Ww]\b|\b[Cc][Ww]\b': 'Common Crawl',
    r'\b[Ll]m\s*[Ss]tudio\b': 'LM Studio',
    r'\b[Cc]ar1\b': 'R1',                            # STT transcreveu R1 como Car1

    # 5. Nomes Próprios do Mundo Real e Política
    r'\b[Ll]eis\s*[Dd]o\b|\b[Ll]isa\s*[Dd]ole\b': 'Lee Sedol', # Corrigido erro bizarro do AlphaGo
    r'\b[Jj]ane\s*[Aa]ustin\'[Ss]?\b': 'Jane Austen', 
    r'\b[Ss]pr\s*[Aa]nd\s*[Pp]rejudice\b': 'Pride and Prejudice',
    r'\b[Cc]amala\s*[Hh]arris\b': 'Kamala Harris',
    r'\b[Rr]onda\s*[Ss]antis\b': 'Ron DeSantis',
    r'\b[Tt]im\s*[Kk]ane\b': 'Tim Kaine',
    r'\b[Tt]om\s*[Cc]ruz\b': 'Tom Cruise',
    r'\b[Jj]ohn\s*[Bb]araso\b': 'John Barrasso',
    r'\b[Bb]uffalo\s*[Ss]avers\b': 'Buffalo Sabres',
    r'\b[Aa]llen[,]?\s*[Ii]nstitute\s*[Oo]f\b|\b[Aa]llen,\b': 'Allen Institute', 

    # 6. Forçando Capitalização para o spaCy (Para não criar nós vazados)
    r'\bllms?\b|\blms?\b': 'LLM',                    # Limpa "Llm", "Llms", "Lm", "LMS", "LM" para um só "LLM"
    r'\brl\b': 'RL',                                 
    r'\bterabyt\b': 'terabytes',
    r'\bai\b': 'AI',                                 # Evita o nó duplicado "Ai" vs "AI"
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