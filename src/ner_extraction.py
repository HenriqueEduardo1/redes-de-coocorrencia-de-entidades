import spacy
from itertools import combinations
from collections import Counter

# 1. Carregar o modelo do spaCy
try:
    nlp = spacy.load("en_core_web_trf")
except OSError:
    print("Modelo do spaCy não encontrado. Execute '!python -m spacy download en_core_web_trf'")
    exit()

# 2. IMPLEMENTAÇÃO DO ENTITY RULER
# Criamos um pipeline de regras que roda ANTES do modelo estatístico (NER)
ruler = nlp.add_pipe("entity_ruler", before="ner")

# Definimos os padrões do nosso domínio técnico.
# Usar "LOWER" garante que a busca seja case-insensitive (pega "html", "HTML", "Html").
tech_patterns = [
    {"label": "TECH_CONCEPT", "pattern": [{"LOWER": "llm"}]},
    {"label": "TECH_CONCEPT", "pattern": [{"LOWER": "llms"}]},
    {"label": "TECH_CONCEPT", "pattern": [{"LOWER": "large"}, {"LOWER": "language"}, {"LOWER": "model"}]},
    {"label": "TECH_CONCEPT", "pattern": [{"LOWER": "html"}]},
    {"label": "TECH_CONCEPT", "pattern": [{"LOWER": "css"}]},
    {"label": "TECH_CONCEPT", "pattern": [{"LOWER": "url"}, {"LOWER": "filtering"}]},
    {"label": "TECH_CONCEPT", "pattern": [{"LOWER": "text"}, {"LOWER": "extraction"}]},
    {"label": "TECH_CONCEPT", "pattern": [{"LOWER": "language"}, {"LOWER": "classifier"}]},
    {"label": "TECH_CONCEPT", "pattern": [{"LOWER": "pre"}, {"TEXT": "-"}, {"LOWER": "training"}]},
    {"label": "TECH_CONCEPT", "pattern": [{"LOWER": "fine"}, {"TEXT": "-"}, {"LOWER": "tuning"}]},
    {"label": "TECH_CONCEPT", "pattern": [{"LOWER": "reinforcement"}, {"LOWER": "learning"}]},
    {"label": "TECH_CONCEPT", "pattern": [{"LOWER": "rl"}]},
    {"label": "TECH_CONCEPT", "pattern": [{"LOWER": "ai"}]},
    {"label": "TECH_CONCEPT", "pattern": [{"LOWER": "artificial"}, {"LOWER": "intelligence"}]}
]

# Adicionamos os padrões ao ruler
ruler.add_patterns(tech_patterns)

VALID_ENTITY_LABELS = {"PERSON", "ORG", "PRODUCT", "GPE", "LOC", "EVENT", "TECH_CONCEPT"}

def is_valid_entity(ent) -> bool:
    """Filtra entidades inválidas ou irrelevantes."""
    if ent.label_ not in VALID_ENTITY_LABELS:
        return False

    text = ent.text.strip()
    if len(text) < 2:
        return False
        
    return True

def standardize_entity(ent) -> str:
    """
    Padronização segura para NER.
    Mantemos siglas em maiúsculo e nomes em formato de título.
    A lógica lida perfeitamente com os nossos TECH_CONCEPTs.
    """
    text = ent.text.strip()
    
    # Se for uma sigla (como AI, GPT, RL, HTML, CSS), mantém tudo maiúsculo
    if text.isupper():
        return text
    
    # Se for "large language model", vira "Large Language Model"
    return text.title()

def process_sentence_window(text) -> Counter:
    """JANELA DE SENTENÇA"""
    cooccurrences = Counter()
    paragraphs = text.split("\n\n")
    
    # nlp.pipe é altamente eficiente para iterar sobre blocos de texto grandes
    for doc in nlp.pipe(paragraphs, disable=["textcat"]): 
        for sent in doc.sents:
            entities = [standardize_entity(ent) for ent in sent.ents if is_valid_entity(ent)]
            unique_entities = sorted(list(set(entities)))
            
            if len(unique_entities) > 1:
                cooccurrences.update(combinations(unique_entities, 2))
                
    return cooccurrences

def process_paragraph_window(text) -> Counter:
    """
    JANELA DE PARÁGRAFO:
    Gera coocorrências se as entidades estão no mesmo parágrafo (contexto mais amplo).
    """
    cooccurrences = Counter()
    paragraphs = text.split("\n\n")
    
    for doc in nlp.pipe(paragraphs, disable=["textcat"]):
        entities = [standardize_entity(ent) for ent in doc.ents if is_valid_entity(ent)]
        unique_entities = sorted(list(set(entities)))
        
        if len(unique_entities) > 1:
            cooccurrences.update(combinations(unique_entities, 2))
            
    return cooccurrences

def process_sliding_window(text, k_tokens=30) -> Counter:
    """JANELA DESLIZANTE"""
    cooccurrences = Counter()
    paragraphs = text.split("\n\n")
    
    # Precisamos manter o tracking absoluto de tokens ao redor dos blocos
    token_offset = 0 
    entity_positions = []
    
    for doc in nlp.pipe(paragraphs, disable=["textcat"]):
        for ent in doc.ents:
            if is_valid_entity(ent):
                # Guarda o index do token de forma contínua através dos parágrafos
                idx = ent.start + token_offset
                std_ent = standardize_entity(ent)
                entity_positions.append((idx, std_ent))
        token_offset += len(doc)
            
    # Processa as conexões pela distância k_tokens
    for i in range(len(entity_positions)):
        for j in range(i + 1, len(entity_positions)):
            pos1, ent1 = entity_positions[i]
            pos2, ent2 = entity_positions[j]
            
            if abs(pos2 - pos1) <= k_tokens:
                if ent1 != ent2:
                    pair = tuple(sorted([ent1, ent2]))
                    cooccurrences[pair] += 1
            else:
                # Como a lista é sequencial, se ultrapassou o k_tokens, 
                # as próximas entidades estarão ainda mais longe.
                break 
                
    return cooccurrences