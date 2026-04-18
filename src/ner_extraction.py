import os
import spacy
from itertools import combinations
from collections import Counter

# Carregar o modelo do spaCy (Certifique-se de ter feito o download do _trf se for usá-lo)
try:
    # Usando o trf para maior precisão, mas requer mais poder de processamento
    nlp = spacy.load("en_core_web_trf")
except OSError:
    print("Modelo do spaCy não encontrado. Execute '!python -m spacy download en_core_web_trf'")
    exit()

VALID_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "NORP", "FAC"}

def is_valid_entity(ent) -> bool:
    """Filtra entidades inválidas ou irrelevantes."""
    if ent.label_ not in VALID_ENTITY_LABELS:
        return False

    text = ent.text.strip()
    # Filtra ruídos comuns de 1 letra
    if len(text) < 2:
        return False
        
    return True

def standardize_entity(ent) -> str:
    """
    Padronização segura para NER.
    Evitamos lematizar nomes próprios para não distorcê-los.
    Mantemos siglas em maiúsculo e nomes em formato de título.
    """
    text = ent.text.strip()
    # Se for uma sigla (como AI, GPT, RL), mantém tudo maiúsculo
    if text.isupper():
        return text
    # Caso contrário, padroniza com a primeira letra maiúscula (ex: Google, Wikipedia)
    return text.title()

def process_sentence_window(text) -> Counter:
    """
    JANELA DE SENTENÇA:
    Otimizado para não estourar a memória (OOM) com modelos Transformer.
    Quebra o texto previamente usando a marcação de parágrafo dupla.
    """
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
    """
    JANELA DESLIZANTE (Proximidade Matemática).
    k_tokens ajustado para 30 (aprox. contexto de leitura imediata).
    """
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