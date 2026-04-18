import os
import spacy
from itertools import combinations
from collections import Counter

# Carregar o modelo do spaCy
try:
    nlp = spacy.load("en_core_web_trf")
except OSError:
    print("Modelo do spaCy não encontrado. Execute 'python -m spacy download en_core_web_trf'")
    exit()

def is_valid_entity(ent):
    """
    Filtra entidades inválidas ou irrelevantes apenas pelo tamanho/formatação.
    """
    text = ent.text.strip()
    if len(text) < 2:
        return False
        
    return True

def standardize_entity(ent):
    """
    Padronização central do trabalho. 
    Lematiza e converte para minúsculo para unificar nós no grafo.
    """
    return ent.lemma_.lower().strip()

def process_sentence_window(text):
    """
    JANELA DE SENTENÇA:
    Extrai coocorrências se as entidades aparecem na mesma frase gramatical.
    """
    doc = nlp(text)
    cooccurrences = Counter()
    
    for sent in doc.sents:
        entities = [standardize_entity(ent) for ent in sent.ents if is_valid_entity(ent)]
        unique_entities = sorted(list(set(entities)))
        if len(unique_entities) > 1:
            cooccurrences.update(combinations(unique_entities, 2))
            
    return cooccurrences

def process_paragraph_window(text):
    """
    JANELA DE PARÁGRAFO:
    Lê os parágrafos já estruturados no text_processing (separados por \\n\\n).
    Gera coocorrências se as entidades estão no mesmo parágrafo.
    """
    cooccurrences = Counter()
    paragraphs = text.split("\n\n")
    
    for para in paragraphs:
        doc = nlp(para)
        entities = [standardize_entity(ent) for ent in doc.ents if is_valid_entity(ent)]
        unique_entities = sorted(list(set(entities)))
        
        if len(unique_entities) > 1:
            cooccurrences.update(combinations(unique_entities, 2))
            
    return cooccurrences

def process_sliding_window(text, k_words=1000):
    """
    JANELA DESLIZANTE DE K-PALAVRAS (Proximidade Matemática).
    """
    doc = nlp(text)
    cooccurrences = Counter()
    
    entity_positions = []
    for ent in doc.ents:
        if is_valid_entity(ent):
            idx = ent.start
            std_ent = standardize_entity(ent)
            entity_positions.append((idx, std_ent))
            
    for i in range(len(entity_positions)):
        for j in range(i + 1, len(entity_positions)):
            pos1, ent1 = entity_positions[i]
            pos2, ent2 = entity_positions[j]
            
            if abs(pos2 - pos1) <= k_words:
                if ent1 != ent2:
                    pair = tuple(sorted([ent1, ent2]))
                    cooccurrences[pair] += 1
            else:
                break
                
    return cooccurrences

