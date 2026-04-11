import json
import os
import spacy
from itertools import combinations
from collections import Counter

# Carregar o modelo do spaCy (Inglês ou Português, dependendo do texto)
# Tente: python -m spacy download pt_core_news_lg (Para PT) ou en_core_web_lg (Para EN)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Modelo do spaCy não encontrado. Execute 'python -m spacy download en_core_web_sm'")
    exit()

def is_valid_entity(ent):
    """
    Filtra entidades inválidas ou irrelevantes.
    Focamos em entidades semânticas fortes e removemos números/datas.
    Para o modelo em Inglês, os labels relevantes são:
    PERSON, ORG, GPE (Países/Cidades), LOC, PRODUCT, WORK_OF_ART, EVENT, LANGUAGE.
    """
    valid_labels = [
        'PERSON', 'ORG', 'GPE', 'LOC', 
        'PRODUCT', 'WORK_OF_ART', 'EVENT', 'LANGUAGE'
    ]
    
    # Remove espaços vazios, pontuações isoladas e entidades muito curtas.
    text = ent.text.strip()
    if ent.label_ not in valid_labels:
        return False
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
        # Extrair e padronizar entidades válidas nesta sentença
        entities = [standardize_entity(ent) for ent in sent.ents if is_valid_entity(ent)]
        
        # Remover duplicatas na mesma sentença
        unique_entities = sorted(list(set(entities)))
        
        # Combinar par a par (arestas não-direcionadas)
        if len(unique_entities) > 1:
            cooccurrences.update(combinations(unique_entities, 2))
            
    return cooccurrences

def process_paragraph_window(blocks_json_path, block_duration_sec=60):
    """
    JANELA DE PARÁGRAFO / BLOCO TEMPORAL:
    Extrai coocorrências se as entidades estão no mesmo bloco de tempo (ex: a cada 60 segundos).
    O json fornecido tem o formato oriundo do youtube-transcript-api.
    """
    cooccurrences = Counter()
    
    with open(blocks_json_path, 'r', encoding='utf-8') as f:
        transcript = json.load(f)
        
    blocks = []
    current_block_text = []
    if not transcript: return cooccurrences
    
    current_block_start = transcript[0]['start']
    
    for segment in transcript:
        text = segment['text'].replace('\n', ' ').strip()
        start = segment['start']
        
        if start - current_block_start >= block_duration_sec:
            blocks.append(" ".join(current_block_text))
            current_block_text = [text]
            current_block_start = start
        else:
            current_block_text.append(text)
    
    if current_block_text:
        blocks.append(" ".join(current_block_text))
        
    for text in blocks:
        doc = nlp(text)
        
        entities = [standardize_entity(ent) for ent in doc.ents if is_valid_entity(ent)]
        unique_entities = sorted(list(set(entities)))
        
        if len(unique_entities) > 1:
            cooccurrences.update(combinations(unique_entities, 2))
            
    return cooccurrences

def process_sliding_window(text, k_words=50):
    """
    JANELA DESLIZANTE DE K-PALAVRAS (Proximidade Matemática):
    Duas entidades são conectadas se aparecem dentro de uma janela limite de K palavras/tokens.
    """
    doc = nlp(text)
    cooccurrences = Counter()
    
    # Extrair todos os tokens que são entidades e a posição deles (índice do token)
    entity_positions = []
    for ent in doc.ents:
        if is_valid_entity(ent):
            idx = ent.start  # Índice de início da entidade no documento
            std_ent = standardize_entity(ent)
            entity_positions.append((idx, std_ent))
            
    # Comparar distâncias par a par ao longo de todo o texto
    for i in range(len(entity_positions)):
        for j in range(i + 1, len(entity_positions)):
            pos1, ent1 = entity_positions[i]
            pos2, ent2 = entity_positions[j]
            
            # Se a distância entre o início de ent1 e ent2 for menor ou igual a K
            if abs(pos2 - pos1) <= k_words:
                if ent1 != ent2: # Não conectar a entidade com ela mesma
                    pair = tuple(sorted([ent1, ent2]))
                    cooccurrences[pair] += 1
            else:
                # Como a lista está ordenada por posição, se passar de K, os próximos também passarão
                break
                
    return cooccurrences

