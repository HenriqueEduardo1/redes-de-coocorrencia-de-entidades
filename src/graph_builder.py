import os
import networkx as nx
from ner_extraction import process_sentence_window, process_paragraph_window, process_sliding_window

def build_cooccurrence_graph(cooccurrences):
    """
    Constrói um grafo não direcionado (nx.Graph) a partir das coocorrências extraídas.
    O peso de cada aresta será a frequência de coocorrência.
    """
    G = nx.Graph()
    for (ent1, ent2), weight in cooccurrences.items():
        if G.has_edge(ent1, ent2):
            G[ent1][ent2]['weight'] += weight
        else:
            G.add_edge(ent1, ent2, weight=weight)
    return G

def calculate_graph_metrics(G):
    """
    Calcula propriedades topológicas importantes do grafo.
    """
    if len(G.nodes) == 0:
        return {}

    metrics = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": nx.density(G),
    }

    # As vezes o grafo é desconectado (múltiplos  componentes).
    # Para calcular o diâmetro, precisamos focar no maior componente gigante (Giant Component).
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    if components:
        giant = G.subgraph(components[0])
        metrics["giant_component_nodes"] = giant.number_of_nodes()
        try:
            metrics["diameter"] = nx.diameter(giant)
        except nx.NetworkXError:
            metrics["diameter"] = "N/A"
    else:
        metrics["diameter"] = 0
        metrics["giant_component_nodes"] = 0

    return metrics

def export_graph(G, window_name, video_id):
    """
    Exporta o grafo para .graphml .
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    graphml_path = os.path.join(processed_dir, f"{video_id}_{window_name}.graphml")
    nx.write_graphml(G, graphml_path)
    print(f"[{window_name}] Grafo exportado: {graphml_path}")

def main():
    video_id = "7xTGNNLPyMI"
    
    # Agora o Graph Builder procura diretamente o texto já LIMPO e PROCESSADO
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clean_text_path = os.path.join(project_root, "data", "processed", f"{video_id}_clean.txt")
    
    if not os.path.exists(clean_text_path):
        print(f"Arquivo limpo não encontrado: {clean_text_path}.")
        print("Certifique-se de executar 'python src/text_processing.py' primeiro.")
        return
        
    with open(clean_text_path, 'r', encoding='utf-8') as f:
        clean_text_content = f.read()

    print("Extraindo coocorrências via spaCy (Isso pode levar alguns minutos)...")
    sent_cooc = process_sentence_window(clean_text_content)
    para_cooc = process_paragraph_window(clean_text_content)
    k_cooc = process_sliding_window(clean_text_content, k_words=50)
    
    print("Construindo e processando grafos...")
    
    # 1. Frase
    G_sent = build_cooccurrence_graph(sent_cooc)
    export_graph(G_sent, "sentenca", video_id)
    
    # 2. Parágrafo (Bloco Temporal)
    G_para = build_cooccurrence_graph(para_cooc)
    export_graph(G_para, "paragrafo", video_id)
    
    # 3. K-Palavras (K=50)
    G_k = build_cooccurrence_graph(k_cooc)
    export_graph(G_k, "k50_palavras", video_id)
    
if __name__ == "__main__":
    main()