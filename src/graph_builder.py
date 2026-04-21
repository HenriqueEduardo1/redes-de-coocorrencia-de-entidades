import os
import networkx as nx
from ner_extraction import process_sentence_window, process_paragraph_window, process_sliding_window

def build_cooccurrence_graph(cooccurrences):
    """
    Constrói um grafo não direcionado (nx.Graph) a partir das coocorrências extraídas.
    """
    G = nx.Graph()
    # Como o Counter do ner_extraction já somou todas as ocorrências de um mesmo par,
    # os pares aqui são únicos. Não precisamos do 'if G.has_edge', basta adicionar.
    for (ent1, ent2), weight in cooccurrences.items():
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

    # Foca no maior componente gigante (Giant Component) para o diâmetro
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
    Exporta o grafo para .graphml e imprime as métricas na tela.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    graphml_path = os.path.join(processed_dir, f"{video_id}_{window_name}.graphml")
    nx.write_graphml(G, graphml_path)
    
    # Imprime os resultados analíticos
    metrics = calculate_graph_metrics(G)
    print(f"\n--- Grafo: {window_name.upper()} ---")
    print(f"Nós: {metrics.get('num_nodes', 0)} | Arestas: {metrics.get('num_edges', 0)}")
    print(f"Densidade: {metrics.get('density', 0):.4f}")
    print(f"Nós no Componente Gigante: {metrics.get('giant_component_nodes', 0)}")
    print(f"Diâmetro do Componente Gigante: {metrics.get('diameter', 'N/A')}")
    print(f"-> Salvo em: {graphml_path}")

def main():
    video_id = "7xTGNNLPyMI"
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clean_text_path = os.path.join(project_root, "data", "processed", f"{video_id}_clean.txt")
    
    if not os.path.exists(clean_text_path):
        print(f"[ERRO] Arquivo limpo não encontrado: {clean_text_path}.")
        print("Certifique-se de executar 'text_processing.py' primeiro.")
        return
        
    print("Lendo o arquivo processado...")
    with open(clean_text_path, 'r', encoding='utf-8') as f:
        clean_text_content = f.read()

    print("\nExtraindo coocorrências via spaCy (Isso pode levar alguns minutos)...")
    sent_cooc = process_sentence_window(clean_text_content)
    para_cooc = process_paragraph_window(clean_text_content)
    
    # Correção: O parâmetro agora é k_tokens, alinhado com a nossa refatoração
    k_cooc = process_sliding_window(clean_text_content, k_tokens=50)
    
    print("\nConstruindo, analisando e exportando grafos...")
    
    # 1. Frase
    G_sent = build_cooccurrence_graph(sent_cooc)
    export_graph(G_sent, "sentenca", video_id)
    
    # 2. Parágrafo (Bloco Temporal)
    G_para = build_cooccurrence_graph(para_cooc)
    export_graph(G_para, "paragrafo", video_id)
    
    # 3. K-Tokens (K=50)
    G_k = build_cooccurrence_graph(k_cooc)
    export_graph(G_k, "k50_tokens", video_id)
    
    print("\nPipeline finalizado com sucesso!")
    
if __name__ == "__main__":
    main()