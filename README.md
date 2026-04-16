<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/NetworkX-0052CC?style=for-the-badge&logo=python&logoColor=white" alt="NetworkX" />
  <img src="https://img.shields.io/badge/spaCy-09A3D5?style=for-the-badge&logo=spacy&logoColor=white" alt="spaCy" />
</p>

<h1 align="center">Redes de Coocorrência de Entidades em Videocasts</h1>

<p>
  Projeto de extração de Entidades Nomeadas (NER) e análise de grafos de coocorrência variando janelas de distância topológica. Desenvolvido para a Unidade 01 da disciplina de Algoritmos e Estruturas de Dados II.
</p>


## 1. Integrantes
A equipe de desenvolvimento é composta por:
- **HENRIQUE EDUARDO COSTA DA SILVA**
- **MURILO DE LIMA BARROS**
- **RAMON VINICIUS FERREIRA DE SOUZA**

---

## 2. Descrição das Atividades Realizadas

O sistema em Python executa um pipeline completo de Extração, Transformação e Carga (Modelagem de Redes) a partir de descrições orais não estruturadas de videocasts no YouTube.

### 2.1. Aquisição e Agregação de Dados

- **Fonte de Dados:** Transcrições oficiais do YouTube de episódios densos sobre inteligência artificial (Ex: Andrej Karpathy introduzindo LLMs).
- **Extração (`youtube-transcript-api`):** As transcrições nativas fornecem textos "picotados" a cada 3~5 segundos. Para mitigar esse ruído narrativo, implementamos um aglomerador linear que processa as listas em **blocos temporais de 60 segundos**.
- **Registro Físico:** Os dados brutos (e agregados) gerados são salvos em `/data/raw/` como `.txt` fluido e `.json` marcado por timestamps.

### 2.2. PNL e Reconhecimento de Entidades (`src/ner_extraction.py`)
- O processamento de linguagem natural é alimentado de ponta a ponta pela biblioteca **spaCy** (modelo `en_core_web_sm`).

### 2.3. As Três Janelas de Distância (`src/graph_builder.py`)
No escopo fundamental do trabalho analítico, extraímos parâmetros determinísticos de NetworkX operando definições variadas de topologia de vizinhança:
1. **Modelagem de Sentença:** Restrita; entidades apenas são conectadas se dividirem a mesma abstração frasal gramatical demarcada pelo parser do texto.
2. **Modelagem de Parágrafo:** Aglomerativa; cria coocorrências se entidades habitarem juntas os limites de blocos preestabelecidos de densidade descritiva (nossos blocos de 60s).
3. **Modelagem de K-Caracteres (K=50 Palavras):** Lógica espacial; independente de parágrafos, vizinhanças são traçadas mediante distância limítrofe no vetor.

As extrações resultam em arestas ponderadas exportadas nativamente para a expansão `/data/processed/*.graphml`.

