# 📓 NotebookLM Local RAG via MCP & Ollama

Pipeline local de **Retrieval-Augmented Generation (RAG)** de alto desempenho, no estilo **NotebookLM**, construído sobre a arquitetura aberta **Model Context Protocol (MCP 2.x)** e inferência local via **Ollama**.

O projeto executa totalmente em ambiente **Ubuntu / WSL2**, garantindo **privacidade total** e **zero envio de dados externos**.

---

## 🌟 Principais Funcionalidades

- **Protocolo MCP 2.x**: Servidor e cliente integrados utilizando comunicação padronizada via `stdio`.
- **Inferência 100% Local**:
  - **Geração de Resposta**: LLM `qwen2.5:7b` via Ollama com respostas em *streaming* token a token.
  - **Embeddings Semânticos**: Modelo `nomic-embed-text` via Ollama para representação vetorial.
- **Armazenamento Vetorial Persistente**: Utiliza **ChromaDB** com cálculo de similaridade por cosseno.
- **Processamento Multi-Formato**: Suporte nativo para documentos `.pdf` (com metadados de página via PyMuPDF), `.md` e `.txt`.
- **Indexação Incremental Inteligente**: Chunking com janela deslizante e sobreposição (overlap). O sistema detecta e indexa apenas novos trechos sem duplicar dados.
- **Segurança & Resiliência**: System Prompt blindado contra ataques de *prompt injection* contidos nos documentos recuperados.
- **Interface CLI Interativa**: Terminal dinâmico com histórico de conversação mantido e comandos de controle (`/metricas`, `/reindex`, `/limpar`).

---

## 🏗️ Arquitetura do Sistema

```
                         +-----------------------------------+
                         |      Documentos Locais            |
                         |   (PDFs, Markdown, TXT)          |
                         +-----------------------------------+
                                           |
                                           v
                         +-----------------------------------+
                         |  Sincronização & Chunking         |
                         |  (PyMuPDF + Overlap de Palavras)  |
                         +-----------------------------------+
                                           |
                                           v
                         +-----------------------------------+
                         |  Ollama (nomic-embed-text)        |
                         |  Geração de Vetores Embeddings    |
                         +-----------------------------------+
                                           |
                                           v
                         +-----------------------------------+
                         |  ChromaDB Vector Store            |
                         |  (Persistência Local)             |
                         +-----------------------------------+
                                           ^
                                           | (Busca Semântica)
                                           v
+-----------------------+     stdio      +-----------------------------------+
|  Cliente CLI          | <------------> |  Servidor MCP                     |
|  (cliente_pipeline.py)|                |  (servidor_mcp.py)               |
+-----------------------+                +-----------------------------------+
            |
            v (Streaming LLM)
+-----------------------------------+
|  Ollama API (qwen2.5:7b)          |
+-----------------------------------+
```

---

## 📂 Estrutura do Projeto

```
projeto_mcp/
├── cadernos/              # Diretório para armazenamento dos documentos (.pdf, .md, .txt)
│   ├── .gitkeep
│   └── exemplo_nota.md    # Documento de exemplo fornecido
├── chroma_db/             # Base de dados vetorial persistente (gerado pelo ChromaDB)
├── cliente_pipeline.py    # Cliente CLI interativo (Sessão MCP + Ollama Streaming)
├── servidor_mcp.py       # Servidor MCP (Tools de busca semântica, métricas e sync)
├── requirements.txt       # Dependências Python do projeto
└── README.md              # Documentação principal do repositório
```

---

## 🛠️ Ferramentas MCP (Tools) Expostas

O servidor (`servidor_mcp.py`) expõe três ferramentas fundamentais baseadas no padrão MCP:

1. **`consultar_caderno`**:
   - **Descrição**: Realiza busca por similaridade semântica na base vetorial ChromaDB a partir de um texto de consulta.
   - **Parâmetros**: `query` (str), `top_k` (int, opcional).
   - **Retorno**: Trechos mais relevantes formatados com nome da fonte, número da página e score de relevância.

2. **`obter_metricas`**:
   - **Descrição**: Exibe estatísticas consolidadas do estado atual da base de conhecimento vetorial.
   - **Retorno**: Total de chunks indexados, total de arquivos únicos e distribuição de chunks por documento.

3. **`reindexar_cadernos`**:
   - **Descrição**: Força uma varredura no diretório `cadernos/` e realiza a ingestão e geração de embeddings de novos arquivos/chunks não indexados.

---

## 📋 Pré-requisitos

Antes de iniciar, certifique-se de ter os seguintes componentes instalados no seu ambiente **Ubuntu / WSL2**:

- **Python**: versão 3.10 ou superior.
- **Ollama**: instalado e em execução (`ollama serve`).
- **Modelos no Ollama**:
  ```bash
  ollama pull qwen2.5:7b
  ollama pull nomic-embed-text
  ```

---

## 🚀 Instalação e Execução

### 1. Clonar o repositório e acessar a pasta
```bash
cd /home/rcruz/projeto_mcp
```

### 2. Criar e ativar o ambiente virtual
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar as dependências
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Adicionar seus documentos
Insira os arquivos que deseja consultar (`.pdf`, `.md`, `.txt`) dentro da pasta `cadernos/`.

### 5. Iniciar o cliente de chat
```bash
python cliente_pipeline.py
```

---

## 💻 Comandos no Terminal Interativo

Durante a execução do `cliente_pipeline.py`, você pode utilizar os seguintes comandos especiais:

| Comando | Descrição |
| :--- | :--- |
| `/metricas` | Exibe estatísticas sobre o volume de chunks e arquivos indexados na base vetorial. |
| `/reindex` | Varre a pasta `cadernos/` e indexa novos documentos/chunks adicionados. |
| `/limpar` | Reinicia a memória da conversa (histórico de turnos), mantendo a base vetorial intacta. |
| `sair` | Encerra a sessão e o cliente interativo. |

---

## ⚙️ Configurações e Variáveis de Ambiente

As configurações utilizam `pydantic-settings` e podem ser sobrescritas por meio de um arquivo `.env` na raiz do projeto:

| Variável | Valor Padrão | Descrição |
| :--- | :--- | :--- |
| `DIRETORIO_CADERNOS` | `cadernos` | Caminho do diretório contendo os documentos a serem indexados. |
| `CAMINHO_DB` | `chroma_db` | Caminho de persistência da base vetorial ChromaDB. |
| `MODELO_EMBEDDING` | `nomic-embed-text` | Nome do modelo de embedding executado via Ollama. |
| `CHUNK_SIZE` | `300` | Tamanho (em palavras) de cada bloco fragmentado dos documentos. |
| `OVERLAP` | `60` | Quantidade de palavras de sobreposição entre chunks consecutivos. |
| `MIN_SCORE` | `0.30` | Pontuação mínima de relevância (1.0 - distância cosseno) para inclusão no contexto. |
| `TOP_K_DEFAULT` | `3` | Quantidade padrão de chunks recuperados por consulta. |
| `MODELO` | `qwen2.5:7b` | Nome do modelo LLM para geração da resposta interativa. |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Endereço do serviço local Ollama. |
| `MAX_HISTORICO_TURNOS` | `6` | Número de pares de mensagem (usuário/assistente) mantidos no histórico. |

---

## 🛡️ Segurança & Privacidade

- **Dados Locais**: Nenhum dado ou documento é enviado para APIs de terceiros na nuvem.
- **Resiliência a Injection**: O prompt do sistema contido no `cliente_pipeline.py` previne expressamente que instruções contidas em documentos indexados alterem o comportamento da LLM.

---

## 📄 Licença

Este projeto é disponibilizado para fins de estudo e desenvolvimento de pipelines RAG locais e extensões MCP.