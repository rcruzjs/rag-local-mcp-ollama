# Local NotebookLM RAG via MCP & Ollama

Pipeline local de Retrieval-Augmented Generation (RAG) construído sobre o protocolo **MCP 2.x (Model Context Protocol)** e **Ollama**, executando no Ubuntu/WSL2 com inferência local no **Qwen 2.5 7B** e embeddings semânticos com **Nomic Embed Text**.

## Arquitetura

- **Interface MCP**: `servidor_mcp.py` implementa `MCPServer` expondo ferramentas via transporte `stdio`.
- **Armazenamento Vetorial**: ChromaDB persistente com similaridade por cosseno.
- **Processamento Documental**: Leitura de Markdown, TXT e PDF (via PyMuPDF) com chunking em janela deslizante e overlap.
- **Cliente Interativo**: `cliente_pipeline.py` com streaming token a token, gerenciamento de histórico e comandos de controle.

## Pré-requisitos

- Ubuntu no WSL2 com suporte a GPU (NVIDIA CUDA)
- Python 3.12+
- Ollama instalado e em execução (`qwen2.5:7b` e `nomic-embed-text`)

## Instalação e Execução

1. Clone o repositório e crie o ambiente virtual:
   ```bash
   git clone <URL_DO_SEU_REPOSITORIO>
   cd <NOME_DO_REPOSITORIO>
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

Certifique-se de que os modelos estão disponíveis no Ollama:

Bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

Adicione documentos (.pdf, .md, .txt) no diretório cadernos/.

Execute o cliente de chat interativo:

Bash
python cliente_pipeline.py

Comandos do Terminal
/metricas: Exibe a distribuição de chunks e documentos indexados.

/reindex: Sincroniza dinamicamente novos arquivos adicionados à pasta cadernos/.

/limpar: Reinicia a memória da conversa mantendo o índice vetorial.

sair: Encerra a aplicação.
EOF
