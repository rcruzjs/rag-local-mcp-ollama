"""
servidor_mcp.py

Servidor MCP para RAG sobre cadernos locais (estilo NotebookLM).

Expõe tools para:
- Consulta semântica em documentos (Markdown, TXT e PDF)
- Métricas da base vetorial
- Reindexação forçada com ingestão incremental em lote (batching)

Usa ChromaDB + embeddings via Ollama (nomic-embed-text por padrão).
"""

from __future__ import annotations

import glob
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import ollama
import pymupdf
from mcp.server.mcpserver import MCPServer
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Configurações do servidor RAG."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    diretorio_cadernos: Path = Path("cadernos")
    caminho_db: Path = Path("chroma_db")
    modelo_embedding: str = "nomic-embed-text"
    chunk_size: int = 300
    overlap: int = 60
    min_score: float = 0.30
    top_k_default: int = 3
    nome_collection: str = "cadernos_conhecimento"
    batch_size: int = 32


settings = Settings()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("servidor_mcp")

# ---------------------------------------------------------------------------
# Modelos de dados
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DocumentPart:
    """Parte de um documento (página ou arquivo inteiro)."""

    texto: str
    fonte: str
    pagina: int


@dataclass(frozen=True, slots=True)
class SyncResult:
    """Resultado de uma sincronização."""

    novos: int
    total: int
    arquivos: int


# ---------------------------------------------------------------------------
# Utilitários de chunking e extração
# ---------------------------------------------------------------------------

def chunk_recursivo(
    texto: str,
    chunk_size: int = 300,
    overlap: int = 60,
) -> list[str]:
    """Divide o texto recursivamente utilizando separadores sintáticos hierárquicos.

    Preserva a estrutura de parágrafos, frases e palavras.

    Args:
        texto: Texto completo a ser fragmentado.
        chunk_size: Número máximo aproximado de palavras por chunk.
        overlap: Número de palavras de sobreposição entre chunks consecutivos.

    Returns:
        Lista de chunks (strings).

    Raises:
        ValueError: Se chunk_size <= overlap ou valores inválidos.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size deve ser maior que zero")
    if overlap < 0:
        raise ValueError("overlap não pode ser negativo")
    if chunk_size <= overlap:
        raise ValueError("chunk_size deve ser maior que overlap")

    texto_limpo = texto.strip()
    if not texto_limpo:
        return []

    separadores = ["\n\n", "\n", ". ", " "]

    def dividir_texto(t: str, idx_sep: int) -> list[str]:
        if not t.strip():
            return []
        palavras = t.split()
        if len(palavras) <= chunk_size or idx_sep >= len(separadores):
            return [t.strip()]

        sep = separadores[idx_sep]
        partes = t.split(sep)
        segmentos: list[str] = []
        for p in partes:
            if p.strip():
                segmentos.extend(dividir_texto(p.strip(), idx_sep + 1))
        return segmentos

    segmentos = dividir_texto(texto_limpo, 0)
    chunks: list[str] = []
    palavras_atuais: list[str] = []

    for seg in segmentos:
        words_seg = seg.split()
        if len(palavras_atuais) + len(words_seg) <= chunk_size:
            palavras_atuais.extend(words_seg)
        else:
            if palavras_atuais:
                chunks.append(" ".join(palavras_atuais))
                palavras_atuais = palavras_atuais[-overlap:] if overlap > 0 else []
            palavras_atuais.extend(words_seg)

    if palavras_atuais:
        chunks.append(" ".join(palavras_atuais))

    return chunks


def chunk_com_overlap(
    texto: str,
    chunk_size: int = 300,
    overlap: int = 60,
) -> list[str]:
    """Função legada mantida para compatibilidade, utilizando chunk_recursivo."""
    return chunk_recursivo(texto, chunk_size=chunk_size, overlap=overlap)


def extrair_texto_arquivo(caminho: Path) -> list[DocumentPart]:
    """Extrai texto de um arquivo suportado (.md, .txt ou .pdf).

    Args:
        caminho: Caminho absoluto ou relativo do arquivo.

    Returns:
        Lista de DocumentPart (uma por página no caso de PDF).

    Raises:
        ValueError: Se a extensão não for suportada.
        OSError: Em caso de erro de leitura.
    """
    ext = caminho.suffix.lower()
    nome_arquivo = caminho.name
    documentos: list[DocumentPart] = []

    if ext in {".md", ".txt"}:
        try:
            conteudo = caminho.read_text(encoding="utf-8", errors="replace")
            if conteudo.strip():
                documentos.append(
                    DocumentPart(texto=conteudo, fonte=nome_arquivo, pagina=1)
                )
        except OSError as exc:
            logger.error("Falha ao ler arquivo de texto %s: %s", caminho, exc)
            raise

    elif ext == ".pdf":
        try:
            with pymupdf.open(caminho) as doc:
                for num_pag, pagina in enumerate(doc, start=1):
                    texto_pag = pagina.get_text().strip()
                    if texto_pag:
                        documentos.append(
                            DocumentPart(
                                texto=texto_pag,
                                fonte=nome_arquivo,
                                pagina=num_pag,
                            )
                        )
        except Exception as exc:
            logger.error("Falha ao processar PDF %s: %s", caminho, exc)
            raise

    else:
        raise ValueError(f"Extensão não suportada: {ext}")

    return documentos


# ---------------------------------------------------------------------------
# Camada de Vector Store
# ---------------------------------------------------------------------------

class VectorStore:
    """Abstração sobre o ChromaDB para indexação e consulta."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = chromadb.PersistentClient(path=str(settings.caminho_db))
        self._collection = self._client.get_or_create_collection(
            name=settings.nome_collection,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Coleção '%s' pronta. Total de chunks: %d",
            settings.nome_collection,
            self.count(),
        )

    def count(self) -> int:
        """Retorna o número total de chunks indexados."""
        return self._collection.count()

    def get_existing_ids(self) -> set[str]:
        """Retorna o conjunto de IDs já existentes na coleção."""
        if self.count() == 0:
            return set()
        dados = self._collection.get(include=[])
        return set(dados.get("ids") or [])

    def add_chunks(
        self,
        ids: list[str],
        documentos: list[str],
        metadados: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        """Adiciona um lote de chunks à base vetorial."""
        if not ids:
            return

        self._collection.add(
            ids=ids,
            documents=documentos,
            embeddings=embeddings,
            metadatas=metadados,
        )
        logger.info("Adicionados %d novos chunks ao ChromaDB", len(ids))

    def query(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> dict[str, Any]:
        """Executa busca por similaridade."""
        n_results = min(top_k, self.count())
        if n_results == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        return self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )

    def get_stats(self) -> dict[str, int]:
        """Retorna estatísticas de distribuição por arquivo."""
        if self.count() == 0:
            return {}

        dados = self._collection.get(include=["metadatas"])
        fontes: dict[str, int] = {}

        for meta in dados.get("metadatas") or []:
            fonte = meta.get("fonte", "desconhecido")
            fontes[fonte] = fontes.get(fonte, 0) + 1

        return fontes


# ---------------------------------------------------------------------------
# Sincronização e Embeddings (com suporte a Batching)
# ---------------------------------------------------------------------------

def gerar_embedding(texto: str, modelo: str) -> list[float]:
    """Gera embedding para um único texto via Ollama.

    Raises:
        RuntimeError: Se a chamada ao Ollama falhar.
    """
    try:
        resposta = ollama.embeddings(model=modelo, prompt=texto)
        return resposta["embedding"]
    except Exception as exc:
        logger.error("Falha ao gerar embedding individual: %s", exc)
        raise RuntimeError(f"Erro ao gerar embedding: {exc}") from exc


def gerar_embeddings_batch(textos: list[str], modelo: str) -> list[list[float]]:
    """Gera embeddings em lote via API do Ollama.

    Args:
        textos: Lista de textos para gerar embeddings.
        modelo: Nome do modelo de embeddings.

    Returns:
        Lista de vetores (embeddings).

    Raises:
        RuntimeError: Se a chamada em lote falhar.
    """
    if not textos:
        return []

    try:
        res = ollama.embed(model=modelo, input=textos)
        vetores = res.get("embeddings") or getattr(res, "embeddings", None)
        if vetores and len(vetores) == len(textos):
            return list(vetores)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Falha na chamada batch ollama.embed (%s) — executando fallback individual", exc)

    # Fallback sequencial se a API embed em lote falhar
    vetores_fallback: list[list[float]] = []
    for t in textos:
        vetores_fallback.append(gerar_embedding(t, modelo))
    return vetores_fallback


def executar_sincronizacao(store: VectorStore, settings: Settings) -> SyncResult:
    """Varre o diretório de cadernos e indexa apenas chunks novos usando processamento em lote.

    Args:
        store: Instância do VectorStore.
        settings: Configurações da aplicação.

    Returns:
        SyncResult com quantidades de novos chunks, total e arquivos processados.
    """
    diretorio = settings.diretorio_cadernos
    if not diretorio.exists():
        logger.warning("Diretório de cadernos não existe: %s", diretorio)
        return SyncResult(novos=0, total=store.count(), arquivos=0)

    extensoes = ("*.md", "*.txt", "*.pdf")
    arquivos: list[Path] = []
    for padrao in extensoes:
        arquivos.extend(Path(p) for p in glob.glob(str(diretorio / padrao)))

    if not arquivos:
        logger.info("Nenhum arquivo encontrado em %s", diretorio)
        return SyncResult(novos=0, total=store.count(), arquivos=0)

    registros_existentes = store.get_existing_ids()
    logger.info(
        "Iniciando sincronização. %d arquivos | %d chunks já indexados",
        len(arquivos),
        len(registros_existentes),
    )

    pedacos_pendentes: list[tuple[str, str, dict[str, Any]]] = []

    for caminho in arquivos:
        try:
            partes = extrair_texto_arquivo(caminho)
        except Exception:
            logger.exception("Pulando arquivo com erro: %s", caminho)
            continue

        for parte in partes:
            pedacos = chunk_recursivo(
                parte.texto,
                chunk_size=settings.chunk_size,
                overlap=settings.overlap,
            )

            for idx, pedaco in enumerate(pedacos):
                chunk_id = f"{parte.fonte}_p{parte.pagina}_chunk{idx}"
                if chunk_id in registros_existentes:
                    continue

                metadados = {
                    "fonte": parte.fonte,
                    "pagina": parte.pagina,
                    "chunk": idx,
                }
                pedacos_pendentes.append((chunk_id, pedaco, metadados))

    if not pedacos_pendentes:
        logger.info("Nenhum novo chunk para indexar.")
        return SyncResult(novos=0, total=store.count(), arquivos=len(arquivos))

    logger.info("Processando %d novos chunks em lotes de %d...", len(pedacos_pendentes), settings.batch_size)

    novos_ids: list[str] = []
    novos_documentos: list[str] = []
    novos_metadados: list[dict[str, Any]] = []
    novos_embeddings: list[list[float]] = []

    batch_size = max(1, settings.batch_size)
    for i in range(0, len(pedacos_pendentes), batch_size):
        lote = pedacos_pendentes[i : i + batch_size]
        lote_ids = [item[0] for item in lote]
        lote_docs = [item[1] for item in lote]
        lote_metas = [item[2] for item in lote]

        try:
            vetores = gerar_embeddings_batch(lote_docs, settings.modelo_embedding)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Erro ao gerar embeddings para lote %d-%d: %s", i, i + len(lote), exc)
            continue

        novos_ids.extend(lote_ids)
        novos_documentos.extend(lote_docs)
        novos_metadados.extend(lote_metas)
        novos_embeddings.extend(vetores)

    if novos_ids:
        store.add_chunks(
            ids=novos_ids,
            documentos=novos_documentos,
            metadados=novos_metadados,
            embeddings=novos_embeddings,
        )

    resultado = SyncResult(
        novos=len(novos_ids),
        total=store.count(),
        arquivos=len(arquivos),
    )
    logger.info(
        "Sincronização concluída: %d novos | total %d | %d arquivos",
        resultado.novos,
        resultado.total,
        resultado.arquivos,
    )
    return resultado


# ---------------------------------------------------------------------------
# Servidor MCP
# ---------------------------------------------------------------------------

mcp = MCPServer("NotebookLM-RAG-Advanced")
store = VectorStore(settings)


@mcp.tool()
def consultar_caderno(query: str, top_k: int | None = None) -> str:
    """Busca trechos semânticos mais relevantes para a consulta.

    Args:
        query: Texto da consulta do usuário.
        top_k: Quantidade máxima de resultados (padrão definido nas settings).

    Returns:
        Trechos formatados com fonte, página e score de relevância,
        ou mensagem informativa se nada for encontrado.
    """
    if not query or not query.strip():
        return "A consulta não pode ser vazia."

    top_k = top_k or settings.top_k_default

    if store.count() == 0:
        return "Nenhum documento encontrado na base vetorial."

    try:
        vetor_query = gerar_embedding(query, settings.modelo_embedding)
    except RuntimeError as exc:
        return f"Erro ao processar a consulta: {exc}"

    resultados = store.query(vetor_query, top_k)

    docs = resultados.get("documents", [[]])[0]
    metas = resultados.get("metadatas", [[]])[0]
    distancias = resultados.get("distances", [[]])[0]

    trechos_formatados: list[str] = []

    for doc, meta, dist in zip(docs, metas, distancias, strict=False):
        score = 1.0 - float(dist)
        if score > settings.min_score:
            trechos_formatados.append(
                f"[Fonte: {meta.get('fonte', '?')} | "
                f"Página: {meta.get('pagina', '?')} | "
                f"Relevância: {score:.2f}]\n{doc}"
            )

    if not trechos_formatados:
        return "Nenhum conteúdo com aderência semântica suficiente foi localizado."

    return "\n\n---\n\n".join(trechos_formatados)


@mcp.tool()
def obter_metricas() -> str:
    """Retorna estatísticas detalhadas sobre os documentos e chunks indexados."""
    total_chunks = store.count()

    if total_chunks == 0:
        return "Base vetorial vazia."

    fontes = store.get_stats()

    linhas = [
        f"Total de Chunks Indexados: {total_chunks}",
        f"Total de Arquivos Únicos: {len(fontes)}",
        "",
        "Distribuição por arquivo:",
    ]

    for arquivo, qtd in sorted(fontes.items()):
        linhas.append(f"  - {arquivo}: {qtd} chunks")

    return "\n".join(linhas)


@mcp.tool()
def reindexar_cadernos() -> str:
    """Executa varredura forçada na pasta de cadernos e indexa novos arquivos."""
    try:
        res = executar_sincronizacao(store, settings)
        return (
            f"Sincronização concluída! "
            f"{res.novos} novos chunks inseridos. "
            f"Total atual: {res.total} chunks de {res.arquivos} arquivos."
        )
    except Exception as exc:
        logger.exception("Erro durante reindexação")
        return f"Erro durante a sincronização: {exc}"


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Iniciando indexação inicial...")
    try:
        executar_sincronizacao(store, settings)
    except Exception:
        logger.exception("Falha na indexação inicial — servidor iniciará mesmo assim")

    logger.info("Servidor MCP iniciando...")
    mcp.run()