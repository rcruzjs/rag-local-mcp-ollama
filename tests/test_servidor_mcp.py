from unittest.mock import patch

import pytest

from servidor_mcp import (
    chunk_recursivo,
    extrair_texto_arquivo,
    gerar_embeddings_batch,
)


def test_chunk_recursivo_preserva_paragrafos():
    texto = "Parágrafo 1 com várias palavras de conteúdo importante.\n\nParágrafo 2 separado por nova linha dupla."
    # Com chunk_size=10, o primeiro parágrafo (8 palavras) e o segundo (7 palavras) não cabem juntos
    chunks = chunk_recursivo(texto, chunk_size=10, overlap=2)
    assert len(chunks) == 2
    assert "Parágrafo 1" in chunks[0]
    assert "Parágrafo 2" in chunks[1]

@patch("ollama.embed")
def test_gerar_embeddings_batch_sucesso(mock_embed):
    mock_embed.return_value = {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
    vetores = gerar_embeddings_batch(["texto 1", "texto 2"], "nomic-embed-text")
    assert len(vetores) == 2
    assert vetores[0] == [0.1, 0.2]
    mock_embed.assert_called_once_with(model="nomic-embed-text", input=["texto 1", "texto 2"])

@patch("ollama.embed", side_effect=Exception("API indisponível"))
@patch("servidor_mcp.gerar_embedding")
def test_gerar_embeddings_batch_fallback(mock_single, mock_embed):
    mock_single.return_value = [0.5, 0.6]
    vetores = gerar_embeddings_batch(["texto 1"], "nomic-embed-text")
    assert len(vetores) == 1
    assert vetores[0] == [0.5, 0.6]

def test_extrair_texto_arquivo_md_e_txt(tmp_path):
    md_file = tmp_path / "nota.md"
    md_file.write_text("# Conteúdo Markdown", encoding="utf-8")

    partes = extrair_texto_arquivo(md_file)
    assert len(partes) == 1
    assert partes[0].fonte == "nota.md"
    assert partes[0].pagina == 1
    assert "Conteúdo Markdown" in partes[0].texto

def test_extrair_texto_arquivo_nao_suportado(tmp_path):
    invalid_file = tmp_path / "foto.png"
    invalid_file.write_text("binario", encoding="utf-8")

    with pytest.raises(ValueError, match="Extensão não suportada"):
        extrair_texto_arquivo(invalid_file)