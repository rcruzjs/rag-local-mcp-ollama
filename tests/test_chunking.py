import pytest

from servidor_mcp import chunk_com_overlap


def test_chunk_com_overlap_basico():
    texto = "palavra " * 500
    chunks = chunk_com_overlap(texto, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert len(chunks[0].split()) == 100

def test_chunk_com_overlap_texto_curto():
    texto = "um texto curto com poucas palavras"
    chunks = chunk_com_overlap(texto, chunk_size=100, overlap=20)
    assert len(chunks) == 1
    assert chunks[0] == texto

def test_chunk_com_overlap_texto_vazio():
    assert chunk_com_overlap("", chunk_size=100, overlap=20) == []

def test_chunk_com_overlap_parametros_invalidos():
    with pytest.raises(ValueError):
        chunk_com_overlap("teste", chunk_size=0, overlap=10)

    with pytest.raises(ValueError):
        chunk_com_overlap("teste", chunk_size=50, overlap=50)

    with pytest.raises(ValueError):
        chunk_com_overlap("teste", chunk_size=50, overlap=-5)