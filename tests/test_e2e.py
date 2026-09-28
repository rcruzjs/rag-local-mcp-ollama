from unittest.mock import patch

from servidor_mcp import consultar_caderno, obter_metricas, store


def test_obter_metricas_vazio():
    with patch.object(store, "count", return_value=0):
        res = obter_metricas()
        assert "Base vetorial vazia" in res

def test_obter_metricas_com_dados():
    with patch.object(store, "count", return_value=5), \
         patch.object(store, "get_stats", return_value={"nota1.md": 3, "nota2.pdf": 2}):
        res = obter_metricas()
        assert "Total de Chunks Indexados: 5" in res
        assert "Total de Arquivos Únicos: 2" in res
        assert "nota1.md: 3 chunks" in res

def test_consultar_caderno_vazio():
    res = consultar_caderno("")
    assert "A consulta não pode ser vazia" in res

def test_consultar_caderno_base_vazia():
    with patch.object(store, "count", return_value=0):
        res = consultar_caderno("qualquer coisa")
        assert "Nenhum documento encontrado" in res

@patch("servidor_mcp.gerar_embedding", return_value=[0.1, 0.2])
def test_consultar_caderno_sucesso(mock_embed):
    mock_query_res = {
        "documents": [["Conteúdo relevante da nota."]],
        "metadatas": [[{"fonte": "exemplo.md", "pagina": 1}]],
        "distances": [[0.1]],  # score = 1.0 - 0.1 = 0.90
    }
    with patch.object(store, "count", return_value=1), \
         patch.object(store, "query", return_value=mock_query_res):
        res = consultar_caderno("teste")
        assert "exemplo.md" in res
        assert "Relevância: 0.90" in res
        assert "Conteúdo relevante da nota" in res