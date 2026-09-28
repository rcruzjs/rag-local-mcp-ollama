from cliente_pipeline import HistoricoConversa, extrair_texto_tool


def test_historico_conversa_adicionar_e_limitar():
    historico = HistoricoConversa(max_turnos=2)
    # 2 turnos = 4 mensagens (2 user + 2 assistant)
    historico.adicionar("user", "Pergunta 1")
    historico.adicionar("assistant", "Resposta 1")
    historico.adicionar("user", "Pergunta 2")
    historico.adicionar("assistant", "Resposta 2")
    historico.adicionar("user", "Pergunta 3")
    historico.adicionar("assistant", "Resposta 3")

    msgs = historico.como_lista_dict()
    assert len(msgs) == 4
    assert msgs[0]["content"] == "Pergunta 2"
    assert msgs[-1]["content"] == "Resposta 3"

def test_historico_conversa_limpar():
    historico = HistoricoConversa(max_turnos=3)
    historico.adicionar("user", "Pergunta 1")
    historico.adicionar("assistant", "Resposta 1")
    historico.limpar()
    assert len(historico.como_lista_dict()) == 0

class FakeBloco:
    def __init__(self, tipo: str, texto: str):
        self.type = tipo
        self.text = texto

class FakeResultado:
    def __init__(self, blocos):
        self.content = blocos

def test_extrair_texto_tool():
    res = FakeResultado([
        FakeBloco("text", "Trecho 1"),
        FakeBloco("image", "imagem.png"),
        FakeBloco("text", "Trecho 2")
    ])
    resultado = extrair_texto_tool(res)
    assert resultado == "Trecho 1\n\nTrecho 2"