# cliente_pipeline.py
import asyncio
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from ollama import AsyncClient

ollama_client = AsyncClient(host="http://127.0.0.1:11434")

server_params = StdioServerParameters(
    command=sys.executable,  # Usa o Python ativo do seu venv (gpu_env)
    args=["servidor_mcp.py"],
    env=None
)

async def executar_pipeline(pergunta: str, modelo: str = "qwen2.5:7b"):
    print(f"\n[1/3] Conectando ao servidor MCP...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print(f"[2/3] Consultando notas para: '{pergunta}'...")
            resultado_tool = await session.call_tool(
                "consultar_caderno",
                arguments={"query": pergunta}
            )

            contexto = "\n\n".join(
                item.text for item in resultado_tool.content if item.type == "text"
            )

            print(f"[3/3] Gerando resposta com {modelo}...\n")
            print("=" * 60)
            print("RESPOSTA REFINADA:")
            print("=" * 60)

            mensagens = [
                {
                    "role": "system",
                    "content": (
                        "Você é um assistente técnico especialista. "
                        "Sintetize e responda à pergunta do usuário baseando-se "
                        "exclusivamente nas informações fornecidas no contexto. "
                        "Seja objetivo e estruturado."
                    )
                },
                {
                    "role": "user",
                    "content": f"### CONTEXTO:\n{contexto}\n\n### PERGUNTA:\n{pergunta}"
                }
            ]

            # Streaming direto no terminal
            stream = await olama_client.chat(
                model=modelo,
                messages=mensagens,
                stream=True
            )

            async for chunk in stream:
                print(chunk["message"]["content"], end="", flush=True)
            print("\n" + "=" * 60)

if __name__ == "__main__":
    prompt = "Quais são as especificações do pipeline definidas nas notas?"
    asyncio.run(executar_pipeline(prompt))