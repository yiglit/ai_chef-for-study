from langchain_ollama import OllamaEmbeddings
def get_embeddings()->OllamaEmbeddings:
    ollama = OllamaEmbeddings(
        model="modelscope.cn/Qwen/Qwen3-Embedding-0.6B-GGUF:latest",
        dimensions=1024
    )
    return ollama