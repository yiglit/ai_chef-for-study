from langchain_mineru import MinerULoader
from dotenv import load_dotenv
load_dotenv()
loader = MinerULoader(
    source="./resource/浙江风味2.pdf",
    mode="precision",
    ocr= True
)
docs = loader.load()
print(docs[0].metadata)
print(docs[0].page_content)
with open("./resource/output/r2.md", "w", encoding="utf-8") as f:
    f.write(docs[0].page_content)