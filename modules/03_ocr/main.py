import os
from mistralai import Mistral
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from loguru import logger
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
load_dotenv()
mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.PersistentClient(path ="data")
ef = OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small")
chroma_client.delete_collection(name="laptop_recommendation")
collection = chroma_client.create_collection(name="laptop_recommendation", embedding_function=ef)


logger.info("uploaded pdf")
uploaded_pdf = mistral_client.files.upload(
    file = {
        "file_name": "laptop.pdf",
        "content": open("laptop.pdf", "rb")
    },
    purpose="ocr"
)

signed_url = mistral_client.files.get_signed_url(
    file_id = uploaded_pdf.id)

ocr_response = mistral_client.ocr.process(
    model="mistral-ocr-latest",
    document ={
        "type":"document_url",
        "document_url": signed_url.url,
    },
)

pages = ocr_response.model_dump().get("pages")
bullet_point = ""
for page_num, page in enumerate(pages[:10], start=1):
    markdown = page.get("markdown")


    SYSTEM_PROMPT = """
    You are a helpful assistant that extracts bullet points from the given text.
    You are data extractor.
    You will need to format the bullet points in markdown format.    
    OUTPUT FORMAT:
    -[point_1]
    -[point_2]
    -[point_3]
    -[point_4]
    -[point_5]

    IMPORTANT:
    - flat bullet point, no nested bullet point.
    - each billet point should be short and to the point.
    - do not add any headings or titles.
    - do not add any explanations
    - always include the number and date if present in the markdown.
    """
    logger.info(f"extracting bullet points from page {page_num}")
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract bullet points from the following text:\n{markdown}"}
        ]
    )
    result = response.choices[0].message.content
    bullet_point += result+ "\n\n"

collection.add(
    documents=[result],
    metadatas=[{"document_id": "id123"}],
    ids=[f"page_{page_num}"]
)

# with open("ocr_response.md","w")as f:
#     f.write(bullet_point)
