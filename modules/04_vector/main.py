import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import os
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool
import json
from loguru import logger


load_dotenv()

chroma_client = chromadb.PersistentClient(path ="data") 
ef = OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small")
collection = chroma_client.get_collection(name="laptop_recommendation", embedding_function=ef)

logger.info("start")
@function_tool
def query_collection(query_text: str, n_results: int = 5) -> str:
    """Get context from vector database for relevant information"""
    logger.info(f"Querying collection with query_text: {query_text} and n_results: {n_results}")
    result = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return json.dumps(result.get("documents")[0])

SYSTEM_PROMPT = """
    You are a helpful assistant that helps people find laptop based on a topic.
    You are assistant agent for answering questions about laptops.

    RULES:
    - Always use query_collection function to get context information about laptops.
    - If there is no relevant data, then say "I am sorry, I do not have that information right now".    
    - If there is relevent data, then use it to answer the question.    
"""
assistant_agent = Agent(
    name ="Laptop Recommender Agent",
    instructions= SYSTEM_PROMPT,
    model ="gpt-4.1",
    tools=[query_collection]
)
logger.info("mid")

async def main():
    messages = []

    while True:
        user_input = input("User: ")
        messages.append({"role": "user", "content": user_input})
        runner = await Runner.run(starting_agent=assistant_agent, input=messages)
        messages = runner.to_input_list()
    
        print(runner.last_agent.name)
        print(runner.final_output)
        print("===="*20)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
# results = collection.query(
#     query_texts=["gaming laptop with high refresh rate"],
#     n_results=3
#  )

# documents = results.get("documents")[0]
# print(documents)