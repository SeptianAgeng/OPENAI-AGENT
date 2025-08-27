from utils import openai_client

def laptop_recomendation(topic:str):
    SYSTEM_PROMPT = """
    You are a helpful assistant that helps people find laptop for the given topic.
    
    GUIDELINES: 
    - Provide 5 laptop with a short description for each laptop.
    - Ensure the laptop are relevant to the topic provided.
    - Format the response as a numbered list.
    - Use laptop_gaming_recomendation function to get laptop prices if needed.

    #Important: 
    - Please broadcast progress of each step using the broadcast function.
    - Broadcast messages should be concise and informative.
    - Always use function calls to accomplish tasks.
    """
    
    response = openai_client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Recommend me 5 laptop about {topic}, with a short description for each laptop."}
        ]
    )
    return response.choices[0].message.content

laptop_recomendation_def = {
    "type": "function",
    "function" : {
        "name": "laptop_recomendation",
        "description": "Recommend laptop based on a topic",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic to recommend laptop about"
                }
            },
            "required": ["topic"]
        }
    }
}