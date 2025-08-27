from utils import openai_client

def laptop_gaming_recomendation(topic:str):
    SYSTEM_PROMPT = """
    You are a helpful assistant that helps people find the laptop price for the given topic.
    
    #GUIDELINES: 
    - Add each laptop prices with a short description for each laptop.
    - Ensure the laptop prices are relevant to the topic provided.
    #Important: 
    - Please broadcast progress of each step using the broadcast function.
    """

    response = openai_client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Give gaming device based on the {topic}"}
        ]
    )
    return response.choices[0].message.content

laptop_gaming_recomendation_def = {
    "type": "function",
    "function" : {
        "name": "laptop_gaming_recomendation",
        "description": "Recommend laptop price based on a topic",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic to recommend laptop price about"
                }
            },
            "required": ["topic"]
        }
    }
}