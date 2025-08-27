from tools.broadcast import broadcast_def,broadcast
from tools.laptop_recomendation import laptop_recomendation_def,laptop_recomendation
from tools.laptop_gaming_recomendation import laptop_gaming_recomendation_def,laptop_gaming_recomendation
from utils import openai_client
import json
from loguru import logger

tools = [broadcast_def,
         laptop_recomendation_def,
         laptop_gaming_recomendation_def]


def execute_function(function_name,function_args):
    if function_name == "broadcast":
        return broadcast(function_args["message"])
    elif function_name == "laptop_gaming_recomendation":
         return laptop_gaming_recomendation(function_args["topic"])
    elif function_name == "laptop_recomendation":
        return laptop_recomendation(function_args["topic"])
    else:
        return {"error": "Function not found"}


def gaming_recommend(topic: str):
    SYSTEM_PROMPT = """
    You are a helpful assistant that helps people find laptop based on a topic.
    """

    messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Find me laptop about {topic}"},
            ]
    logger.info(f"Executing function: {SYSTEM_PROMPT} with args: {messages}")

    while True:
        response = openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        message = response.choices[0].message
        messages.append(message)

        if message.tool_calls:
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                logger.info(f"Executing function: {function_name} with args: {function_args}")

                function_response = execute_function(function_name, function_args)
                messages.append({
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                    "tool_call_id": tool_call.id
                })
                logger.info(f"Function response: {function_response}","Messages: {messages}")
        else:
            break
    return "Recommendation process completed."

if __name__ == "__main__":
    topic = "Any laptop games and their prices you can recommend me ?"
    gaming_recommend(topic)