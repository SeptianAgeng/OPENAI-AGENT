
from agents import Agent, Runner, function_tool
from dotenv import load_dotenv


@function_tool
def get_price(laptop_name: str) -> str:
    # Simulated function to get laptop price
    prices = {
        "Gaming Laptop A": "$1200",
        "Gaming Laptop B": "$1500",
        "Programming Laptop A": "$1000",
        "Programming Laptop B": "$1300"
    }
    return prices.get(laptop_name, "Price not found")

load_dotenv()

agent = Agent(
    name ="Laptop Recommender Agent",
    instructions="you are a helpful assistant that helps people find laptop based on a topic.",
    model ="gpt-4.1",
    tools=[get_price],
)

async def main():
    messages = []

    while True:
        user_input = input("User: ")
        messages.append({"role": "user", "content": user_input})
        runner = await Runner.run(starting_agent=agent, input=messages)
        messages = runner.to_input_list()
        print(messages)
        print("===="*20)
        print(runner.last_agent)
        print("===="*20)
        print(runner.final_output)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())