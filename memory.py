from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage


def create_memory():
    return InMemoryChatMessageHistory()

def add_to_memory(memory, question, answer):
    memory.add_message(HumanMessage(content=question))
    memory.add_message(AIMessage(content=answer))

def get_history(memory):
    return memory.messages


if __name__ == "__main__":
    memory = create_memory()
    add_to_memory(memory, "What is the capital of France?", "The capital of France is Paris.")
    add_to_memory(memory, "What is the largest planet?", "The largest planet in our solar system is Jupiter.")
    for msg in get_history(memory):
        role = "You " if isinstance(msg, HumanMessage) else "AI"
        print(f"{role}: {msg.content}")
