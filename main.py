import os
from dotenv import load_dotenv, main
from langchain.messages import HumanMessage, AIMessage
from retriever import load_retriever
from chain import build_chain
from memory import create_memory, add_to_memory, get_history

load_dotenv()

print("AI Study Buddy")

print("Loading retriever...")
retriever = load_retriever("faiss_index", k=5)
print("Building chain...")
chain = build_chain(retriever)
print("Creating memory...")
memory = create_memory()

print("You can start asking questions. Type 'exit' to quit or 'history' to see chat history.")

while True:
    question = input("You: ")
    if question.lower() == "exit":
        print("Goodbye!")
        break
    elif question.lower() == "history":
        print("Chat History:")
        if not get_history(memory):
            print("No chat history yet.")
        
        for msg in get_history(memory):
            role = "You" if isinstance(msg, HumanMessage) else "AI"
            print(f"{role}: {msg.content}")

    else:
        try:
            answer = chain.invoke({
                "question": question,
                "chat_history": get_history(memory)
            })
            add_to_memory(memory, question, answer)
            print(f"AI: {answer}")
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()