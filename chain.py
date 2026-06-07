import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_chain(retriever):
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant",
        temperature=0.2
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful study assistant.
Answer the student's question using ONLY the context provided below.
If the answer is not in the context, say "I don't have that information in the provided notes."
Always be clear, concise and helpful.

Context:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


if __name__ == "__main__":
    from retriever import load_retriever

    retriever = load_retriever("faiss_index", k=5)
    chain = build_chain(retriever)

    question = "What are the main topics covered in the notes?"
    answer = chain.invoke({
        "question": question,
        "chat_history": []
    })

    print(f"Question: {question}")
    print(f"Answer: {answer}")