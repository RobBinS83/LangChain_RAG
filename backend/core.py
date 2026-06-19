import os
from typing import Any, List, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage, HumanMessage

from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool

load_dotenv()

# initialize the embeddings 
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

# initialize the vector store
vector_store = PineconeVectorStore(
    index_name="langchain-doc-assist",
    embedding=embeddings,
)

# initialize the chat model
model = init_chat_model(
    model="gpt-5.5",
    model_provider="openai",
)

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """
    Retrieve revelant documentation to help answer user queries about LangChain. 
    """
    retrieved_docs = vector_store.as_retriever().invoke(query, k=5)

    # Seriealize the retrieved documents into a single string
    serialized_doc = "\n\n".join(
        [f"Source: {doc.metadata.get('source', 'unknown')}\n\nContent: {doc.page_content}" 
         for doc in retrieved_docs]
    )

    return serialized_doc, retrieved_docs

def run_llm(query:str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentations.

    Args:
        query (str): The user's question.

    Returns:
        dictionary containing:
            - answer: the generated answer from the LLM
            - context: List of retrieved documents

    """
    # Create the agent with the retrieve_context tool

    system_prompt = (
        "You are a helpful Ai assistant that answers questions about LangChain documentation. "
        "You have access to a tool that can retrieve relevant documentation based on user queries. "
        "Use the tool to find relavant information before answering questions. "
        "Always cite the sources you use in your answers. " 
        "If you cannot find the answer in the retrieved documents, say you don't know instead of making up an answer. "
    )


    agent = create_agent(
        model=model,
        tools=[retrieve_context],
        system_prompt=system_prompt,
    )

    # build messages list
    messages = [{"role": "user", "content": query}]

    # Run the agent with the user query
    response = agent.invoke({"messages": messages})

    # extract the answer from the last AI message
    answer = response["messages"][-1].content

    # extract context documents from ToolMessage artifacts
    context_docs = []
    for message in response["messages"]:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            context_docs.extend(message.artifact)

    return {
        "answer": answer,
        "context": context_docs
    }

if __name__ == '__main__':
    result = run_llm(query="what are deep agents?")
    print(result)