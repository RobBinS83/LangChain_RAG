import asyncio
import os
import ssl
from typing import Any, List, Dict

import certifi
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap 

from logger import Colors, log_error, log_info, log_success, log_warning, log_header

load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", 
    show_progress_bar=False,
    chunk_size=50,
    retry_min_seconds=10,
    )

#chroma = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
vector_store = PineconeVectorStore(
    index_name="langchain-doc-assist",
    embedding=embeddings,
)

tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()

async def index_documents_async(documents: List[Document], batch_size: int = 50) -> None:
    """Asynchronously index documents into the vector store."""
    log_header("VECTOR STORAGE INDEXING PHASE")
    log_info(f"VectorStore Indexing: Adding {len(documents)} documents into the vector store...", Colors.DARKCYAN)

    # create batches
    batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
    log_info(f"VectorStore Indexing: Created {len(batches)} batches of size {batch_size}.")

    # process bacthes concurrently
    async def add_batch(batch: List[Document], batch_number: int) -> bool:
        try:
            await vector_store.aadd_documents(batch)
            log_success(f"VectorStore Indexing: Successfully added {batch_number}/{len(batches)} {len(batch)} documents into the vector store.")
        except Exception as e:
            log_error(f"VectorStore Indexing: failed to add batch {batch_number} - {e}")
            return False
        return True
    
    # Process batches condurrently
    async with vector_store:
        tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # count successes and failures
    success_count = sum(1 for result in results if result is True)
    if success_count == len(batches):
        log_success(f"VectorStore Indexing: Successfully indexed all batches into the vector store. ({success_count}/{len(batches)})")
    else:
        log_warning(f"VectorStore Indexing: Completed with some failures. Successfully indexed {success_count}/{len(batches)} batches.")


async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "TavilyCrawl: Starting to crawl documentation from https://python.langchain.com",
        Colors.PURPLE,
        )
    
    # Crawl the documentation site
    res = tavily_crawl.invoke({
        "url": "https://python.langchain.com",
        "max_depth": 5,
        "max_breadth": 100,
        "limit": 500,
        "extract_depth": "advanced",
        #"instructions": "Find all documentation pages about Agents"
    })

    #all_docs = [Document(page_content=result["raw_content"], metadata={"source": result["url"]}) 
    #            for result in res["results"]]
    
    results = res.get("results", [])
    all_docs = []

    for result in results:
        content = result.get("raw_content")
        if not content:
            log_warning(f"Skipping empty content for {result.get('url')}")
            continue
        all_docs.append(Document(page_content=content, metadata={"source": result.get("url")}))

    log_success(f"TavilyCrawl: Successfully crawled {len(all_docs)} documents from the site.")

    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 chunk overlap.",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(f"Text Splitter: Successfully split {len(all_docs)} documents into {len(splitted_docs)} chunks.")

    # Process documents asynchronously
    await index_documents_async(splitted_docs, batch_size=50)

    log_header("PIPELINE COMPLETED")
    log_success("Documentation ingestion and indexing pipeline completed successfully.")
    log_info("Summary:", Colors.BOLD)
    log_info(f" -- Documents extracted: {len(all_docs)}")
    log_info(f" -- Chunks created: {len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(main()) 