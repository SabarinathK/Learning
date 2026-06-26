from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import asyncio
from crawl4ai import AsyncWebCrawler
from langchain_core.documents import Document

URL = "https://novagito.com/"

import re
async def main():
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(URL)

        text = result.markdown

        # Remove markdown images
        text = re.sub(r"!\[.*?\]\(.*?\)", "", text)

        # Keep only link text
        text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)

        # Remove SVG/data URI garbage
        text = re.sub(r"data:image.*?(?=\n|$)", "", text)
        text = re.sub(r"%3c.*?%3e", "", text)

        # Remove extra blank lines
        text = re.sub(r"\n\s*\n+", "\n\n", text)

        metadata = {"url": result.url}
        content = text
    return [Document(page_content=content, metadata=metadata)]

docs=asyncio.run(main())

data_split = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
all_split = data_split.split_documents(docs)
print(len(all_split))

