from langchain_community.document_loaders import WebBaseLoader


loader = WebBaseLoader("https://novagito.com/")

# text=loader.load()

# print(text)

import asyncio
from crawl4ai import AsyncWebCrawler

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

        print(text)


asyncio.run(main())
