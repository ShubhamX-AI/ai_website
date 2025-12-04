import chromadb
import aiohttp
import uuid
import urllib.parse
import re
from bs4 import BeautifulSoup


class ScarapeANDSave:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path="./vector_db")
        self.collection = self.chroma_client.get_or_create_collection(name="web_chunks",)

    async def fetch_url(self, url):
        try:
            print(f"Fetching {url}...")
            headers = {'User-Agent': 'Mozilla/5.0'}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    text = await response.text()
                    soup = BeautifulSoup(text, "html.parser")
                    payload = {"status": 0,"data": soup, "meassage": ""}
                    return payload
        except aiohttp.ClientError as e:
            print(f"Error fetching {url}: {e}")
            return {"status": -1, "data": None, "meassage": str(e)}

    async def extract_site_meta(self, soup):
        try:
            async def get_meta_content(props):
                """Helper to check multiple tag attributes for content"""
                for prop in props:
                    tag = soup.find("meta", property=prop)
                    if tag and tag.get("content"):
                        return tag.get("content")
                    tag = soup.find("meta", attrs={"name": prop})
                    if tag and tag.get("content"):
                        return tag.get("content")
                return ""

            # 1. Extract Title
            title = await get_meta_content(["og:title", "twitter:title"])
            if not title and soup.title:
                title = soup.title.string

            # 2. Extract Description
            description = await get_meta_content(["og:description", "description", "twitter:description"])

            # 3. Extract Image
            image = await get_meta_content(["og:image", "twitter:image"])

            # 4. Fallback Image: Look for Favicon/Logo (search several possibilities)
            if not image:
                link_tag = soup.find("link", rel="icon")
                if not link_tag:
                    link_tag = soup.find("link", rel="shortcut icon")
                if not link_tag:
                    link_tag = soup.find("link", rel="apple-touch-icon")
                if link_tag and link_tag.get("href"):
                    image = link_tag.get("href")


            playlaod = {
                "status": 0,
                "message": "",
                "data": {
                    "title": (title or "Website Section").strip(),
                    "description": (description or "").strip(),
                    "image": (image or "").strip()
                }
            }

            return playlaod
        except Exception as e:
            print(f"Error extracting site meta: {e}")
            return {"status": -1, "message": str(e), "data": {}}

    async def extract_semantic_chunks(self, soup, base_url: str):

        try:
            # Clean Noise
            for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside", "iframe", "svg"]):
                tag.decompose()

            site_meta_res = await self.extract_site_meta(soup)
            if site_meta_res['status'] != 0:
                return {"status": -1, "message": site_meta_res['message'], "data": {}}
            
            global_image = site_meta_res['data'].get("image", "")
            
            ids, docs, metas = [], [], []

            current_header = "Introduction"
            current_text = []
            current_images = []

            # HELPER: Extracts URL from style="background-image: url('...')"
            async def get_background_image_url(style_str):
                if not style_str: return None
                # Regex to find url(...) ignoring quotes
                match = re.search(r'url\s*\((?:[\'"]?)(.*?)(?:[\'"]?)\)', style_str, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
                return None

            async def flush_chunk():
                nonlocal current_header, current_text, current_images
                full_text = " ".join(current_text).strip()
                
                if len(full_text) > 50:
                    snippet = " ".join(full_text.split()[:8]) 
                    safe_snippet = urllib.parse.quote(snippet)
                    deep_link = f"{base_url}#:~:text={safe_snippet}"

                    # Priority: Local Image, Global Image
                    thumb = current_images[0] if current_images else global_image
                    
                    chunk_id = str(uuid.uuid4())
                    document_content = f"Topic: {current_header}\nContent: {full_text}"
                    
                    meta = {
                        "url": deep_link,
                        "base_url": base_url,
                        "title": current_header,
                        "image": thumb,
                        "snippet": full_text[:200]
                    }

                    ids.append(chunk_id)
                    docs.append(document_content)
                    metas.append(meta)

                current_text = []
                current_images = []

            # UPDATED LIST: We now include 'div' and 'span' to check their styles
            elements = soup.body.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'ul', 'ol', 'img', 'div', 'span'])

            for element in elements:
                
                # --- CHECK 1: Is it a Header? ---
                if element.name in ['h1', 'h2', 'h3', 'h4']:
                    await flush_chunk()
                    current_header = element.get_text(strip=True)
                
                # --- CHECK 2: Is it a Standard Image? ---
                elif element.name == 'img':
                    src = element.get('src')
                    if src:
                        if not src.startswith(('http', '//')):
                            src = urllib.parse.urljoin(base_url, src)
                        current_images.append(src)

                # --- CHECK 3: Is it a Background Image? ---
                # We check every element in the loop for a style attribute
                elif element.has_attr('style'):
                    bg_url = await get_background_image_url(element['style'])
                    if bg_url:
                        if not bg_url.startswith(('http', '//')):
                            bg_url = urllib.parse.urljoin(base_url, bg_url)
                        current_images.append(bg_url)

                # --- CHECK 4: Is it Text? ---
                if element.name in ['p', 'ul', 'ol', 'div', 'span']:
                    text = element.get_text(" ", strip=True)
                    # Filter out empty text or text that is just the image url
                    if text and len(text) > 20: 
                        current_text.append(text)

            await flush_chunk()

            playlaod = {
                "status": 0,
                "message": "",
                "data": {
                    "ids": ids,
                    "docs": docs,
                    "metas": metas
                }
            }
            return playlaod

        except Exception as e:
            print(f"Error extracting semantic chunks: {e}")
            return {"status": -1, "message": str(e), "data": {}}
        
    async def save_chunks(self, ids, docs, metas):
        try:
            self.collection.add(ids=ids, documents=docs, metadatas=metas)
            return {"status": 0, "message": "", "data": {"len": len(ids)}}
        except Exception as e:
            print(f"Error saving chunks: {e}")
            return {"status": -1, "message": str(e), "data": {}}
        

if __name__ == "__main__":
    import asyncio
    scrape = ScarapeANDSave()
    url = "https://intglobal.com/"

    fetch_url_res = asyncio.run(scrape.fetch_url(url))
    if fetch_url_res['status'] != 0:
        print(fetch_url_res['meassage'])
        exit()

    soup = fetch_url_res['data']

    extract_semantic_chunks_res = asyncio.run(scrape.extract_semantic_chunks(soup, url))
    if extract_semantic_chunks_res['status'] != 0:
        print(extract_semantic_chunks_res['message'])
        exit()

    ids = extract_semantic_chunks_res['data']['ids']
    docs = extract_semantic_chunks_res['data']['docs']
    metas = extract_semantic_chunks_res['data']['metas']

    save_chunks_res = asyncio.run(scrape.save_chunks(ids=ids, docs=docs, metas=metas))
    if save_chunks_res['status'] != 0:
        print(save_chunks_res['message'])
        exit()