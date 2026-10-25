def chunk_pages(pages,chunk_size=800,overlap=100):
    chunks = []
    for page in pages:
        text = page['text']
        page_number = page['page']
        source = page['source']
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append({
                    'text':chunk_text,
                    'page':page_number,
                    'source':source
                })
            start += chunk_size-overlap
    return chunks