def chunk_text(text: str, max_chunk_size: int = 2000, overlap: int = 200) -> list:
    """
    Splits the input text into smaller chunks with specified size 
    with overlap to help maintain context continuity between chunks.
    
    Args:
        text (str): The text to be chunked.
        max_chunk_size (int): The maximum size of each chunk.
        overlap (int): The number of overlapping characters between consecutive chunks.
    
    Returns:
        list: A list of text chunks.
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + max_chunk_size, text_length)
        chunks.append(text[start:end])
        start += max_chunk_size - overlap  # Move start forward with overlap

        if start < 0:
            start = 0  # Ensure start does not go negative
    
    return chunks

def chunked_summarize(text: str, summarize_function, max_chunk_size: int = 2000) -> str:
    """
    Summarizes large text by chunking it into smaller parts, summarizing each part,
    and then combining the summaries.
    
    Args:
        text (str): The text to be summarized.
        summarize_function (callable): The function to use for summarization.
        max_chunk_size (int): The maximum size of each chunk.
    
    Returns:
        str: The combined summarized text from all chunks.
    """
    chunks = chunk_text(text, max_chunk_size = max_chunk_size, overlap = 200)
    summarized_chunks = [summarize_function(chunk) for chunk in chunks]
    combined_summary = " ".join(summarized_chunks)
    
    return combined_summary