from transformers import pipeline

def summarize_text(text: str, model_name: str = "facebook/bart-large-cnn", max_length: int = 150, min_length: int = 30) -> str:
    """
    Summarizes the given text using a pre-trained transformer model.
    
    Args:
        text (str): The text to be summarized.
        model_name (str): The name of the pre-trained model to use for summarization.
        max_length (int): The maximum length of the summary.
        min_length (int): The minimum length of the summary.
    
    Returns:
        str: The summarized text.
    """
    # Initialize the summarization pipeline
    summarizer = pipeline("summarization", model=model_name)
    
    # Generate the summary using the above pipeline
    # do_sample=False for deterministic output i.e no randomness
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    
    # Extract the summarized text from the result
    summarized_text = summary[0]['summary_text']
    
    return summarized_text