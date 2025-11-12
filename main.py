import os # For file path manipulations
from transcriber import extract_audio, transcribe_audio # Importing functions from transcriber module
from summarizer import summarize_text # Importing summarize function from summarizer module
from utils import chunked_summarize # Importing chunked summarization utility
from rephraser import rephrase_text_with_correction # Importing rephrasing function

def video_to_summary(input_video_path: str, 
                     audio_output_path: str = "extracted_audio.wav", 
                     whisper_model_size: str = "base", 
                     summarization_model_name: str = "facebook/bart-large-cnn", 
                     use_chunking: bool = False) -> str:
    """
    Converts a video file to a summarized text by extracting audio, transcribing it,
    and summarizing the transcription.
    Args:
        input_video_path (str): Path to the input video file.
        audio_output_path (str): Path to save the extracted audio file.
        whisper_model_size (str): Size of the Whisper model to use for transcription.
        summarization_model_name (str): Name of the pre-trained model for summarization.
        use_chunking (bool): Whether to use chunked summarization for long texts.
    Returns:
        str: The summarized text from the video.
    """ 
    # Step 1: Extract audio from the video file
    audio_path = extract_audio(input_video_path, audio_output_path)

    # Step 2: Transcribe the extracted audio
    transcribed_text = transcribe_audio(audio_path, model_size = whisper_model_size)

    # Step 3: Summarize the transcribed text
    # if use_chunking:
    #     summarized_text = chunked_summarize(
    #         transcribed_text, 
    #         summarize_function=lambda text: summarize_text(
    #             text, 
    #             model_name=summarization_model_name
    #         ),
    #         max_chunk_size=2000
    #     )
    # else:
    #     summarized_text = summarize_text(
    #         transcribed_text, 
    #         model_name=summarization_model_name
    #     )
    
    # Clean up the extracted audio file
    if os.path.exists(audio_path):
        os.remove(audio_path)

    return transcribed_text

if __name__ == "__main__":

    summary = video_to_summary(
        input_video_path = "sample_video.mp4",
        audio_output_path = "extracted_audio.wav",
        whisper_model_size = "base",
        summarization_model_name = "facebook/bart-large-cnn",
        use_chunking = True  # Enable chunked summarization for long videos
    )

    print("Video Summary:")
    print(summary)

    with open("text_data.txt", "w", encoding="utf-8") as file:
        file.write(summary)

    # corrected_summary = rephrase_text_with_correction(summary)
    # print("\nCorrected Summary:")