import subprocess # For running the shell commands
import whisper # openai speech-to-text library
import os # For file path manipulations

def extract_audio(input_video_path: str, output_audio_path: str = "extracted_audio.wav") -> str:
    """
    Extracts audio from a video file and saves it as a WAV file.
    
    Args:
        input_video_path (str): Path to the input video file.
        output_audio_path (str): Path to save the extracted audio file.
    """
    if os.path.exists(output_audio_path):
        os.remove(output_audio_path) # Remove existing file to avoid conflicts

    command = [
        "ffmpeg", # Command line tool for processing video and audio files
        "-i", input_video_path, # Input video file path
        "-q:a", "0",  # Generates Highest quality audio
        "-map", "a",  # Selects all audio streams from the video
        output_audio_path, # Output audio file path
        "-y"  # Overwrite output file if it exists
    ] 

    # Execute the above command
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return output_audio_path

def transcribe_audio(audio_path: str, model_size: str = "base") -> str:
    """
    Transcribes the given audio file using OpenAI's Whisper model.
    
    Args:
        audio_path (str): Path to the audio file to be transcribed.
        model_size (str): Size of the Whisper model to use (e.g., "tiny", "base", "small", "medium", "large").
    
    Returns:
        str: The transcribed text.
    """
    # Load the specified Whisper model
    model = whisper.load_model(model_size)
    
    # Transcribe the audio file
    result = model.transcribe(audio_path)

    # Extract the transcribed text from the result
    transcripted_text = result["text"]
    
    return transcripted_text