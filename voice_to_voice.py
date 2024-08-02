import uuid
import os
import sys
import gradio as gr
import assemblyai as aai
from translate import Translator
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def voice_to_voice(audio_file):
    transcription_response = audio_transcription(audio_file)

    if transcription_response.status == aai.TranscriptStatus.error:
        raise gr.Error(transcription_response.error)

    text = transcription_response.text

    paths = []
    languages = ["fr", "it", "es"]

    for lang in languages:
        translation = translate_text(text, "en", lang)
        result_file_path = text_to_speach(translation)
        paths.append(Path(result_file_path))

    return paths


def audio_transcription(audio_file):
    aai.settings.api_key = os.getenv('AAI_API_KEY')
    transcriber = aai.Transcriber()
    transcription = transcriber.transcribe(audio_file)

    return transcription


def translate_text(text, from_lang, to_lang):
    translator = Translator(from_lang=from_lang, to_lang=to_lang)
    text = translator.translate(text)
    return text


def text_to_speach(text):
    my_api_key = os.getenv('ELEVENLABS_API_KEY')
    client = ElevenLabs(
        api_key=my_api_key,
    )

    response = client.text_to_speech.convert(
        voice_id=os.getenv('VOICE_ID'),  # my voice
        optimize_streaming_latency="0",
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_multilingual_v2",
        # use the turbo model for low latency, for other languages use the `eleven_multilingual_v2`
        voice_settings=VoiceSettings(
            stability=0.5,
            similarity_boost=0.75,
            style=0,
            use_speaker_boost=True,
        ),
    )

    # uncomment the line below to play the audio back
    # play(response)

    # Generating a unique file name for the output MP3 file
    save_file_path = f"{uuid.uuid4()}.mp3"

    # Writing the audio to a file
    with open(save_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

    print(f"{save_file_path}: A new audio file was saved successfully!")

    # Return the path of the saved audio file
    return save_file_path


def cleanup(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp3"):
            file_path = os.path.join(folder_path, filename)
            try:
                os.remove(file_path)
                print(f"Deleted: {filename}")
            except OSError as e:
                print(f"Error deleting {filename}: {e}")


audio_input = gr.Audio(
    sources=["microphone"],
    type="filepath"
)

demo = gr.Interface(
    fn=voice_to_voice,
    inputs=audio_input,
    outputs=[gr.Audio(label="French"), gr.Audio(label="Italian"), gr.Audio(label="Spanish")]
)

if __name__ == "__main__":
    cleanup(os.getcwd())
    demo.launch()
    cleanup(os.getcwd())
