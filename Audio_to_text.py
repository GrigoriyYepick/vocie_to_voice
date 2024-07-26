import assemblyai as aai
import os
from dotenv import load_dotenv

load_dotenv()

aai.settings.api_key = os.getenv('AAI_API_KEY')
transcriber = aai.Transcriber()


transcription = transcriber.transcribe(r'c:\temp\6_2.mp3')

# transcript = transcriber.transcribe("https://storage.googleapis.com/aai-web-samples/news.mp4")
# transcript = transcriber.transcribe("./my-local-audio-file.wav")

print(transcription.text)