from faster_whisper import WhisperModel

audio_file = "data/audio/questio.wav"

print("Chargement du modèle...")

model = WhisperModel(
    "medium",
    device="cpu",
    compute_type="int8"
)


print("Transcription en cours...")

segments, info = model.transcribe(
    audio_file,
    beam_size=5
)

print(f"\nLangue détectée : {info.language}")
print(f"Probabilité : {info.language_probability:.2f}\n")

texte = ""

for segment in segments:
    texte += segment.text + " "

print("TRANSCRIPTION :")
print(texte)