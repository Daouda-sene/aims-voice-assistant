from gtts import gTTS

text = """
An agent is a system that uses an LLM,
tools, memory and decision loops
to solve tasks.
"""

tts = gTTS(text=text, lang="en")

tts.save("response.mp3")

print("Audio sauvegardé : response.mp3")