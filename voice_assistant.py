import os
import torch
import speech_recognition as sr
import pyttsx3

from tokenizer import load_vocab, encode, decode
from model import MiniGPT
from maths_solver import try_solve_maths

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pt")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        return torch.device("cpu")


DEVICE = get_device()
WAKE_WORDS = ("vector", "victor")


def load_model():
    if not os.path.exists(MODEL_PATH):
        print("model.pt not found. Run 'python train.py' first.")
        return None, None, None

    stoi, itos = load_vocab()
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model = MiniGPT(checkpoint["vocab_size"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, stoi, itos


def generate_reply(model, stoi, itos, prompt, max_new_tokens=60):
    maths_answer = try_solve_maths(prompt)
    if maths_answer is not None:
        return maths_answer

    if not prompt.strip():
        prompt = "Vector is"

    prompt = prompt.lower()

    encoded = encode(prompt, stoi)
    if not encoded:
        encoded = encode("Vector is", stoi)

    context = torch.tensor([encoded], dtype=torch.long).to(DEVICE)
    out = model.generate(context, max_new_tokens=max_new_tokens, temperature=0.8)[0].tolist()
    text = decode(out, itos)

    cut_pos = text.find("KRB:", len(prompt))
    if cut_pos != -1:
        text = text[:cut_pos].strip()

    text = text.replace(" <unk>", "").replace("<unk> ", "").replace("<unk>", "")
    text = " ".join(text.split())

    return text


class VoiceAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()

        print("Calibrating microphone for Vector... stay quiet for 2 seconds.")
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
        print("Ready. Say 'Vector' to start.")

    def speak(self, text):
        print(f"Vector: {text}")
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
        del engine

    def listen_once(self, timeout=None, phrase_time_limit=8):
        with self.mic as source:
            audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        try:
            text = self.recognizer.recognize_google(audio, language="en-IN")
            return text.lower()
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            print(f"Speech recognition service error: {e}")
            return ""

    def run(self, model, stoi, itos):
        while True:
            try:
                heard = self.listen_once(timeout=None)
            except sr.WaitTimeoutError:
                continue

            if not heard:
                continue

            print(f"Heard: {heard!r}")

            matched_wake_word = next((w for w in WAKE_WORDS if w in heard), None)
            if matched_wake_word is None:
                continue

            command = heard.split(matched_wake_word, 1)[-1].strip()

            if not command:
                self.speak("Yes, go ahead.")
                try:
                    command = self.listen_once(timeout=5)
                except sr.WaitTimeoutError:
                    command = ""

            if not command:
                self.speak("I didn't hear anything.")
                continue

            reply = generate_reply(model, stoi, itos, command)
            self.speak(reply)


def main():
    model, stoi, itos = load_model()
    if model is None:
        return

    assistant = VoiceAssistant()
    assistant.speak("Vector is online.")
    assistant.run(model, stoi, itos)


if __name__ == "__main__":
    main()
