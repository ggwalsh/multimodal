"""
Speech Recogniser for Multimodal AI System
AAI202 Applications of Artificial Intelligence
Author: Geoff Walsh (A00186663)

Captures speech from the default microphone and converts it to text
using the SpeechRecognition library with the Google Web Speech API.

Dependencies:
    SpeechRecognition
    PyAudio

Usage:
    python speech_recognizer.py
"""

import speech_recognition as sr
import time
from datetime import datetime


def list_microphones():
    """
    List available microphone devices.
    """
    mic_list = sr.Microphone.list_microphone_names()
    print("\nAvailable microphones:")
    for i, name in enumerate(mic_list):
        print(f"  [{i}] {name}")
    return mic_list


def recognize_speech_from_mic(recognizer, microphone, prompt="Speak now..."):
    """
    Capture audio from the microphone and transcribe it.
    """
    result = {
        "success": False,
        "error": None,
        "transcription": None,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with microphone as source:
        print(f"\n[INFO] {prompt}")
        print("[INFO] Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("[INFO] Listening...")

        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            result["error"] = "Listening timed out while waiting for speech."
            print(f"[ERROR] {result['error']}")
            return result

    try:
        transcription = recognizer.recognize_google(audio)
        result["success"] = True
        result["transcription"] = transcription
        print(f'[SUCCESS] Recognised: "{transcription}"')
    except sr.UnknownValueError:
        result["error"] = "Could not understand the audio."
        print(f"[ERROR] {result['error']}")
    except sr.RequestError as e:
        result["error"] = f"Could not reach Google Speech Recognition service: {e}"
        print(f"[ERROR] {result['error']}")
    except Exception as e:
        result["error"] = f"Unexpected error: {e}"
        print(f"[ERROR] {result['error']}")

    return result


def main():
    """
    Run a series of speech recognition tests from the microphone.
    """
    print("=" * 70)
    print("Speech Recogniser - Microphone Capture")
    print("=" * 70)

    recognizer = sr.Recognizer()

    list_microphones()

    try:
        microphone = sr.Microphone()
        print("\n[INFO] Using default system microphone.")
    except Exception as e:
        print(f"[ERROR] Could not initialise microphone: {e}")
        print("Ensure PyAudio is installed and a microphone is available.")
        return

    test_prompts = [
        "Hello how are you today",
        "The weather is nice outside",
        "Artificial intelligence is fascinating",
    ]

    print("\n[INFO] Starting recognition tests.")
    print("Speak clearly when prompted. Short sentences work best.")
    input("Press Enter to begin the first test...")

    results = []
    num_tests = 3

    for i in range(num_tests):
        print(f"\n{'=' * 40}")
        print(f"TEST {i + 1} of {num_tests}")
        print(f"{'=' * 40}")

        if i < len(test_prompts):
            print(f'Suggested: "{test_prompts[i]}"')

        result = recognize_speech_from_mic(
            recognizer,
            microphone,
            prompt=f"Test {i + 1}: Speak after the prompt.",
        )
        results.append(result)

        if i < num_tests - 1:
            time.sleep(0.5)
            input("Press Enter for the next test...")

    print("\n" + "=" * 70)
    print("RECOGNITION SUMMARY")
    print("=" * 70)

    successful = 0
    for idx, res in enumerate(results, 1):
        print(f"\nTest {idx}  ({res['timestamp']})")
        if res["success"]:
            successful += 1
            print("  Status      : SUCCESS")
            print(f'  Transcript  : "{res["transcription"]}"')
        else:
            print("  Status      : FAILED")
            print(f"  Error       : {res['error']}")

    print(f"\nOverall: {successful}/{num_tests} successful.")
    print("=" * 70)
    print("[INFO] Session completed.")


if __name__ == "__main__":
    main()
