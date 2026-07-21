import os
import wave

DATASET = "dataset"

def check_audio(folder):
    path = os.path.join(DATASET, folder)

    files = [
        f for f in os.listdir(path)
        if f.endswith(".wav")
    ]

    print("\n", folder)
    print("Số file:", len(files))

    for file in files[:5]:
        wav_path = os.path.join(path, file)

        with wave.open(wav_path, "rb") as w:
            channels = w.getnchannels()
            rate = w.getframerate()
            sample = w.getsampwidth()
            frames = w.getnframes()

            duration = frames / rate

            print(
                file,
                "|",
                rate, "Hz",
                "|",
                channels, "channel",
                "|",
                sample*8, "bit",
                "|",
                round(duration,2),
                "s"
            )


check_audio("positive")
check_audio("negative")