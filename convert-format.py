import os
import subprocess

INPUT_DIR = "input"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for file in os.listdir(INPUT_DIR):

    if not file.lower().endswith((".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg")):
        continue

    input_file = os.path.join(INPUT_DIR, file)

    output_name = os.path.splitext(file)[0] + ".wav"
    output_file = os.path.join(OUTPUT_DIR, output_name)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_file,
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        output_file
    ]

    subprocess.run(cmd)

    print("Converted:", output_name)

print("Done!")
