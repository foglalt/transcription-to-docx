# Transcriber (CLI)

Simple CLI to transcribe long audio with `gpt-4o-transcribe`, run a correction pass with another OpenAI model, and save the result to a DOCX file.

## Requirements

- Python 3.9+
- ffmpeg (required by `pydub` for splitting large audio files)
- OpenAI API key

## Setup

1. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python transcribe_to_docx.py "path\to\audio.mp3"
```

If no file is specified, the script will look for a `resources` folder in the project root and transcribe every `.mp3` inside.
All outputs are saved to the `output` folder in the project root with the same base filename.

### Options
- `--language`: Language hint (default: `auto` or `DEFAULT_LANGUAGE` from `.env`).
- `--context` / `--context-file`: Context hints to improve transcription accuracy.
- `--correction-model`: Model for the correction pass (default: `gpt-4o-mini`).
- `--no-correction`: Skip the correction pass.
- `--correction-chunk-words`: Word limit per correction chunk (default: 1200).
