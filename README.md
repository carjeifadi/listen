# listen

This project demonstrates an end-to-end audio conversation pipeline powered by
OpenAI services. The pipeline converts text to speech, re-transcribes the audio,
and generates a conversational response that is played back to the user.

## Setup

1. Create and activate a Python 3.9+ virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and update the values with your API
   credentials.

## Usage

Run the conversation loop from the command line:

```bash
python -m src.conversation
```

You will be prompted for an initial piece of text. The script will synthesize
speech, play it, transcribe the audio, request a chat response, and finally
synthesize and play the model's reply.

> **Note:** Audio playback requires the optional `simpleaudio` dependency and an
> audio device supported by the host operating system. If playback is not
> available, the script logs a warning but continues executing the rest of the
> flow.
