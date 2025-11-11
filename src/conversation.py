"""Audio conversation utilities using OpenAI APIs.

This module exposes helper functions to convert text to speech, speech to text,
and to run a full round-trip conversation with a chat model.  It expects the
following environment variables to be configured (e.g. via a .env file):

- ``OPENAI_API_KEY`` – required for authenticating with OpenAI services.
- ``OPENAI_API_BASE`` – optional custom API endpoint.
- ``TTS_MODEL`` – optional override for the text-to-speech model name.
- ``TTS_VOICE`` – optional default voice for text-to-speech.
- ``STT_MODEL`` – optional override for the speech-to-text model name.
- ``CHAT_MODEL`` – optional override for the chat completion model name.
- ``OUTPUT_DIR`` – optional directory where generated audio files are stored.
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise RuntimeError(
        "The 'openai' package is required. Install dependencies first."
    ) from exc

try:
    import simpleaudio  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    simpleaudio = None

load_dotenv()

LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
DEFAULT_TTS_FORMAT = os.getenv("TTS_FORMAT", "wav")


def _build_client() -> OpenAI:
    """Create an OpenAI client using configured environment variables."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Create a .env file or export the variable."
        )

    client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_API_BASE"))
    return client


@dataclass
class TextToSpeechResult:
    """Container for the TTS result."""

    text: str
    audio_path: Path


@dataclass
class SpeechToTextResult:
    """Container for the STT result."""

    audio_path: Path
    transcript: str


@dataclass
class ChatResult:
    """Container for the chat model result."""

    prompt: str
    response: str


def _ensure_output_path(output_path: Optional[Union[Path, str]]) -> Path:
    """Ensure the directory for the audio output exists."""

    suffix = (
        DEFAULT_TTS_FORMAT if DEFAULT_TTS_FORMAT.startswith(".") else f".{DEFAULT_TTS_FORMAT}"
    )

    if output_path is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"tts_{uuid.uuid4().hex}{suffix}"
        output_path = DEFAULT_OUTPUT_DIR / filename
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def text_to_speech(
    text: str,
    *,
    voice: Optional[str] = None,
    output_path: Optional[Union[Path, str]] = None,
) -> TextToSpeechResult:
    """Convert ``text`` into speech using OpenAI's TTS API.

    Args:
        text: The text that should be synthesized into audio.
        voice: Optional override for the configured voice.
        output_path: Optional path where the audio file should be stored.

    Returns:
        :class:`TextToSpeechResult` describing where the audio was stored.
    """

    if not text or not text.strip():
        raise ValueError("text must be a non-empty string")

    client = _build_client()
    output_file = _ensure_output_path(output_path)

    tts_model = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
    voice_name = voice or os.getenv("TTS_VOICE", "alloy")

    LOGGER.info("Synthesizing speech using model '%s'", tts_model)

    try:
        with client.audio.speech.with_streaming_response.create(
            model=tts_model,
            voice=voice_name,
            input=text,
            format=DEFAULT_TTS_FORMAT.lstrip("."),
        ) as response:
            response.stream_to_file(output_file)
    except AttributeError:
        # Fallback for older versions of the OpenAI client that do not yet
        # support streaming responses.
        response = client.audio.speech.create(
            model=tts_model,
            voice=voice_name,
            input=text,
            format=DEFAULT_TTS_FORMAT.lstrip("."),
        )
        if hasattr(response, "stream_to_file"):
            response.stream_to_file(output_file)
        else:
            audio_content = getattr(response, "data", None) or response
            if hasattr(audio_content, "read"):
                output_file.write_bytes(audio_content.read())
            elif isinstance(audio_content, (bytes, bytearray)):
                output_file.write_bytes(audio_content)
            else:  # pragma: no cover - defensive fallback
                raise RuntimeError("Unexpected TTS response format")

    LOGGER.info("Audio written to %s", output_file)
    return TextToSpeechResult(text=text, audio_path=output_file)


def _play_audio(audio_path: Path) -> None:
    """Play an audio file if playback dependencies are installed."""

    if simpleaudio is None:
        LOGGER.warning(
            "simpleaudio is not installed; skipping playback of %s", audio_path
        )
        return

    if audio_path.suffix.lower() not in {".wav", ".wave"}:
        LOGGER.warning(
            "Audio playback skipped; simpleaudio supports WAV files only (got %s)",
            audio_path.suffix,
        )
        return

    try:
        wave_obj = simpleaudio.WaveObject.from_wave_file(str(audio_path))
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except simpleaudio.SimpleAudioError as exc:  # pragma: no cover - runtime error
        LOGGER.error("Unable to play audio: %s", exc)


def speech_to_text(
    audio_path: Union[Path, str], *, prompt: Optional[str] = None
) -> SpeechToTextResult:
    """Transcribe speech from ``audio_path`` using Whisper."""

    path = Path(audio_path)

    if not path.exists():
        raise FileNotFoundError(path)

    client = _build_client()
    stt_model = os.getenv("STT_MODEL", "whisper-1")

    LOGGER.info("Transcribing %s using model '%s'", audio_path, stt_model)

    with path.open("rb") as audio_file:
        response = client.audio.transcriptions.create(
            model=stt_model,
            file=audio_file,
            prompt=prompt,
        )

    transcript = response.text if hasattr(response, "text") else response.get("text")
    if not transcript:
        raise RuntimeError("No transcription returned from the API")

    LOGGER.info("Transcription complete: %s", transcript)
    return SpeechToTextResult(audio_path=path, transcript=transcript)


def chat_with_model(prompt_text: str) -> ChatResult:
    """Send ``prompt_text`` to a chat model and return the response."""

    if not prompt_text or not prompt_text.strip():
        raise ValueError("prompt_text must be a non-empty string")

    client = _build_client()
    chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini")

    LOGGER.info("Requesting chat response using model '%s'", chat_model)

    try:
        response = client.responses.create(
            model=chat_model,
            input=[{"role": "user", "content": prompt_text}],
        )
        message = response.output_text
    except AttributeError:
        completion = client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": prompt_text}],
        )
        message_payload = completion.choices[0].message
        if isinstance(message_payload, dict):
            message = message_payload.get("content")
        else:
            message = getattr(message_payload, "content", None)

    if not message:
        raise RuntimeError("Chat model returned an empty response")

    LOGGER.info("Chat response received: %s", message)
    return ChatResult(prompt=prompt_text, response=message)


def run_conversation(user_text: str) -> ChatResult:
    """Run an end-to-end conversation cycle.

    The flow is: user text -> text to speech -> playback -> speech to text -> chat
    response -> text to speech -> playback.

    Args:
        user_text: The text input from the user.

    Returns:
        The chat result from the model.
    """

    LOGGER.info("Starting conversation with user text: %s", user_text)

    tts_result = text_to_speech(user_text)
    _play_audio(tts_result.audio_path)

    stt_result = speech_to_text(tts_result.audio_path)
    chat_result = chat_with_model(stt_result.transcript)

    LOGGER.info("Synthesizing chat response")
    response_tts = text_to_speech(chat_result.response)
    _play_audio(response_tts.audio_path)

    LOGGER.info("Conversation flow complete")
    return chat_result


def main() -> int:
    """CLI entry point for manual testing of the conversation flow."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    user_input = input("Enter text to start the conversation: ")

    try:
        result = run_conversation(user_input)
    except Exception as exc:  # pragma: no cover - CLI convenience
        LOGGER.error("Conversation failed: %s", exc)
        return 1

    print("Model response:", result.response)
    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution
    sys.exit(main())
