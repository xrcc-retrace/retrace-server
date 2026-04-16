"""Ephemeral token generation for Gemini Live API.

The server holds the Gemini API key and mints short-lived tokens that the
phone uses to open WebSocket connections directly to Gemini Live.
"""

import logging
from datetime import UTC, datetime, timedelta

from fastapi import HTTPException
from google import genai
from google.genai import types

from config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

GEMINI_LIVE_WSS_BASE = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1alpha."
    "GenerativeService.BidiGenerateContentConstrained"
)

# Lazy-initialized client — v1alpha required for auth_tokens API.
_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise HTTPException(500, "GEMINI_API_KEY not configured")
        _client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"api_version": "v1alpha"},
        )
    return _client


async def create_ephemeral_token(
    system_instruction: str,
    tools: list[dict],
    model: str = "gemini-2.5-flash-live-preview",
    voice: str = "Puck",
    expire_seconds: int = 3600,
    new_session_window_seconds: int = 120,
    resumption_handle: str | None = None,
) -> dict:
    """Mint an ephemeral token locked to the given session config.

    Pass `resumption_handle` to bind the new session to a prior one that
    emitted a `sessionResumptionUpdate.new_handle`. This lets the phone
    re-open a WebSocket that continues (with compressed context) from where
    the old one left off.

    Returns {"token": str, "expires_at": str, "websocket_url": str}.
    """
    now = datetime.now(UTC)
    expire_time = now + timedelta(seconds=expire_seconds)
    new_session_expire = now + timedelta(seconds=new_session_window_seconds)

    resumption_config = (
        types.SessionResumptionConfig(handle=resumption_handle)
        if resumption_handle
        else types.SessionResumptionConfig()
    )

    try:
        client = _get_client()
        token = client.auth_tokens.create(
            config=types.CreateAuthTokenConfig(
                uses=0,  # unlimited uses within expiry window
                expire_time=expire_time,
                new_session_expire_time=new_session_expire,
                live_connect_constraints=types.LiveConnectConstraints(
                    model=model,
                    config=types.LiveConnectConfig(
                        response_modalities=["AUDIO"],
                        media_resolution="MEDIA_RESOLUTION_LOW",
                        system_instruction=types.Content(
                            parts=[types.Part(text=system_instruction)]
                        ),
                        tools=[types.Tool(function_declarations=[
                            types.FunctionDeclaration(
                                name=t["name"],
                                description=t["description"],
                                parameters=t["parameters"],
                            )
                            for t in tools
                        ])],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=voice,
                                )
                            )
                        ),
                        context_window_compression=types.ContextWindowCompressionConfig(
                            trigger_tokens=100_000,
                            sliding_window=types.SlidingWindow(
                                target_tokens=40_000,
                            ),
                        ),
                        session_resumption=resumption_config,
                        proactivity=types.ProactivityConfig(
                            proactive_audio=True,
                        ),
                        input_audio_transcription=types.AudioTranscriptionConfig(),
                        output_audio_transcription=types.AudioTranscriptionConfig(),
                    ),
                ),
            ),
        )
    except Exception as exc:
        logger.error("Failed to create ephemeral token: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"Could not create Gemini ephemeral token: {exc}",
        ) from exc

    token_value = token.name
    expires_at = expire_time.isoformat()
    websocket_url = f"{GEMINI_LIVE_WSS_BASE}?access_token={token_value}"

    logger.info(
        "Created ephemeral token (expires %s, model=%s, resumption=%s)",
        expires_at,
        model,
        "yes" if resumption_handle else "no",
    )

    return {
        "token": token_value,
        "expires_at": expires_at,
        "websocket_url": websocket_url,
    }
