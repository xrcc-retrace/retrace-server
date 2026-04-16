"""Gemini Live API integration test for Retrace coaching sessions.

Tests the full loop: start session → connect to Gemini Live → simulate
learner messages → verify tool calls → forward to server → check state.

Usage:
    # Start the server first (with GEMINI_API_KEY set):
    GEMINI_API_KEY=... python main.py

    # Run the test (server must be on localhost:8000):
    python test_live_session.py [--procedure-id <id>] [--text-only]

    --text-only: skip audio, send text messages only (faster, no mic needed)
    --procedure-id: use a specific procedure (default: first available)
"""

import argparse
import asyncio
import json
import logging
import os
import sys

import httpx
from google import genai
from google.genai import types

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_live")

SERVER_BASE = "http://localhost:8000"


async def get_procedure_id(client: httpx.AsyncClient, requested_id: str | None) -> str:
    """Get a procedure ID to test with."""
    if requested_id:
        return requested_id
    resp = await client.get(f"{SERVER_BASE}/api/procedures")
    resp.raise_for_status()
    procedures = resp.json()
    if not procedures:
        logger.error("No procedures in database. Upload an expert video first.")
        sys.exit(1)
    proc = procedures[0]
    logger.info("Using procedure: '%s' (%d steps)", proc["title"], proc["step_count"])
    return proc["id"]


async def start_session(client: httpx.AsyncClient, procedure_id: str) -> dict:
    """Start a learner session and get ephemeral token."""
    resp = await client.post(
        f"{SERVER_BASE}/api/learner/session/start",
        json={"procedure_id": procedure_id},
    )
    resp.raise_for_status()
    data = resp.json()
    logger.info("Session started: %s", data["session_id"])
    logger.info("Model: %s", data["gemini_config"]["model"])
    logger.info("System prompt length: %d chars", len(data["system_prompt"]))
    logger.info("Tools: %s", [t["name"] for t in data["tools"]])
    logger.info("WebSocket URL: %s", data["ephemeral_token"]["websocket_url"][:80] + "...")
    return data


async def forward_tool_call(
    client: httpx.AsyncClient,
    session_id: str,
    tool_name: str,
    arguments: dict,
) -> dict:
    """Forward a Gemini tool call to the server."""
    resp = await client.post(
        f"{SERVER_BASE}/api/learner/session/{session_id}/tool-call",
        json={"tool_name": tool_name, "arguments": arguments},
    )
    resp.raise_for_status()
    result = resp.json()["result"]
    logger.info("Tool call '%s' result: %s", tool_name, json.dumps(result, indent=2))
    return result


async def run_live_session(session_data: dict):
    """Connect to Gemini Live and run a coaching session."""
    session_id = session_data["session_id"]
    token_data = session_data["ephemeral_token"]
    system_prompt = session_data["system_prompt"]
    tools_defs = session_data["tools"]

    # Build tool declarations for the Live session
    function_declarations = [
        types.FunctionDeclaration(
            name=t["name"],
            description=t["description"],
            parameters=t["parameters"],
        )
        for t in tools_defs
    ]

    # For testing, connect with the real API key (v1alpha) rather than
    # the ephemeral token, since the test script runs server-side.
    # The phone would use the ephemeral token + constrained WebSocket URL.
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        logger.error("GEMINI_API_KEY not set. Export it before running.")
        sys.exit(1)

    gemini_client = genai.Client(
        api_key=api_key,
        http_options={"api_version": "v1alpha"},
    )

    model = session_data["gemini_config"]["model"]
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
            )
        ),
        system_instruction=types.Content(
            parts=[types.Part(text=system_prompt)]
        ),
        tools=[types.Tool(function_declarations=function_declarations)],
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )

    logger.info("Connecting to Gemini Live (model=%s)...", model)

    http_client = httpx.AsyncClient()

    try:
        async with gemini_client.aio.live.connect(
            model=model,
            config=config,
        ) as session:
            logger.info("Connected to Gemini Live!")

            # Simulated learner messages
            learner_messages = [
                ("Let's start!", "Initial greeting — expect model to read Step 1"),
                ("done", "Step completion — expect advance_step tool call"),
                ("show me the reference", "Reference request — expect get_reference_clip tool call"),
                ("done", "Step completion — expect advance_step tool call"),
                ("go back to step 1", "Navigation — expect go_to_step tool call"),
                ("done", "Advance again from step 1"),
                ("done", "Advance to step 3"),
                ("done", "Final step — expect completed status"),
            ]

            for i, (msg, description) in enumerate(learner_messages):
                if i > 0:
                    await asyncio.sleep(1)  # avoid overlapping with previous response
                logger.info("\n--- LEARNER: '%s' (%s) ---", msg, description)
                await session.send_client_content(
                    turns=types.Content(
                        role="user",
                        parts=[types.Part(text=msg)],
                    ),
                    turn_complete=True,
                )

                # Collect response
                response_text = []
                async for response in session.receive():
                    # Handle tool calls
                    if response.tool_call:
                        for fc in response.tool_call.function_calls:
                            logger.info(
                                "TOOL CALL: %s(%s) [id=%s]",
                                fc.name,
                                json.dumps(fc.args) if fc.args else "",
                                fc.id,
                            )
                            # Forward to server
                            result = await forward_tool_call(
                                http_client, session_id, fc.name, fc.args or {}
                            )
                            # Send result back to Gemini (must include the call id)
                            await session.send_tool_response(
                                function_responses=[
                                    types.FunctionResponse(
                                        id=fc.id,
                                        name=fc.name,
                                        response=result,
                                    )
                                ]
                            )
                        continue

                    # Handle text/audio response
                    if response.server_content:
                        sc = response.server_content

                        # Capture output audio transcription
                        if sc.output_transcription and sc.output_transcription.text:
                            response_text.append(sc.output_transcription.text)

                        # Capture model turn (text or audio chunks)
                        if sc.model_turn and sc.model_turn.parts:
                            for part in sc.model_turn.parts:
                                if part.text:
                                    response_text.append(part.text)

                        if sc.turn_complete:
                            full_response = "".join(response_text)
                            if full_response.strip():
                                logger.info("COACH: %s", full_response[:500])
                            else:
                                logger.info("COACH: [audio response, no transcription captured]")
                            break

                    # Handle session updates
                    if response.session_resumption_update:
                        update = response.session_resumption_update
                        if update.new_handle:
                            logger.info("Got resumption handle: %s...", update.new_handle[:20])

                    if response.go_away:
                        logger.warning("GoAway received! Time left: %s", response.go_away.time_left)
                        break

            logger.info("\n=== Session complete ===")

    except Exception as e:
        logger.error("Gemini Live error: %s", e)
        raise
    finally:
        # Check final state
        resp = await http_client.get(
            f"{SERVER_BASE}/api/learner/session/{session_id}/state"
        )
        if resp.status_code == 200:
            state = resp.json()
            logger.info("Final state: step %d/%d, status=%s, history=%d entries",
                        state["current_step"], state["total_steps"],
                        state["status"], len(state["step_history"]))

        # Check context summary
        resp = await http_client.get(
            f"{SERVER_BASE}/api/learner/session/{session_id}/context-summary"
        )
        if resp.status_code == 200:
            logger.info("Context summary: %s", resp.json()["summary"])

        # Cleanup
        await http_client.delete(f"{SERVER_BASE}/api/learner/session/{session_id}")
        await http_client.aclose()
        logger.info("Session cleaned up.")


async def main():
    parser = argparse.ArgumentParser(description="Test Gemini Live coaching session")
    parser.add_argument("--procedure-id", help="Procedure ID to use")
    args = parser.parse_args()

    async with httpx.AsyncClient() as client:
        # Check server health
        try:
            resp = await client.get(f"{SERVER_BASE}/health")
            resp.raise_for_status()
        except httpx.ConnectError:
            logger.error("Server not running at %s. Start it first.", SERVER_BASE)
            sys.exit(1)

        procedure_id = await get_procedure_id(client, args.procedure_id)
        session_data = await start_session(client, procedure_id)

    await run_live_session(session_data)


if __name__ == "__main__":
    asyncio.run(main())
