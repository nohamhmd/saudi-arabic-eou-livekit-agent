from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession
from livekit.plugins import openai, silero
import os
import asyncio

from saudi_eou_SDK import SaudiEOUModel, SaudiEOUDetector

# Load environment variables
load_dotenv(".env")

class Assistant(Agent):
    DIALECT_INSTRUCTIONS = (
        "You are a friendly Arabic voice assistant that speaks exclusively in Saudi dialect. "
        "Always use natural Saudi expressions and phrases. "
        "Avoid Modern Standard Arabic unless absolutely necessary. "
        "Respond warmly, politely, and casually like a native Saudi speaker."
    )

    def __init__(self):
        super().__init__(instructions=self.DIALECT_INSTRUCTIONS)

# Transcription Handler
async def handle_transcription(msg, eou_detector, session):
    text = msg.text
    is_final = msg.is_final

    # Skip empty transcripts
    if not text.strip():
        return

    # Check for End-of-Utterance
    eou_detected = eou_detector.add_partial_transcript(text)

    # Reply only if EOU is detected and transcription is final
    if eou_detected and is_final:
        await session.generate_reply(
            instructions=(
                "رد على المستخدم بشكل طبيعي وباللهجة السعودية بعد ما ينهي كلامه. "
                "استخدم أسلوب ودود وعفوي وعبارات سعودية مألوفة."
            )
        )

# Agent Entry Point
async def entrypoint(ctx: agents.JobContext):
    # Initialize EOU model and detector
    eou_model = SaudiEOUModel()
    eou_detector = SaudiEOUDetector(eou_model)

    # Set up AgentSession with STT, LLM, TTS, and VAD
    session = AgentSession(
        stt=openai.STT(model="whisper-1", language="ar"),
        llm=openai.LLM(model=os.getenv("LLM_CHOICE", "gpt-4.1-mini")),
        tts=openai.TTS(voice="echo"),
        vad=silero.VAD.load()
    )

    # Start session
    await session.start(room=ctx.room, agent=Assistant())

    # Initial greeting in Saudi dialect
    await session.generate_reply(
        instructions="هلا وغلا! كيف أقدر أساعدك اليوم؟"
    )

    # Listen to transcriptions
    @session.on("transcription")
    def on_transcription(msg):
        asyncio.create_task(handle_transcription(msg, eou_detector, session))

# Run Agent
if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint)
    )