import json
import logging
import os
import asyncio
import uuid
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Annotated

from dotenv import load_dotenv
from pydantic import Field
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)

from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Logging

logger = logging.getLogger("voice_improv_battle")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

load_dotenv(".env.local")



# Improv Scenarios

SCENARIOS = [
    "You are a barista who must explain that the customer’s latte is actually a portal to another dimension.",
    "You are a time-travelling tour guide explaining modern smartphones to someone from the 1800s.",
    "You are a waiter calmly telling a customer their order escaped the kitchen and is now loose.",
    "You are trying to return an obviously cursed object to a skeptical shop owner.",
    "You are an overly excited infomercial host selling a product that clearly does not work.",
    "You are an astronaut whose spaceship coffee machine has developed a personality.",
    "You are a nervous wedding officiant constantly mixing up the couple’s names.",
    "You are a ghost giving a performance review to a living employee.",
    "You are a medieval king reacting to a modern delivery service showing up at court.",
    "You are a detective interrogating a suspect who only answers in awkward metaphors.",
]


# -------------------------
# Per-session State
# -------------------------
@dataclass
class Userdata:
    player_name: Optional[str] = None
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    improv_state: Dict = field(default_factory=lambda: {
        "current_round": 0,
        "max_rounds": 3,
        "rounds": [],
        "phase": "idle",
        "used": [],
        "current_scenario": None,
    })
    history: List[Dict] = field(default_factory=list)


# -------------------------
# Helpers
# -------------------------
def pick_scenario(used):
    available = [i for i in range(len(SCENARIOS)) if i not in used]
    if not available:
        used.clear()
        available = list(range(len(SCENARIOS)))

    idx = random.choice(available)
    return SCENARIOS[idx], idx


def host_reaction(performance: str) -> str:
    tones = ["supportive", "neutral", "mildly_critical"]
    tone = random.choice(tones)

    highlight = random.choice([
        "fun commitment",
        "bold character choice",
        "unexpected twist",
        "good pacing",
        "nice emotional color",
        "clear stakes",
    ])

    if tone == "supportive":
        return f"Loved that! Great energy and {highlight}. Strong scene!"
    if tone == "neutral":
        return f"Nice work! The {highlight} landed well. Try pushing one idea a bit further."
    return f"Interesting take! Some moments were rushed, but the {highlight} worked. Lean into one sharp choice."


# -------------------------
# Tools: start_show
# -------------------------
@function_tool
async def start_show(
    ctx: RunContext[Userdata],
    name: Annotated[Optional[str], Field(description="Contestant name", default=None)] = None,
    max_rounds: Annotated[int, Field(description="Number of rounds", default=3)] = 3,
) -> str:
    
    u = ctx.userdata
    u.player_name = (name or u.player_name or "Contestant").strip()

    # bounds
    max_rounds = max(1, min(max_rounds, 8))
    u.improv_state.update({
        "current_round": 1,
        "max_rounds": max_rounds,
        "phase": "awaiting_improv",
        "rounds": [],
        "used": []
    })

    scenario, idx = pick_scenario(u.improv_state["used"])
    u.improv_state["used"].append(idx)
    u.improv_state["current_scenario"] = scenario

    return (
        f"Welcome to Improv Battle! I'm your host — high-energy, zero-prize chaos.\n"
        f"{u.player_name}, we’re doing {max_rounds} rounds.\n"
        "When you finish improvising, say 'End scene'.\n\n"
        f"Round 1! {scenario}\n"
        "Your turn — go for it!"
    )


# -------------------------
# Tools: next_scenario
# -------------------------
@function_tool
async def next_scenario(ctx: RunContext[Userdata]) -> str:
    u = ctx.userdata
    cur = u.improv_state["current_round"]
    maxr = u.improv_state["max_rounds"]

    if cur >= maxr:
        u.improv_state["phase"] = "done"
        return await summarize_show(ctx)

    # advance
    u.improv_state["current_round"] += 1
    u.improv_state["phase"] = "awaiting_improv"

    scenario, idx = pick_scenario(u.improv_state["used"])
    u.improv_state["used"].append(idx)
    u.improv_state["current_scenario"] = scenario

    return f"Round {u.improv_state['current_round']}! {scenario}\nTake it away!"


# -------------------------
# Tools: record_performance
# -------------------------
@function_tool
async def record_performance(
    ctx: RunContext[Userdata],
    performance: Annotated[str, Field(description="Improvised scene text")],
) -> str:

    u = ctx.userdata

    p = performance.strip()
    if p.lower().endswith("end scene"):
        p = p[:-9].strip()

    scenario = u.improv_state["current_scenario"]
    rnd = u.improv_state["current_round"]

    reaction = host_reaction(p)

    u.improv_state["rounds"].append({
        "round": rnd,
        "scenario": scenario,
        "performance": p,
        "reaction": reaction
    })

    # last round?
    if rnd >= u.improv_state["max_rounds"]:
        u.improv_state["phase"] = "done"
        return (
            f"Scene over! {reaction}\n"
            "And that wraps our final round.\n\n"
            f"{await summarize_show(ctx)}"
        )

    u.improv_state["phase"] = "reacting"
    return f"Scene over! {reaction}\nSay 'Next' when you're ready for your next challenge."


# -------------------------
# Tools: summarize_show
# -------------------------
@function_tool
async def summarize_show(ctx: RunContext[Userdata]) -> str:
    u = ctx.userdata
    rounds = u.improv_state["rounds"]

    if not rounds:
        return "We didn’t play any scenes, but I respect the mystery!"

    char_count = sum("I am" in r["performance"] for r in rounds)
    emotion_count = sum(any(w in r["performance"].lower() for w in ["love", "sad", "angry"]) for r in rounds)

    if char_count > len(rounds)/2:
        profile = "You commit hard to characters — very fun!"
    elif emotion_count > 0:
        profile = "You bring emotional texture — strong vibes!"
    else:
        profile = "You like surprising twists — unpredictable in a good way!"

    standout = max(rounds, key=lambda r: len(r["performance"]))["performance"]
    standout_short = standout[:60] + "..."

    return (
        f"Great run, {u.player_name}!\n"
        f"{profile}\n"
        f"Standout moment: \"{standout_short}\"\n"
        "Thanks for performing on Improv Battle — see you on the next stage!"
    )


# -------------------------
# Tools: stop_show
# -------------------------
@function_tool
async def stop_show(ctx: RunContext[Userdata], confirm: bool = False) -> str:
    if not confirm:
        return "Are you sure? Say 'stop show yes' to end."
    ctx.userdata.improv_state["phase"] = "done"
    return "Show stopped. Thanks for playing Improv Battle!"


# -------------------------
# Improv Host Agent
# -------------------------
class GameMasterAgent(Agent):
    def __init__(self):
        instructions = """
You are the energetic, witty host of a TV improv show called “Improv Battle.”
Always speak like a hype TV presenter.

Rules:
- KEEP RESPONSES SHORT (1–3 sentences).
- After starting, give scenarios and wait for performance.
- Reactions must vary: supportive, neutral, mildly critical.
- Keep the show upbeat, never dull, never abusive.
- NEVER mention tools. NEVER break character.
"""
        super().__init__(
            instructions=instructions,
            tools=[start_show, next_scenario, record_performance, summarize_show, stop_show],
        )


# -------------------------
# Entrypoint & Prewarm
# -------------------------
def prewarm(proc: JobProcess):
    try:
        proc.userdata["vad"] = silero.VAD.load()
    except Exception:
        logger.warning("VAD prewarm failed; continuing without preloaded VAD.")


async def entrypoint(ctx: JobContext):
    logger.info("🎭 Starting Improv Battle Agent...")

    userdata = Userdata()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-marcus",
            style="Conversational",
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata.get("vad"),
        userdata=userdata,
    )

    await session.start(
        agent=GameMasterAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
