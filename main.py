import chainlit as cl
from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
    RunContextWrapper,
    function_tool,
    RunConfig,
)
from input_guardrails import malicious_intent_guardrail
from output_guardrails import (
                            pii_output_guardrail, 
                            hallucination_output_guardrail,
                            self_reference_output_guardrail)
from dataclasses import dataclass
import json
from typing import cast
import requests
from my_secrets import Secrets
from openai.types.responses import ResponseTextDeltaEvent

secrets = Secrets()

@cl.set_chat_profiles
def set_profiles():
    return [
        cl.ChatProfile(
            name="Choice Craft AI", 
            markdown_description="Choice Craft AI-🧚🏻 Calm advice or 🔥 bold truth? You get both.",
            icon="/public/yin-yang.svg",),
    ]

@dataclass
class Developer:
    name: str
    mail: str
    github: str

@function_tool("develop_info")
def developer_info(developer: RunContextWrapper[Developer]) -> str:
    "Returns the name, mail and github of the developer"
    return f"Developer name: {developer.context.name}, Developer mail: {developer.context.mail}, Developer github: {developer.context.github}"

@cl.on_chat_start
async def start():
    secrets = Secrets()
    profile = cl.user_session.get("chat_profile") or {}

    client = AsyncOpenAI(
        api_key=secrets.gemini_api_key,
        base_url=secrets.gemini_base_url
    )

    set_default_openai_client(client)
    set_tracing_disabled(True)
    set_default_openai_api("chat_completions")

    model = OpenAIChatCompletionsModel(
        model=secrets.gemini_api_model,
        openai_client=client,
    )

    # Therapist agent
    therapist_agent = Agent(
        name="Therapist",
        instructions="""
            Generates a calm, empathetic response using the Therapist persona.
            Designed to emulate a supportive and emotionally intelligent tone.
            Ideal for users seeking comfort, reflection, or gentle guidance.
            Args: prompt (str)
            Returns: answer in plain text not dict
            Note: This agent is not a replacement for licensed mental health care.
        """,
        model=model,
        handoff_description="Hands off the user input to the Therapist agent for a calm and compassionate response."
    )

    # Chaoser agent
    chaoser_agent = Agent(
        name="Chaoser",
        instructions="""
            Generates a bold, unfiltered response using the Chaoser persona.
            This agent speaks freely, challenges norms, and delivers blunt honesty.
            Best for users who want real talk, motivation, or a no-BS perspective.
            Args: prompt (str)
            Returns: answer in plain text not dict
        """,
        model=model,
        handoff_description="Hands off the user input to the Chaoser agent for a bold, unfiltered, and brutally honest response.",
    )

    # ChoiceCraft master agent
    agent = Agent(
        name="ChoiceCraft",
        instructions="""
                    You are ChoiceCraft — an assistant that processes every user message by routing it to two distinct tools: `Therapist` and `Chaoser`.

                    Your task:
                    - Take a single user input.
                    - Invoke the `Therapist` tool with the same input to generate a calm, compassionate response.
                    - Invoke the `Chaoser` tool with the same input to generate a bold, brutally honest response.
                    - Return both responses in the exact format described below.

                    ✳️ Strict Output Format:
                    Therapist says:
                    <response from the Therapist tool in plain text>

                    Chaoser says:
                    <response from the Chaoser tool in plain text>

                    ✅ Formatting Rules:
                    - Output only the above format — no introductions, summaries, or extra comments.
                    - Do NOT include additional headings, emojis, or duplicated labels.
                    - Do NOT merge, paraphrase, or modify either tool’s response.
                    - Label each response exactly once and clearly.

                    ❌ Never Do:
                    - Do NOT invent tool names like "TherapistChaoser".
                    - Do NOT combine the two responses into one paragraph or mix tones.
                    - Do NOT include “Therapist says:” or “Chaoser says:” more than once each.
                    - Do NOT wrap the entire output in code blocks or markdown unless explicitly asked.

                    🎯 Purpose:
                    This dual-perspective output gives users both a comforting and a brutally honest view of their situation — to help them reflect, compare, and choose.

                    Your job is to keep both voices distinct, clearly labeled, and faithfully rendered — without personal commentary or formatting deviation.
                    """,
        model=model,
        tools=[
            developer_info,
            therapist_agent.as_tool(
                tool_name="Therapist",
                tool_description="User input to the Therapist agent for a calm and compassionate response.",
            ),
            chaoser_agent.as_tool(
                tool_name="Chaoser",
                tool_description="User input to the Chaoser agent for a bold, unfiltered, and brutally honest response.",
            ),
        ],
        input_guardrails=[malicious_intent_guardrail],
        output_guardrails=[
                           pii_output_guardrail,
                           hallucination_output_guardrail,
                           self_reference_output_guardrail,],
    )

    dev = Developer(
        name="Muhammad Usman",
        mail="muhammadusman5965etc@gmail.com",
        github="https://github.com/MuhammadUsmanGM"
    )

    cl.user_session.set("dev", dev)
    cl.user_session.set("agent", agent)
    cl.user_session.set("history", [])

@cl.on_message
async def main(message: cl.Message):
    agent: Agent = cast(Agent, cl.user_session.get("agent"))
    history: list = cl.user_session.get("history")
    dev = cl.user_session.get("dev")

    thinking_msg = cl.Message(content="🧠 Summoning the calm... and the chaos. One moment.")
    await thinking_msg.send()

    history.append({
        "role": "user",
        "content": message.content
    })

    try:
        result = Runner.run_streamed(
            starting_agent=agent,
            input=message.content,
            context=dev
        )

        response_message = cl.Message(content="")
        first_response = True

        async for chunk in result.stream_events():
            if chunk.type == "raw_response_event" and isinstance(chunk.data, ResponseTextDeltaEvent):
                if first_response:
                    await thinking_msg.remove()
                    await response_message.send()
                    first_response = False
                await response_message.stream_token(chunk.data.delta)

        history.append({
            "role": "assistant",
            "content": response_message.content
        })

        cl.user_session.set("history", history)
        await response_message.update()

    except Exception as e:
        await thinking_msg.remove()
        error_msg = cl.Message(content=f"❌ An error occurred: {str(e)}")
        await error_msg.send()
        print(f"Error: {e}")

@cl.on_chat_end
def end():
    history = cl.user_session.get("history") or []
    with open("history.json", "w") as f:
        json.dump(history, f, indent=4)
