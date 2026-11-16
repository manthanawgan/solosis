
import json
from pathlib import Path
import os

from api.schemas import MemoryExtraction

from google import genai
from google.genai import types
from google.genai.types import HttpOptions


"""
goal extraction format: 

{
  "memories": [
    {
      "content": "User is a second-year computer science student.",
      "category": "profile",
      "source_message_ids": ["..."]
    },
    {
      "content": "User is building a monocular SLAM project in Python.",
      "category": "project",
      "source_message_ids": ["..."]
    },
    {
      "content": "User prefers code examples over theory.",
      "category": "preference",
      "source_message_ids": ["..."]
    }
  ]
}

"""


def ingestion(file_path: Path):

    with open(f"{file_path}", "r") as file:
        data = json.load(file)

        memory_candidates = []

        for session in data["sessions"]:

            for message in session["messages"]:

                candidates = {
                    "user_id": session["user_id"],
                    "session_id": session["session_id"],
                    "source_message_id": message["message_id"],
                    "role": message["role"],
                    "content": message["content"],
                    "timestamp": message["timestamp"]
                }

        memory_candidates.append(candidates)

    return memory_candidates



client = genai.Client(http_options = HttpOptions(api_version="v1"))
model = "gemini-3.8-flash"

def extract_memories(memory_candidates: list[dict]) -> MemoryExtraction:

  prompt = f"""
    You are a memory extraction component for an AI agent.

    Your task is to identify useful, persistent information about the
    user from the provided conversation messages.

    Rules:
    1. Extract only information supported by the messages.
    2. Extract atomic memories: one independently useful fact per memory.
    3. A single message can produce multiple memories.
    4. Ignore greetings, casual conversation and temporary information
       that has no long-term value.
    5. Do not treat assistant statements as facts about the user unless
       independently supported by the user's messages.
    6. Assign an importance score from 1 to 10.
    7. Assign confidence between 0 and 1.
    8. Include the IDs of the messages supporting each memory.
    9. If nothing is worth remembering, return an empty memories list.
    
    Conversation messages:
    {json.dumps(memory_candidates, ensure_ascii=False, indent=2)}
    """


  response = client.models.generate_content(
       model = model,
       contents = prompt,
       config = types.GenerateContentConfig(
          response_mime_type = "application/json",
          response_schema = MemoryExtraction,
          temperature = 0.2
       ),
    )

  return response.parsed
