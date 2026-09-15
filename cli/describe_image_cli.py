import argparse
import base64
import mimetypes
import os

from dotenv import load_dotenv
from openai import OpenAI

model = "openrouter/free"
instructions = """
Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary
"""


def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    parser = argparse.ArgumentParser(description="Image + text → rewritten query")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--query", required=True, help="Text query")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image file not found: {args.image}")

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"
    with open(args.image, "rb") as f:
        img = f.read()
    data_url = f"data:{mime};base64,{base64.b64encode(img).decode()}"

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instructions.strip()},
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": args.query.strip()},
            ],
        }
    ]

    response = client.chat.completions.create(model=model, messages=messages)
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("No text in API response")

    print(f"Rewritten query: {content.strip()}")
    if response.usage is not None:
        print(f"Total tokens:    {response.usage.total_tokens}")


if __name__ == "__main__":
    main()
