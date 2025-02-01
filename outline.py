# outline.py

from ollama import chat
from ollama import ChatResponse
from vectorstore import get_all_points

def generate_outline(collection_name: str) -> str:
    """
    Generates a hierarchical outline (up to 3 levels) for the given text
    using the Ollama Python package directly (no LangChain).
    """

    model = "llama3.2"

    # 1. loop through the chunks in the collection
    #    for each chunk, call the Ollama model to generate an outline
    #    and accumulate the results into a single string

    chunks = get_all_points(collection_name)
    outline_pieces = []

    for chunk in chunks:
        # Call the Ollama model and stream the response
        response: ChatResponse = chat(model="llama3.2", messages=[
            {"role": "system", "content": f"""You are an assistant that creates a detailed summary from text. I want a general summary to be of what the keypoints are in the text."""},
            {"role": "user", "content": f"""Generate a summary for a reviewer covering the major keypoints of the text.
             
            Text: {chunk}
            Outline:
            """}
        ])

        outline_pieces.append(response.message.content)
    

    # 2. Build your prompt string for combining the individual outlines
    system_prompt = f"""You are an assistant that creates a hierarchical outline from text. 
    Provide a clear table of contents with up to 2 levels. 
    List major topics (Level 1) and subtopics (Level 2).
    """

    # 3. Generate the response. Ollama's `generate()` returns a generator
    #    that yields chunks of the response. We'll accumulate them into a single string.
    response: ChatResponse = chat(model=model, messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"""Here are partial summaries of a longer text. Please produce a single hierarchical outline (2 levels deep) summarizing the entire text.
         
        Text: {'   '.join(outline_pieces)}
        Outline:
        """}
    ])
    # print(response.message.content)
    # outline_pieces = []
    # for chunk in response['message']['content']:
    #     # chunk is typically a dict with a "response" key
    #     outline_pieces.append(chunk["response"])
    # for message in response.message:
    #     outline_pieces.append(message.content)
    # 4. Return the combined outline text
    # return "".join(outline_pieces)
    return response['message']['content']
