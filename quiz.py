# quiz.py (or study.py)
from ollama import ChatResponse, chat

def generate_study_summary(chunk_text: list, model_name="llama3.2") -> str:
    """
    Summarize the given text in a concise, study-friendly way using Ollama.
    """
    print(chunk_text)
    output_chunks = []
    for text in chunk_text:
        # Call the Ollama model and stream the response
        response: ChatResponse = chat(model=model_name, messages=[
            {"role": "system", "content": f"""You are a tutor. Summarize the text in a concise, study-friendly way."""},
            {"role": "user", "content": f"""Summarize the text in a concise, study-friendly way. text: {text}"""}
        ])

        output_chunks.append(response.message.content)
    # print(output_chunks)



    return "\n\n".join(output_chunks)


def generate_test_questions(chunk_text: str, difficulty: str, model_name="deepseek-r1:8b") -> str:
    """
    Generate 3 multiple-choice questions (each with 4 options and 1 correct answer)
    from the given text, using Ollama.
    """
    system_prompt = f"""You are a tutor. Given the text below, generate 3 multiple-choice questions.
    Each question should have 4 options (A, B, C, D), and exactly one correct answer. Make use of plausible distractors.
    """

    response_chunks: ChatResponse = chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Generate 3 {difficulty}-level multiple-choice questions from the text. text: {chunk_text}"""}
        ]
    )

    result = []
    # for chunk in response_chunks:
    #     result.append(chunk["response"])

    # return "".join(result)
    results = response_chunks.message.content
    results = results.split("</think>")[-1].strip()

    return results


def evaluate_answer(chunk_text: str, user_answer: str, model_name="llama3.2") -> str:
    """
    Evaluate the user's answer against the given text, providing feedback.
    """
    prompt = f"""Text: {chunk_text}
User's answer: {user_answer}
Check if it's correct. Provide feedback.
"""

    client = Client()
    response_chunks = client.generate(
        model=model_name,
        prompt=prompt
    )

    result = []

    return response_chunks.message.content
