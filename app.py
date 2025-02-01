import gradio as gr
import os
from ingestion import extract_text_from_pdf
from outline import generate_outline
from vectorstore import store_document_in_qdrant
from quiz import generate_study_summary, generate_test_questions
from qdrant_client import QdrantClient
from vectorstore import get_ollama_embedding, get_all_points

# Global (for demo) - in production, use a database to track user sessions, outlines, etc.
OUTLINE_CACHE = {}
DOCUMENT_TEXT = ""

def upload_document(pdf_file):
    if pdf_file is None:
        return "Please upload a PDF.", "", [], ""  # 4 outputs

    text = extract_text_from_pdf(pdf_file.name)
    store_document_in_qdrant(text, collection_name="my_collection")

    # Optional: Generate an outline from the entire text
    outline_text = generate_outline(collection_name="my_collection")

    # Retrieve all chunks from Qdrant so we can iterate over them
    chunk_list = get_all_points("my_collection")
    if chunk_list:
        first_chunk = chunk_list[0]
    else:
        first_chunk = "No chunks found."

    return "Document processed and stored!", outline_text, chunk_list, first_chunk


def study_section(section_text):
    chunks = search_similar_chunks(section_text)
    summary = generate_study_summary(chunks)
    return summary

def test_section(section_text, difficulty):
    questions = generate_test_questions(section_text, difficulty)
    return questions

def search_similar_chunks(query: str, collection="my_collection"):
    client = QdrantClient(url="http://localhost:6333")
    query_embedding = get_ollama_embedding(query)

    # Perform vector search
    results = client.query_points(
        collection_name=collection,
        query=query_embedding,
        limit=5,  # top 5 chunks
        score_threshold=0.7,
        with_payload=True
    )

    # print(f"Found {len(results)} similar chunks.")
    # print(results.payload)
    # print(type(results))
    # print(results)

    try: 
        points = results.result
        # print("error here")
    except: points = results
    # print(type(points))  # Should now be <class 'list'> of ScoredPoint

    output_chunks = []
    for point in points:
        try: 
            # print(point[1][0].payload['text'])
            output_chunks.append(point[1][0].payload['text'])
        except: pass

    return output_chunks

    # return [print(result.payload) for result in results]



# Callback for "Next" button
def show_next_chunk(chunk_list, current_index):
    if not chunk_list:
        return 0, "No chunks available."
    # clamp index
    new_index = min(current_index + 1, len(chunk_list) - 1)
    return new_index, chunk_list[new_index]

# Callback for "Previous" button
def show_prev_chunk(chunk_list, current_index):
    if not chunk_list:
        return 0, "No chunks available."
    # clamp index
    new_index = max(current_index - 1, 0)
    return new_index, chunk_list[new_index]

# --------------------
# GRADIO UI
# --------------------
with gr.Blocks() as demo:
    gr.Markdown("# Tutor/Reviewer App")

    pdf_input = gr.File(label="Upload your PDF")
    upload_btn = gr.Button("Process Document")

    status_output = gr.Textbox(label="Status", interactive=False)
    outline_output = gr.Textbox(label="Generated Outline", interactive=False, lines=10)

    # We can still keep chunk_list_state and chunk_index_state if needed.
    chunk_list_state = gr.State([])
    chunk_index_state = gr.State(0)

    # Textbox for user to paste a section
    section_input = gr.Textbox(label="Paste a section from the outline here to study/test")

    # Dropdown for question difficulty
    difficulty_dropdown = gr.Dropdown(
        choices=["Easy", "Intermediate", "Hard"], 
        label="Question Difficulty",
        value="Intermediate"  # default selection
    )

    study_btn = gr.Button("Study")
    test_btn = gr.Button("Test")

    study_output = gr.Textbox(label="Study Summary", interactive=False, lines=6)
    test_output = gr.Textbox(label="Test Questions", interactive=False, lines=6)

    # Wiring upload_document to handle PDF ingestion
    upload_btn.click(
        fn=upload_document,
        inputs=[pdf_input],
        outputs=[status_output, outline_output, chunk_list_state]
    )

    # Study
    study_btn.click(
        fn=study_section, 
        inputs=[section_input], 
        outputs=[study_output]
    )

    # Test (now takes 2 inputs: the section text + difficulty)
    test_btn.click(
        fn=test_section, 
        inputs=[section_input, difficulty_dropdown], 
        outputs=[test_output]
    )

demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)
