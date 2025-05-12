from flask import Flask, request, jsonify, render_template
import os
import textwrap
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from retrieval import get_relevant_docs
import torch
import re

app = Flask(__name__)

def initialize_components():
    model_path = "C:/techai/program/model"

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )

    llm = HuggingFacePipeline(pipeline=pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200
    ))

    return llm

llm = initialize_components()

# Function to extract the first "Helpful Answer"
def extract_helpful_answer(text):
    match = re.search(r"Helpful Answer:\s*(.*?)(?:\n\s*Question:|\Z)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return "Helpful Answer not found."

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    query = data.get('question', '')

    if not query:
        return jsonify({'error': 'No question provided'}), 400

    try:
        # Retrieve relevant documents using custom retriever
        retriever = get_relevant_docs(query)

        # Define QA chain
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False
        )

        # Prompt template
        template = '''You are an AI tutor designed to assist students by providing
        clear, concise, and easy-to-understand answers to their queries.

        Your task:
        1. Retrieve relevant information from the knowledge base.
        2. Understand the context and generate an informative yet simple response.
        3. Use examples and analogies when necessary to enhance understanding.
        4. If the question is unclear, rephrase it before answering.
        5. Avoid unnecessary details and technical jargon; focus on student-friendly explanations.

        Example Format:
        **Question:** What is photosynthesis?
        **Answer:** Photosynthesis is the way plants make their own food using sunlight.
        They take in sunlight, water, and air, and turn it into energy to grow. Think
        of it like a plant's way of cooking food using sunlight!

        Now, answer the following question based on the retrieved information:
        {query}'''

        prompt = PromptTemplate(template=template, input_variables=["query"])
        formatted_prompt = prompt.format(query=query)

        # Invoke QA chain
        response = qa.invoke(formatted_prompt)

        # Extract and format helpful answer
        result_text = response['result'].strip()
        helpful_answer = extract_helpful_answer(result_text)
        wrapped_response = textwrap.fill(helpful_answer, width=80)

        return jsonify({'answer': wrapped_response})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
