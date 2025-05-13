import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
import textwrap
from langchain.prompts import PromptTemplate
from retrieval import get_relevant_docs



#Load the model
model_path = "C:/techai/program/model"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",  # Automatically chooses GPU if available
    torch_dtype=torch.float16,  # Ensures efficiency
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

#Initialize the pipeline
llm = HuggingFacePipeline(pipeline=pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,  # Adjust as needed
))



print("========================================================================================================")
print("Model loaded successfully!")

#User input query
query = input("Enter your query: ")
print("Retrieving relevant documents...")

retriever = get_relevant_docs(query)
print("Documents retrieved successfully!")

print("Initializing RetrievalQA...")
#Answer generation using relevant data
qa = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type = "stuff",
    retriever = retriever,
    return_source_documents=False
)


print("Generating answer...")
#Run the QA model
response = qa.invoke(query)

print(response)


