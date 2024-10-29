#installations needed
#pip install -qU langchain-huggingface
#pip install -qU langchain_community pypdf
#pip install -qU langchain-pinecone pinecone-notebooks

from flask import Flask, render_template, request, jsonify
from uuid import uuid4
from pinecone import Pinecone, ServerlessSpec
from pathlib import Path
import os

import time
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

from huggingface_hub import InferenceClient

#populate keys below
inference_api_key = ""
pinecone_api_key = ""



app = Flask(__name__)

path = Path(__file__).parent.absolute()


#1 object to create embeddings
#https://python.langchain.com/docs/integrations/text_embedding/huggingfacehub/
#https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=inference_api_key, model_name="sentence-transformers/all-MiniLM-l6-v2"
)



#2 PyPDF for reading and processing PDFs
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=240,
    chunk_overlap=10
)

loader = PyPDFLoader(
    str(os.path.join(path, "./example_data/5ry2FxDro2.pdf") )
)

docs = loader.load_and_split(
    text_splitter=text_splitter
)


#3 create pc client
pc = Pinecone(api_key=pinecone_api_key, 
              #proxy_url='http://proxy.server:3128')
            )
index_name = "langchain-test-index"  # change if desired
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)


#create index and store data
index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)


#4 add docs to the vector store
uuids = [str(uuid4()) for _ in range(len(docs))]
vector_store.add_documents(documents=docs, ids=uuids)



#5 retrieve data from store
results = vector_store.similarity_search(query="Biology science",k=2)
retrievedContent=""
for res in results:
    print(res.page_content)
    print(res.metadata)
    retrievedContent+=res.page_content


#completed data prepration
print("#completed data prepration...ready to serve!")

#6 generator
inferenceClient = InferenceClient(api_key=inference_api_key)
messages = [
    {"role": "system", "content": "Answer the user based on context: "+retrievedContent}, 
	{ "role": "user", "content": "Tell me a story" }
]

#https://huggingface.co/docs/api-inference/tasks/chat-completion
stream = inferenceClient.chat.completions.create(
    model="microsoft/Phi-3-mini-4k-instruct", 
	messages=messages, 
	temperature=0.5,
	max_tokens=1024,
	top_p=0.7,
	stream=False
)

answer=stream.choices[0].message.content


print("answer: "+ answer)

# Serve the HTML page
@app.route('/')
def home():
    return render_template('index.html')  # This assumes index.html is in the 'templates' folder

# API endpoint that receives a GET request
@app.route('/api', methods=['GET'])
def api():
    # Get the query parameter from the URL
    query = request.args.get('query', default="")
    print("user question: " + query)
    results = vector_store.similarity_search(query=query,k=2)
    
    retrievedContent=""
    for res in results:
        print(res.page_content)
        print(res.metadata)
        retrievedContent+=res.page_content

    #start our RAG code

    messages = [
        {"role": "system", "content": "Answer the user based on context: "+retrievedContent}, 
    	{ "role": "user", "content": query }
    ]
    stream = inferenceClient.chat.completions.create(
        model="microsoft/Phi-3-mini-4k-instruct", 
    	messages=messages, 
    	temperature=0.5,
    	max_tokens=1024,
    	top_p=0.7,
    	stream=False
    )
    
    answer=stream.choices[0].message.content

    print("answer: " + answer)
    #complete our RAG code


    response = {
        "answer": f"Generated answer: {answer}"
    }

    return jsonify(response)




# if __name__ == '__main__':
#    app.run(debug=True)