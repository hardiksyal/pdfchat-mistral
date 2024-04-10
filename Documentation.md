# Local PDF Chat Application with Mistral 7B LLM, Langchain, Ollama, and Streamlit

## Introduction

The Local PDF Chat Application is a project aimed at developing a chatbot capable of answering questions related to PDF files. This chatbot utilizes state-of-the-art language models, specifically Mistral 7B LLM, and incorporates technologies such as Langchain, Ollama, and Streamlit to provide a seamless user experience.

## Project Overview

### Purpose

The primary purpose of this project is to create a chatbot that can understand user queries and provide relevant answers based on the content of PDF documents. This enables users to interact with the chatbot to obtain information from PDF files in a conversational manner.

### Technologies Used

#### Mistral 7B LLM

Mistral 7B LLM (Large Language Model) is a powerful language model that forms the backbone of the chatbot's natural language understanding and generation capabilities. It is trained on vast amounts of text data and can generate human-like responses to a wide range of queries.

#### Langchain

Langchain is a framework used for building conversational AI applications. It provides tools and libraries for managing conversational contexts, handling user inputs, and generating responses. In this project, Langchain is utilized for implementing the chatbot's conversational logic.

#### Ollama

Ollama is a tool for running large language models locally. It simplifies the setup and configuration of language models, allowing them to be deployed and used on local machines. Ollama is employed in this project to run Mistral 7B LLM locally, enabling fast and efficient inference.

#### Streamlit

Streamlit is a popular framework for building interactive web applications with Python. It provides a simple and intuitive way to create data-driven web interfaces. In this project, Streamlit is used to develop the user interface for the chatbot, allowing users to interact with the application through a web browser.

### Features

- **PDF Upload**: Users can upload PDF files containing the information they seek.
- **Conversational Interface**: The chatbot interacts with users in a conversational manner, understanding their queries and providing relevant responses.
- **Natural Language Understanding**: The chatbot utilizes Mistral 7B LLM for natural language understanding, allowing it to interpret user queries accurately.
- **Document Analysis**: Upon PDF upload, the chatbot analyzes the content of the document to extract relevant information for answering user queries.
- **Real-time Response Generation**: Responses are generated in real-time, providing users with immediate feedback during the conversation.

## Setup and Installation

### Prerequisites

Before running the application, ensure that the following prerequisites are met:

- Python 3.x installed on your system
- Docker installed (for Windows users)
- WSL 2 enabled (for Windows users)

### Installation Steps

1. Clone the GitHub repository containing the project code:

    ```bash
    git clone https://github.com/hardiksyal/pdfchat-mistral.git
    ```

2. Install the required dependencies using pip:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up Ollama locally by following the instructions provided in the Ollama GitHub repository for your respective operating system (Mac/Linux or Windows).

4. Run the Streamlit application using the following command:

    ```bash
    streamlit run app.py
    ```

5. Access the application through your web browser at the provided URL.

## Code Explanation

Now, let's explain the code used in the application:

```python
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import streamlit as st
import os
import time
```

- **Imports**: These are the necessary imports for the application, including libraries and modules for interacting with PDF files, managing conversational contexts, and building the user interface using Streamlit.

```python
if not os.path.exists('files'):
    os.mkdir('files')

if not os.path.exists('jj'):
    os.mkdir('jj')
```

- **Directory Creation**: These lines of code ensure that the required directories for storing PDF files and other data are created if they do not already exist.

```python
if 'template' not in st.session_state:
    st.session_state.template = """You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional, informative, and detailed. Make sure to carefully format your answers in readable and presentable format. If you don't know the answer just say you cannot answer as it's not in the context, don't try to make up an answer but try your best to look for the answer in the context again.\n\n
    Context: {context}
    History: {history}

    User: {question}
    Chatbot:"""

if 'prompt' not in st.session_state:
    st.session_state.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.session_state.template,
    )
```

- **Session State Initialization**: These lines initialize the session state variables used by the Streamlit application for managing conversation history, prompts, and templates.

```python
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question")
```

- **Memory Initialization**: This line initializes the conversation memory used by the chatbot to remember previous interactions with the user.

```python
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = Chroma(persist_directory='jj',
                                          embedding_function=OllamaEmbeddings(base_url='http://localhost:11434', model="mistral")
                                          )
```

- **Vector Store Initialization**: This line initializes the vector store, which is used for storing and retrieving embeddings of text data.

```python
if 'llm' not in st.session_state:
    st.session_state.llm = Ollama(base_url="http://localhost:11434",
                                  model="mistral",
                                  verbose=True,
                                  callback_manager=CallbackManager(
                                      [StreamingStdOutCallbackHandler()]),
                                  )
```

- **LLM Initialization**: This line initializes Mistral 7B LLM using Ollama, specifying the base URL and model name.

```python
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
```

- **Chat History Initialization**: This line initializes the chat history, which stores the conversation between the user and the chatbot.

```python
st.title("PDF Chatbot")
```

- **Title Display**: This line displays the title of the application, indicating that it is a PDF chatbot.

```python
uploaded_file = st.file_uploader("Upload your PDF", type='pdf')
```

- **PDF Upload**: This line creates a file uploader component using Streamlit, allowing users to upload PDF files.
```python
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])
```

- **Display Chat History**: This loop iterates over the chat history stored in the session state and displays each message using Streamlit's `chat_message` component.

```python
if uploaded_file is not None:
    if not os.path.isfile("files/"+uploaded_file.name+".pdf"):
        with st.status("Analyzing your document..."):
            bytes_data = uploaded_file.read()
            f = open("files/"+uploaded_file.name+".pdf", "wb")
            f.write(bytes_data)
            f.close()
            loader = PyPDFLoader("files/"+uploaded_file.name+".pdf")
            data = loader.load()

            # Initialize text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200,
                length_function=len
            )
            all_splits = text_splitter.split_documents(data)

            # Create and persist the vector store
            st.session_state.vectorstore = Chroma.from_documents(
                documents=all_splits,
                embedding=OllamaEmbeddings(model="mistral")
            )
            st.session_state.vectorstore.persist()
```

- **PDF Analysis**: If a PDF file is uploaded, this block of code analyzes the document's content. It checks if the PDF file has already been processed and, if not, extracts text data from the PDF using `PyPDFLoader` and splits it into chunks for processing. The text chunks are then used to create and persist a vector store using `Chroma`.

```python
st.session_state.retriever = st.session_state.vectorstore.as_retriever()
```

- **Retriever Initialization**: This line initializes the retriever component using the vector store created earlier.

```python
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = RetrievalQA.from_chain_type(
        llm=st.session_state.llm,
        chain_type='stuff',
        retriever=st.session_state.retriever,
        verbose=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": st.session_state.prompt,
            "memory": st.session_state.memory,
        }
    )
```

- **QA Chain Initialization**: This block of code initializes the question-answering chain using `RetrievalQA`. It configures the components required for question answering, including the language model, retriever, prompt, and memory.

```python
if user_input := st.chat_input("You:", key="user_input"):
    user_message = {"role": "user", "message": user_input}
    st.session_state.chat_history.append(user_message)
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Assistant is typing..."):
            response = st.session_state.qa_chain(user_input)
        message_placeholder = st.empty()
        full_response = ""
        for chunk in response['result'].split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    chatbot_message = {"role": "assistant", "message": response['result']}
    st.session_state.chat_history.append(chatbot_message)
```

- **User Interaction**: This block of code handles user interaction with the chatbot. It displays a chat input box for the user to input their queries. Upon receiving input, the chatbot generates a response using the initialized question-answering chain. The response is then displayed in the chat interface, simulating typing with a blinking cursor effect.

```python
else:
    st.write("Please upload a PDF file.")
```

- **No PDF Uploaded Message**: If no PDF file is uploaded, this line displays a message prompting the user to upload a PDF file.

---

This concludes the explanation of the code used in the application. The provided code implements the functionality of the PDF chatbot, including PDF analysis, question-answering, and user interaction through the Streamlit interface.

---
