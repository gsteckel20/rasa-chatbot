import streamlit as st
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, BertTokenizerFast, BertForQuestionAnswering
import json

# Read in the complete Q&A dataset
with open("physics.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f) 

# Extract the context (think of it like a text book of knowldge) and store in a list
contexts = []
for item in qa_data:
    
    context = item.get('context')   # Safely access the 'context' key
    question = item.get('question')  # Safely access the 'question' key
    answer = item.get('answer')      # Safely access the 'answer' key
    if 'answer' == None:
        print(item)
    # create context
    contexts.append(context + answer)
    
# Load a pre-trained sentence-transformer model for retrieval
retrieval_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Load the fine-tuned BERT-QA model and tokenizer
model_name = "./checkpoint-4000/"  # Replace with your fine-tuned model
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Initialize the question-answering pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# A collection of physics answers (knowledge base) stored in 'contexts'
contexts = contexts

# Encode the knowledge base for retrieval
knowledge_embeddings = retrieval_model.encode(contexts, convert_to_tensor=True)

# Function to retrieve the most relevant context based on the question
def retrieve_context(question):
    # Encode the question for similarity search
    question_embedding = retrieval_model.encode(question, convert_to_tensor=True)
    
    # Compute cosine similarities between the question and the knowledge base
    similarities = util.pytorch_cos_sim(question_embedding, knowledge_embeddings)
    
    # Find the most similar context from the knowledge base
    closest_context_idx = similarities.argmax().item()
    closest_context = contexts[closest_context_idx]
    
    return closest_context

# Chatbot function
def chatbot(question):
    while True:
        
        # Retrieve the most relevant context from the knowledge base
        context = retrieve_context(question)
        
        # Extract the answer using the fine-tuned BERT-QA model
        result = qa_pipeline(question=question, context=context,max_length=50)
        
        answer = result['answer']
        answer = answer + '.' + context
        
        return answer

# Streamlit part

# Initialize session state to store messages
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Set Streamlit page configuration
st.set_page_config(
    page_title="Q&A Chatbot",
    page_icon="💬",
    layout="centered",
)

# App title
st.title("💬 Physics Q&A Chatbot")

# Chat container
chat_container = st.container()

# User input
with st.form(key='chat_form', clear_on_submit=True):
    user_input = st.text_input("You:", "")
    submit_button = st.form_submit_button(label='Send')

if submit_button and user_input:
    # Append user message to the conversation
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Get bot response
    bot_response = chatbot(user_input)
    
    # Append bot message to the conversation
    st.session_state.messages.append({"role": "bot", "content": bot_response})

# Display the conversation
with chat_container:
    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            # User message
            user_col, empty_col = st.columns([4, 1])
            with user_col:
                st.markdown(f"**You:** {msg['content']}")
        else:
            # Bot message
            empty_col, bot_col = st.columns([1, 4])
            with bot_col:
                st.markdown(f"**Bot:** {msg['content']}")

# Optional: Add some spacing at the bottom
st.markdown("<br>", unsafe_allow_html=True)
