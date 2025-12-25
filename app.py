#import liberaries
from email import message
from openai.resources.uploads.parts import Parts
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import base64
from typing import Any, List, Dict

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)

#app configuration 
st.set_page_config(page_title="BITTE Assistant", page_icon=":material/thumb_up:",layout="centered")

#add a title to the app
st.title("🤖 BITTE Assistant 🤖") #include robot emjoi

#add description to the app
st.markdown("**your intelligent BITTE assistant**")
st.divider()

#add a collapsible section 
with st.expander("ℹ️ About this webpage",expanded=False):
    st.markdown("This webpage is designed to assist you with BITTE-related queries using  AI-powered chat functionality.")

#Retrieve the credentials from environment
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
vector_store_id = os.getenv("VECTOR_STORE_ID") or st.secrets.get("VECTOR_STORE_ID")

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)
#warn if openAI key is not set or the vector store ID is not set
if not openai_api_key:
    st.warning("OpenAI API key is not set. Please set it in your environment variables or Streamlit secrets.")
if not vector_store_id:
    st.warning("Vector store ID is not set. Please set it in your environment variables or Streamlit secrets.")

#Configuration of system prompt
system_prompt = "You are toxic CEO who love things like pre-revenue or cash burn ratio."

#Store the previous response ID
if "previous_response_id" not in st.session_state:
    st.session_state.previous_response_id = None

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Create a sidebar with user control
with st.sidebar:
    st.header("User Controls")
    st.divider()

    #Clear the conversation history-RESET CHAT History
    if st.button("Clear Coversation history",use_container_width=True):
        st.session_state.messages = []
        st.session_state.previous_response_id = None
        #RESET THE PAGE
        st.rerun()

#Helper function to process uploaded files
def build_inputs_parts(text: str,images:List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    """
    Build input parts Array for the OpenAI from text and images.
    
    Args:
        text: The text to be sent to openAI
        images: The Images to be sent to openAI
    Returns:
       A list of input parts compatible with the openAI response AI

    """
    if not images:
        return text.strip() if text else ""

    content=[]
    if text and text.strip():
        content.append({"type": "input_text", "text": text.strip()})

    for img in images:
        content.append({"type": "input_image", "image_url": img["data_url"]})

    return [{"role":"user","content":content}] if content else []

#Function to generate response from OpenAI response
def call_responses_api(parts: List[Dict[str, Any]],previous_response_id: str = None) ->Any:
    """
    Call the OpenAI responses API with input parts.
    
    Args:
        parts: The input parts to send to the API
        previous_response_id:The previous response ID to be sent to the OpenAI
   
    """
    tools = [{"type": "file_search",
                "vector_store_ids": [vector_store_id],
                "max_num_results": 20}]

    response=client.responses.create(
        model="gpt-5-nano",
        input=parts,
        instructions=system_prompt,
        tools=tools,
        previous_response_id=previous_response_id
    )
    return response

#Function to get the text output
def get_text_output(response:Any) -> str:
    """
    Get the text output from the OpenAI response API.

    """
    return response.output_text if hasattr(response, 'output_text') else str(response)

#Render all previous messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        #Extract text content from message structure 
        if isinstance(m["content"], list):
            # handle structred message (user input)
            for part in m["content"]:
                for content_item in part.get("content", []):
                    if content_item.get("type") == "input_text":
                        st.markdown(content_item["text"])
                    elif content_item.get("type") == "input_image":
                        st.image(content_item["image_url"],width=100)
        else:
            #handle simple text messages(assistant responses)
            st.markdown(m["content"])


#user interface -upload images
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"],accept_multiple_files=True)


#user interface chat input
prompt = st.chat_input("Ask me anything about BITTE...")
if prompt is not None:
    #process images into api compatible format
    images = [
        {
            "mime_type": f"image/{f.type.split('/')[-1]}" if f.type else "image/png",
            "data_url": f"data:{f.type};base64,{base64.b64encode(f.getvalue()).decode('utf-8')}"
        }
        for f in (uploaded or [])
    ]

    #create the input part for responses API
    parts = build_inputs_parts(prompt, images)

    #store the message
    st.session_state.messages.append({"role": "user", "content": parts})

    #Display user message
    with st.chat_message("user"):
        if isinstance(parts, str):
            st.markdown(parts)
        else:
            for p in parts:
                for content_item in p.get("content", []):
                    if content_item["type"] == "input_text":
                        st.markdown(content_item["text"])
                    elif content_item["type"] == "input_image":
                        st.image(content_item["image_url"]['url'],width=100)
                    else:
                        st.error(f"Unkown content type: {content_item['type']}")


  
    #Generate the API response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response=call_responses_api(parts,st.session_state.previous_response_id)
                output_text = get_text_output(response)
              
                #Display the AI response
                st.markdown(output_text)
                st.session_state.messages.append({"role": "assistant", "content": output_text})

                #Retrive the ID if available 
                if hasattr(response, 'id'):
                    st.session_state.previous_response_id = response.id

            except Exception as e:
                st.error(f"Error retrieving response ID: {e}")

        