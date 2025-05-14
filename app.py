import streamlit as st
import logging
import traceback
import os
import json
from typing import Tuple, Optional, Dict, List

# Import AutoGen components directly in app.py
import autogen
from autogen import UserProxyAgent, GroupChatManager, GroupChat, Agent

# Import necessary components from local modules
from LLMConfiguration import LLMConfiguration, VERTEX_AI, AZURE, ANTHROPIC
from common_functions import (
    create_agent,
    create_groupchat,
    create_groupchat_manager,
    initiate_chat_task,
    run_agent_step,
    send_user_message,
    read_system_message, # Still used for User.md and potentially display names
    load_personas # New import
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
USER_NAME = "User" # Internal code name for the user agent
AI_ASSISTANT_NAME = "AI_Assistant" # Generic name for the AI agent driven by personas

USER_SYS_MSG_FILE = "User.md"
PERSONA_DIR = "." # Directory for persona markdown files

CONTENT_INJECTION_MARKER = "## Content" 
MAX_MESSAGES_DISPLAY = 50
CONTENT_TEXT_KEY = "content_text_input" 
TASK_PROMPT_KEY = "initial_prompt_input"
USER_EDIT_KEY = "user_editable_prompt" # This key will now store the content of User.md
SELECTED_PERSONAS_KEY = "selected_personas_key"

AGENT_DISPLAY_NAMES_KEY = "agent_display_names"

# AGENT_CONFIG now primarily signals that User.md is the source for USER_NAME's prompt
AGENT_CONFIG = {
    USER_NAME: {"file": USER_SYS_MSG_FILE, "key": USER_EDIT_KEY},
}

CONTEXT_LIMIT = 1_000_000
WARNING_THRESHOLD = 0.85

# --- Helper Functions ---

def list_persona_files(persona_dir: str) -> List[Tuple[str, str]]:
    """Lists all .md files in the persona directory and their display names.
    Returns a list of tuples (filename, display_name).
    """
    persona_options: List[Tuple[str, str]] = []
    try:
        candidate_files = [f for f in os.listdir(persona_dir) if f.endswith(".md") and f != USER_SYS_MSG_FILE]
        candidate_files = sorted(candidate_files)

        for f_name in candidate_files:
            file_path_for_read = os.path.join(persona_dir, f_name)
            try:
                # read_system_message is imported from common_functions
                # It should return (display_name_from_file, content)
                display_name_from_file, _ = read_system_message(file_path_for_read)

                if display_name_from_file and display_name_from_file.strip():
                    display_name = display_name_from_file.strip()
                else:
                    # Fallback to filename without .md extension
                    display_name = os.path.splitext(f_name)[0]
                persona_options.append((f_name, display_name))
            except Exception as e:
                logger.warning(f"Could not extract display name from '{file_path_for_read}': {e}. Using filename as display name.")
                display_name = os.path.splitext(f_name)[0]
                persona_options.append((f_name, display_name))
        
        return persona_options
    except FileNotFoundError:
        logger.error(f"Persona directory '{persona_dir}' not found.")
        return []
    except Exception as e:
        logger.error(f"Error listing or processing persona files in '{persona_dir}': {e}")
        return []

def initialize_prompts_and_personas(): # Renamed for clarity as user prompt is no longer "editable" via UI
    if AGENT_DISPLAY_NAMES_KEY not in st.session_state:
        st.session_state[AGENT_DISPLAY_NAMES_KEY] = {}

    # Initialize User Agent prompt from User.md
    user_config = AGENT_CONFIG[USER_NAME]
    if user_config["key"] not in st.session_state: # USER_EDIT_KEY
        try:
            display_name, system_content = read_system_message(user_config["file"])
            st.session_state[user_config["key"]] = system_content # Load content into USER_EDIT_KEY
            actual_display_name = display_name or USER_NAME
            st.session_state[AGENT_DISPLAY_NAMES_KEY][USER_NAME] = actual_display_name
            logger.info(f"Loaded system message for '{USER_NAME}' (Display: '{actual_display_name}') from '{user_config['file']}' into state ('{user_config['key']}').")
        except Exception as e:
            st.session_state[user_config["key"]] = "" # Set to empty if error
            st.session_state[AGENT_DISPLAY_NAMES_KEY][USER_NAME] = USER_NAME # Fallback
            logger.error(f"Failed to load system message for {USER_NAME} from {user_config['file']}: {e}")
            st.sidebar.warning(f"Could not load system message for User agent from {user_config['file']}. Please ensure it exists and is readable.")


    # Initialize selected personas (empty list by default)
    if SELECTED_PERSONAS_KEY not in st.session_state:
        st.session_state[SELECTED_PERSONAS_KEY] = []
    
def update_token_warning():
    content_text = st.session_state.get(CONTENT_TEXT_KEY, "")
    task_text = st.session_state.get(TASK_PROMPT_KEY, "")
    user_prompt = st.session_state.get(USER_EDIT_KEY, "") # This is now from User.md
    
    selected_persona_files = st.session_state.get(SELECTED_PERSONAS_KEY, [])
    persona_content = ""
    if selected_persona_files:
        try:
            persona_content = load_personas(PERSONA_DIR, selected_persona_files)
        except Exception as e:
            logger.warning(f"Could not estimate persona tokens for warning: {e}")

    persona_tokens = estimate_tokens(persona_content)
    user_tokens = estimate_tokens(user_prompt)
    content_tokens = estimate_tokens(content_text)
    task_tokens = estimate_tokens(task_text)

    total_system_prompt_tokens = persona_tokens + user_tokens
    current_sidebar_input_tokens = content_tokens + task_tokens
    total_estimated_for_new_chat_tokens = current_sidebar_input_tokens + total_system_prompt_tokens

    cumulative_input = st.session_state.get("total_input_tokens", 0)
    cumulative_output = st.session_state.get("total_output_tokens", 0)
    total_cumulative_tokens = cumulative_input + cumulative_output

    token_info_str = f"Cumulative Tokens (In/Out): ~{cumulative_input:,} / ~{cumulative_output:,} (Total: ~{total_cumulative_tokens:,})"
    if not st.session_state.get("chat_initialized", False):
        token_info_str += f"\nEstimated for New Chat (System + Input): ~{total_estimated_for_new_chat_tokens:,} / {CONTEXT_LIMIT:,}"
    else:
        token_info_str += f"\nContext Limit: {CONTEXT_LIMIT:,}"

    if hasattr(st.session_state, 'token_info_placeholder') and st.session_state.token_info_placeholder:
        st.session_state.token_info_placeholder.caption(token_info_str)

    if hasattr(st.session_state, 'token_warning_placeholder') and st.session_state.token_warning_placeholder:
        if not st.session_state.get("chat_initialized", False) and total_estimated_for_new_chat_tokens > CONTEXT_LIMIT * WARNING_THRESHOLD:
            st.session_state.token_warning_placeholder.warning(f"Inputs for a new chat are approaching context limit ({WARNING_THRESHOLD*100:.0f}%). Est. new chat: ~{total_estimated_for_new_chat_tokens:,}")
        else:
            st.session_state.token_warning_placeholder.empty()

def setup_chat(
    llm_provider: str,
    model_name: str,
    content_text: Optional[str],
    user_agent_prompt: str, # This will be the content from User.md via USER_EDIT_KEY
    selected_persona_files: List[str],
    agent_display_names: Dict[str, str]
    ) -> Tuple[GroupChatManager, UserProxyAgent]:

    logger.info(f"Setting up chat with provider: {llm_provider}, model: {model_name}")
    logger.info(f"Selected personas for AI Assistant: {selected_persona_files}")

    if llm_provider == VERTEX_AI:
        try:
            if "gcp_credentials" not in st.secrets: raise ValueError("Missing 'gcp_credentials' in Streamlit secrets.")
            required_keys = ["project_id", "private_key", "client_email", "type"]
            if not all(key in st.secrets["gcp_credentials"] for key in required_keys): raise ValueError(f"Missing required keys in 'gcp_credentials'.")
            vertex_credentials = dict(st.secrets["gcp_credentials"])
            llm_config = LLMConfiguration(
                VERTEX_AI, model_name, project_id=vertex_credentials.get('project_id'),
                location="us-central1", vertex_credentials=vertex_credentials)
            logger.info("Vertex AI LLM configuration loaded.")
        except Exception as e: logger.error(f"Vertex AI credential error: {e}", exc_info=True); raise
    else: raise ValueError(f"Unsupported LLM provider: {llm_provider}")

    if not llm_config.get_config(): raise ValueError("Invalid LLM configuration.")

    agents_list = []
    
    if not user_agent_prompt: # Check if User.md content is empty
        raise ValueError(f"User agent system message is empty. Please ensure '{USER_SYS_MSG_FILE}' contains content.")
        
    user_agent = create_agent(
        name=USER_NAME,
        llm_config=llm_config,
        system_message_content=user_agent_prompt,
        agent_type="user_proxy"
    )
    agents_list.append(user_agent)
    if USER_NAME not in agent_display_names or not agent_display_names[USER_NAME]:
        agent_display_names[USER_NAME] = USER_NAME 
        logger.warning(f"User display name was missing, defaulted to {USER_NAME}")

    ai_assistant_system_message = "You are a helpful AI assistant."
    if selected_persona_files:
        loaded_persona_content = load_personas(PERSONA_DIR, selected_persona_files)
        if loaded_persona_content:
            ai_assistant_system_message = loaded_persona_content
        else:
            logger.warning(f"Loading personas {selected_persona_files} resulted in empty content. Using default system message for {AI_ASSISTANT_NAME}.")
    else:
        logger.info(f"No personas selected. Using default system message for {AI_ASSISTANT_NAME}.")

    ai_assistant = create_agent(
        name=AI_ASSISTANT_NAME,
        llm_config=llm_config,
        system_message_content=ai_assistant_system_message,
        agent_type="assistant"
    )
    agents_list.append(ai_assistant)
    
    # Update how AI Assistant display name is constructed
    # Uses the display names from the persona_display_names_map if available
    if selected_persona_files:
        # Get the display names from the map we created for multiselect
        global persona_display_names_map # Make sure map is accessible or passed
        
        display_name_parts = []
        for f_name in selected_persona_files:
            # Attempt to get display name from the map first
            # The map should have been populated when list_persona_files was called
            # We need to ensure this map is available here or re-fetch/re-create it based on selected_persona_files
            # For simplicity, let's re-fetch display names here or pass the map if possible
            # For now, we'll rely on a global or ensure it's passed if this were a class.
            # A cleaner way would be to pass the map to setup_chat or retrieve display names again.
            
            # Let's assume we can fetch the display name again if needed, or better yet, pass the map.
            # For this modification, we'll reconstruct the display names from the selected files.
            # This part ALREADY uses os.path.splitext(f)[0] as a fallback if the map isn't used.
            # The existing logic for display_name_parts based on filenames is a fallback.
            # If we want the *actual* display names (from DisplayName: field) used in the AI Assistant name:
            
            # Re-evaluate how to get display names for the AI_ASSISTANT.
            # The `setup_chat` function is called with `selected_persona_files` (filenames).
            # To get the *true* display names, we might need to re-read them or pass a map.
            
            # Let's adjust to use the persona_display_names_map if available in st.session_state
            # This map is created in the UI section. We can store it in session_state.
            
            if 'persona_display_names_map_for_setup' in st.session_state:
                 display_name_parts.append(st.session_state['persona_display_names_map_for_setup'].get(f_name, os.path.splitext(f_name)[0]))
            else: # Fallback if map not in session state for some reason
                 display_name_parts.append(os.path.splitext(f_name)[0])

        assistant_display_name = " & ".join(display_name_parts)
        if not assistant_display_name : assistant_display_name = AI_ASSISTANT_NAME
    else:
        assistant_display_name = AI_ASSISTANT_NAME
    
    agent_display_names[AI_ASSISTANT_NAME] = assistant_display_name
    st.session_state[AGENT_DISPLAY_NAMES_KEY][AI_ASSISTANT_NAME] = assistant_display_name
    logger.info(f"AI Assistant created with name '{AI_ASSISTANT_NAME}' and display name '{assistant_display_name}'.")

    try:
        groupchat = create_groupchat(agents_list, agent_display_names, max_round=50)
        manager = create_groupchat_manager(groupchat, llm_config)
        logger.info("GroupChat and Manager created.")
    except Exception as e: logger.error(f"GroupChat/Manager creation error: {e}", exc_info=True); raise

    return manager, user_agent

def estimate_tokens(text: str) -> int:
    return len(text or "") // 4

def display_messages(messages):
    num_messages = len(messages)
    start_index = max(0, num_messages - MAX_MESSAGES_DISPLAY)
    if num_messages > MAX_MESSAGES_DISPLAY: st.warning(f"Displaying last {MAX_MESSAGES_DISPLAY} of {num_messages} messages.")

    initial_task_prompt = st.session_state.get(TASK_PROMPT_KEY, "").strip()

    for msg in messages[start_index:]:
        internal_sender_name = msg.get("name", "System")
        sender_display_name = st.session_state.get(AGENT_DISPLAY_NAMES_KEY, {}).get(internal_sender_name, internal_sender_name)
        
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = [item["text"] if isinstance(item, dict) and "text" in item else str(item) for item in content]
            content = "\n".join(parts)
        elif not isinstance(content, str): content = str(content)

        if internal_sender_name == USER_NAME and content.strip() == initial_task_prompt:
            if messages.index(msg) > 0 :
                 continue

        avatar_map = {USER_NAME: "🧑", AI_ASSISTANT_NAME: "🤖"}
        avatar = avatar_map.get(internal_sender_name, "⚙️")

        with st.chat_message("user" if internal_sender_name == USER_NAME else "assistant", avatar=avatar):
            st.markdown(f"**{sender_display_name}:**\n{content}")

# --- Streamlit App UI ---
st.title("AI table discussion")

# --- Initialization ---
default_values = {
    "chat_initialized": False, "processing": False, "error_message": None,
    "config": None, "manager": None, "user_agent": None,
    "messages": [], "next_agent": None,
    TASK_PROMPT_KEY: "", CONTENT_TEXT_KEY: "",
    "total_input_tokens": 0, "total_output_tokens": 0,
    SELECTED_PERSONAS_KEY: [],
    "persona_display_names_map_for_setup": {} # For storing map for setup_chat
}
for key, value in default_values.items():
    if key not in st.session_state: st.session_state[key] = value

initialize_prompts_and_personas() # Load User.md here

if not st.session_state.config:
    try:
        st.session_state.config = {"llm_provider": VERTEX_AI, "model_name": "gemini-1.5-pro-002"}
    except Exception as e: st.sidebar.error(f"Config load failed: {e}"); st.stop()

# --- Sidebar UI ---
with st.sidebar.expander("Configure AI Agents & Task", expanded=True):
    st.caption("Select AI personas, define User agent (via User.md), and set the initial task.")

    available_persona_options: List[Tuple[str, str]] = list_persona_files(PERSONA_DIR)
    
    if not available_persona_options:
        st.warning(f"No persona files (.md) found in '{PERSONA_DIR}'. The AI assistant will use a default prompt.")
        persona_filenames_for_options = []
        # Ensure the map in session_state is empty if no options
        st.session_state['persona_display_names_map_for_setup'] = {}
    else:
        persona_filenames_for_options = [option[0] for option in available_persona_options]
        # Store the map in session_state so setup_chat can use it
        st.session_state['persona_display_names_map_for_setup'] = {filename: display_name for filename, display_name in available_persona_options}

    st.session_state[SELECTED_PERSONAS_KEY] = st.multiselect(
        "Select AI Personas (1 or 2):",
        options=persona_filenames_for_options, 
        default=st.session_state.get(SELECTED_PERSONAS_KEY, []), 
        format_func=lambda filename: st.session_state['persona_display_names_map_for_setup'].get(filename, filename),
        max_selections=2,
        key="persona_multiselect",
        help="Choose up to two personas to define the AI assistant's behavior. Content from these files will be combined.",
        on_change=update_token_warning,
        disabled=st.session_state.chat_initialized
    )

    # User Agent Prompt is now implicitly from User.md
    user_display_name_for_label = st.session_state.get(AGENT_DISPLAY_NAMES_KEY, {}).get(USER_NAME, USER_NAME)
    st.markdown(f"**{user_display_name_for_label} Prompt:** Loaded from `{USER_SYS_MSG_FILE}`.")
    
    st.session_state.token_info_placeholder = st.sidebar.empty()
    st.session_state.token_warning_placeholder = st.sidebar.empty()

    content_text_input = st.sidebar.text_area(
        "Enter General Context (Optional):", height=100, key=CONTENT_TEXT_KEY,
        disabled=st.session_state.chat_initialized, on_change=update_token_warning,
        help="This text can be used as general context for the discussion, separate from persona definitions."
    )
    initial_prompt_input = st.sidebar.text_area(
        "Enter the Initial Task or Question:", height=150, key=TASK_PROMPT_KEY,
        disabled=st.session_state.chat_initialized, on_change=update_token_warning)
    
    update_token_warning()

# --- Chat Control Buttons ---
if st.sidebar.button("🚀 Start Chat", key="start_chat",
                    disabled=st.session_state.chat_initialized or \
                             not st.session_state.get(TASK_PROMPT_KEY, "").strip() or \
                             not st.session_state.get(SELECTED_PERSONAS_KEY) or \
                             len(st.session_state.get(SELECTED_PERSONAS_KEY, [])) == 0 or \
                             not st.session_state.config):
    
    task_prompt = st.session_state.get(TASK_PROMPT_KEY, "").strip()
    content_text = st.session_state.get(CONTENT_TEXT_KEY, "").strip()
    selected_personas = st.session_state.get(SELECTED_PERSONAS_KEY, [])
    user_agent_prompt_content = st.session_state.get(USER_EDIT_KEY, "") # Loaded from User.md

    if not selected_personas :
        st.sidebar.warning("Please select at least one AI Persona.")
    elif not task_prompt:
         st.sidebar.warning("Please enter an initial task or question.")
    elif not user_agent_prompt_content: # Check if User.md content is missing
        st.sidebar.warning(f"User agent system message is missing. Please ensure '{USER_SYS_MSG_FILE}' is not empty and readable.")
    elif not st.session_state.chat_initialized and st.session_state.config: 
        st.session_state.processing = True; st.session_state.error_message = None
        st.session_state.total_input_tokens = 0
        st.session_state.total_output_tokens = 0
        
        if AGENT_DISPLAY_NAMES_KEY not in st.session_state:
            st.session_state[AGENT_DISPLAY_NAMES_KEY] = {}
        current_user_display_name = st.session_state[AGENT_DISPLAY_NAMES_KEY].get(USER_NAME, USER_NAME)
        current_display_names_for_setup = {USER_NAME: current_user_display_name}

        try:
            with st.spinner("Brewing conversations..."):
                st.session_state.manager, st.session_state.user_agent = setup_chat(
                    llm_provider=st.session_state.config.get("llm_provider", VERTEX_AI),
                    model_name=st.session_state.config.get("model_name", "gemini-1.5-pro-002"),
                    content_text=content_text,
                    user_agent_prompt=user_agent_prompt_content,
                    selected_persona_files=selected_personas,
                    agent_display_names=current_display_names_for_setup 
                )
                initial_messages, next_agent = initiate_chat_task(
                    st.session_state.user_agent,
                    st.session_state.manager,
                    task_prompt,
                    system_content_for_group=content_text
                )
                st.session_state.messages = initial_messages
                st.session_state.next_agent = next_agent
                st.session_state.chat_initialized = True

                st.session_state.total_input_tokens += estimate_tokens(task_prompt)
                if content_text:
                    st.session_state.total_input_tokens += estimate_tokens(content_text)
                
                st.session_state.total_input_tokens += estimate_tokens(user_agent_prompt_content)
                ai_persona_content = load_personas(PERSONA_DIR, selected_personas)
                st.session_state.total_input_tokens += estimate_tokens(ai_persona_content)

                for msg in initial_messages:
                    msg_content = msg.get("content", "")
                    if isinstance(msg_content, list):
                         msg_content = "\n".join(p.get("text", "") for p in msg_content if isinstance(p, dict) and "text" in p)
                    if msg.get("name") != USER_NAME:
                        st.session_state.total_output_tokens += estimate_tokens(msg_content)
        except Exception as e:
            st.session_state.error_message = f"Setup/Initiation failed: {e}"; st.session_state.chat_initialized = False
            logger.error(f"Chat start error: {traceback.format_exc()}")
        finally: st.session_state.processing = False
        update_token_warning()
        st.rerun()

if st.session_state.error_message: st.error(st.session_state.error_message)

chat_container = st.container()
with chat_container:
    if st.session_state.chat_initialized and st.session_state.manager:
        display_messages(st.session_state.messages)
        next_agent_code_name = st.session_state.next_agent.name if st.session_state.next_agent else None
        
        display_next_agent_name = st.session_state.get(AGENT_DISPLAY_NAMES_KEY, {}).get(next_agent_code_name, next_agent_code_name)

        if next_agent_code_name and not st.session_state.processing:
            if next_agent_code_name == USER_NAME:
                with st.form(key=f'user_input_form_{len(st.session_state.messages)}'):
                    col1, col2 = st.columns([0.9, 0.1])
                    with col1:
                        user_input = st.text_area("Enter your message:", height=80, key=f"user_input_{len(st.session_state.messages)}", label_visibility="collapsed")
                    with col2:
                        send_button_pressed = st.form_submit_button("➤", help="Send message")

                    if send_button_pressed:
                        if user_input.strip():
                            st.session_state.processing = True; st.session_state.error_message = None
                            st.session_state.total_input_tokens += estimate_tokens(user_input)
                            with st.spinner("Sending message..."):
                                try:
                                    new_msgs, next_ag = send_user_message(st.session_state.manager, st.session_state.user_agent, user_input)
                                    st.session_state.messages.extend(new_msgs); st.session_state.next_agent = next_ag
                                except Exception as e: st.session_state.error_message = f"Send error: {e}"; logger.error(f"Send error: {traceback.format_exc()}")
                            st.session_state.processing = False
                            update_token_warning(); st.rerun()
                        else: st.warning("Please enter a message.")
            else: 
                st.markdown(f"**Running turn for:** {display_next_agent_name}...")
                st.session_state.processing = True; st.session_state.error_message = None
                with st.spinner(f"Thinking... {display_next_agent_name} is responding..."):
                    try:
                        new_msgs, next_ag_after_ai = run_agent_step(st.session_state.manager, st.session_state.next_agent)
                        st.session_state.messages.extend(new_msgs)
                        for msg in new_msgs:
                            msg_content = msg.get("content", "")
                            if isinstance(msg_content, list): 
                                msg_content = "\n".join(p.get("text", "") for p in msg_content if isinstance(p, dict) and "text" in p)
                            if msg.get("name") != USER_NAME : 
                                st.session_state.total_output_tokens += estimate_tokens(msg_content)
                        
                        st.session_state.next_agent = st.session_state.user_agent 
                        
                    except Exception as e:
                        st.session_state.error_message = f"Error during {display_next_agent_name}'s turn: {e}"; st.session_state.next_agent = st.session_state.user_agent
                        logger.error(f"{display_next_agent_name} turn error: {traceback.format_exc()}")
                st.session_state.processing = False
                update_token_warning(); st.rerun()
        elif not next_agent_code_name and st.session_state.chat_initialized and not st.session_state.processing:
             st.success("Chat finished or awaiting next step.")

if st.session_state.chat_initialized or st.session_state.error_message:
     if st.sidebar.button("Clear Chat / Reset", key="clear_chat"):
         keys_to_clear = list(default_values.keys()) 
         keys_to_clear.extend([USER_EDIT_KEY, AGENT_DISPLAY_NAMES_KEY, "persona_multiselect"]) 
         
         for key in keys_to_clear:
             if key in st.session_state: 
                 if key == SELECTED_PERSONAS_KEY: st.session_state[key] = []
                 # Make sure to also reset the new map in session_state
                 elif key == "persona_display_names_map_for_setup": st.session_state[key] = {}
                 elif key == AGENT_DISPLAY_NAMES_KEY: st.session_state[key] = {}
                 elif key == "messages": st.session_state[key] = []
                 else: del st.session_state[key]

         for key, value in default_values.items():
            if key not in st.session_state: st.session_state[key] = value
         
         initialize_prompts_and_personas() 
         logger.info("Chat state cleared and prompts re-initialized.")
         st.session_state.total_input_tokens = 0
         st.session_state.total_output_tokens = 0
         update_token_warning(); st.rerun()
