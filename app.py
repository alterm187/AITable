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
PERSONA_DIR = "AITable" # Directory for persona markdown files

CONTENT_INJECTION_MARKER = "## Content" 
MAX_MESSAGES_DISPLAY = 50
CONTENT_TEXT_KEY = "content_text_input" 
TASK_PROMPT_KEY = "initial_prompt_input"
USER_EDIT_KEY = "user_editable_prompt"
SELECTED_PERSONAS_KEY = "selected_personas_key"

AGENT_DISPLAY_NAMES_KEY = "agent_display_names"

# Simplified AGENT_CONFIG, primarily for the User agent's editable prompt for now
AGENT_CONFIG = {
    USER_NAME: {"file": USER_SYS_MSG_FILE, "key": USER_EDIT_KEY},
}

CONTEXT_LIMIT = 1_000_000
WARNING_THRESHOLD = 0.85

# --- Helper Functions ---

def list_persona_files(persona_dir: str) -> List[str]:
    """Lists all .md files in the persona directory."""
    try:
        files = [f for f in os.listdir(persona_dir) if f.endswith(".md") and f != USER_SYS_MSG_FILE]
        # Sort for consistent order in UI
        return sorted(files)
    except FileNotFoundError:
        logger.error(f"Persona directory not found: {persona_dir}")
        return []
    except Exception as e:
        logger.error(f"Error listing persona files in {persona_dir}: {e}")
        return []

def initialize_editable_prompts_and_personas():
    if AGENT_DISPLAY_NAMES_KEY not in st.session_state:
        st.session_state[AGENT_DISPLAY_NAMES_KEY] = {}

    # Initialize User Agent prompt
    user_config = AGENT_CONFIG[USER_NAME]
    if user_config["key"] not in st.session_state:
        try:
            display_name, system_content = read_system_message(user_config["file"])
            st.session_state[user_config["key"]] = system_content
            actual_display_name = display_name or USER_NAME
            st.session_state[AGENT_DISPLAY_NAMES_KEY][USER_NAME] = actual_display_name
            logger.info(f"Loaded prompt for '{USER_NAME}' (Display: '{actual_display_name}') into state ('{user_config['key']}').")
        except Exception as e:
            st.session_state[user_config["key"]] = f"Error loading from {user_config['file']}: {e}"
            st.session_state[AGENT_DISPLAY_NAMES_KEY][USER_NAME] = USER_NAME # Fallback
            logger.error(f"Failed to load prompt for {USER_NAME} from {user_config['file']}: {e}")

    # Initialize selected personas (empty list by default)
    if SELECTED_PERSONAS_KEY not in st.session_state:
        st.session_state[SELECTED_PERSONAS_KEY] = []
    
    # Pre-populate display names for persona files if needed (e.g. if we want to show their DisplayName: from file)
    # For now, the multiselect will just show filenames. The combined persona will be for AI_ASSISTANT_NAME.

def update_token_warning():
    content_text = st.session_state.get(CONTENT_TEXT_KEY, "")
    task_text = st.session_state.get(TASK_PROMPT_KEY, "")
    user_prompt = st.session_state.get(USER_EDIT_KEY, "")
    
    # Estimate tokens for selected personas
    selected_persona_files = st.session_state.get(SELECTED_PERSONAS_KEY, [])
    persona_content = ""
    if selected_persona_files: # Check if list is not empty
        try:
            # Temporarily load personas to estimate tokens. This might be inefficient if called often.
            # Consider a lighter way if this becomes a bottleneck.
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
    user_agent_prompt: str,
    selected_persona_files: List[str], # New: list of selected persona filenames
    agent_display_names: Dict[str, str] # Contains display name for User, and will for AI_ASSISTANT_NAME
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
    
    # 1. Create User Agent
    if not user_agent_prompt: raise ValueError("User agent prompt cannot be empty.")
    user_agent = create_agent(
        name=USER_NAME,
        llm_config=llm_config, # User agent might not strictly need full LLM config if human_input_mode="ALWAYS"
        system_message_content=user_agent_prompt,
        agent_type="user_proxy"
    )
    agents_list.append(user_agent)
    # Ensure display name for User is in the session state map for display_messages
    if USER_NAME not in agent_display_names or not agent_display_names[USER_NAME]:
        agent_display_names[USER_NAME] = USER_NAME 
        logger.warning(f"User display name was missing, defaulted to {USER_NAME}")


    # 2. Create AI Assistant Agent (single agent combining selected personas)
    ai_assistant_system_message = "You are a helpful AI assistant." # Default
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
    
    # Set up display name for the AI Assistant
    # Could be a combination of persona names, or a generic one if multiple are too long.
    if selected_persona_files:
        # Simple approach: Join first part of filenames (before .md)
        display_name_parts = [os.path.splitext(f)[0] for f in selected_persona_files]
        assistant_display_name = " & ".join(display_name_parts)
        if not assistant_display_name : assistant_display_name = AI_ASSISTANT_NAME # Fallback
    else:
        assistant_display_name = AI_ASSISTANT_NAME # Default if no personas
    
    agent_display_names[AI_ASSISTANT_NAME] = assistant_display_name
    st.session_state[AGENT_DISPLAY_NAMES_KEY][AI_ASSISTANT_NAME] = assistant_display_name # Update session state for display_messages
    logger.info(f"AI Assistant created with name '{AI_ASSISTANT_NAME}' and display name '{assistant_display_name}'.")


    try:
        # The agent_display_names dict passed to create_groupchat MUST contain entries for ALL agent names in agents_list
        # We've ensured User and AI_Assistant display names are now set.
        groupchat = create_groupchat(agents_list, agent_display_names, max_round=50)
        manager = create_groupchat_manager(groupchat, llm_config)
        logger.info("GroupChat and Manager created.")
    except Exception as e: logger.error(f"GroupChat/Manager creation error: {e}", exc_info=True); raise

    return manager, user_agent # Return the main user_proxy agent

def estimate_tokens(text: str) -> int:
    return len(text or "") // 4 # Simple approximation

def display_messages(messages):
    num_messages = len(messages)
    start_index = max(0, num_messages - MAX_MESSAGES_DISPLAY)
    if num_messages > MAX_MESSAGES_DISPLAY: st.warning(f"Displaying last {MAX_MESSAGES_DISPLAY} of {num_messages} messages.")

    initial_task_prompt = st.session_state.get(TASK_PROMPT_KEY, "").strip()
    # content_text = st.session_state.get(CONTENT_TEXT_KEY, "").strip() # Not strictly needed for filtering display

    for msg in messages[start_index:]:
        internal_sender_name = msg.get("name", "System")
        # Use AGENT_DISPLAY_NAMES_KEY from session_state as it's updated in setup_chat
        sender_display_name = st.session_state.get(AGENT_DISPLAY_NAMES_KEY, {}).get(internal_sender_name, internal_sender_name)
        
        content = msg.get("content", "")
        if isinstance(content, list): # Handle Gemini's list format for content
            parts = [item["text"] if isinstance(item, dict) and "text" in item else str(item) for item in content]
            content = "\n".join(parts)
        elif not isinstance(content, str): content = str(content)

        # Avoid re-displaying the exact initial task prompt if it appears as a message content (e.g. echoed by user_proxy)
        if internal_sender_name == USER_NAME and content.strip() == initial_task_prompt:
            # Check if this is the very first message. If so, it's the task, display it.
            # If not the first, and matches, might be an echo - skip.
            # This simple check might need refinement.
            if messages.index(msg) > 0 : # messages is the full list here from st.session_state.messages
                 continue


        avatar_map = {USER_NAME: "🧑", AI_ASSISTANT_NAME: "🤖"} # Updated for new AI agent name
        # Add more avatars if you have more personas/fixed agent names or use a default
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
    SELECTED_PERSONAS_KEY: [], # Initialize selected personas
}
for key, value in default_values.items():
    if key not in st.session_state: st.session_state[key] = value

initialize_editable_prompts_and_personas() # Changed name for clarity

if not st.session_state.config:
    try:
        st.session_state.config = {"llm_provider": VERTEX_AI, "model_name": "gemini-1.5-pro-002"}
    except Exception as e: st.sidebar.error(f"Config load failed: {e}"); st.stop()

# --- Sidebar UI ---
with st.sidebar.expander("Configure AI Agents & Task", expanded=True):
    st.caption("Select AI personas, define User agent, and set the initial task.")

    # Persona Selection
    available_persona_files = list_persona_files(PERSONA_DIR)
    if not available_persona_files:
        st.warning(f"No persona files (.md) found in '{PERSONA_DIR}'. The AI assistant will use a default prompt.")
    
    st.session_state[SELECTED_PERSONAS_KEY] = st.multiselect(
        "Select AI Personas (1 or 2):",
        options=available_persona_files,
        default=st.session_state.get(SELECTED_PERSONAS_KEY, []), # Persist selection
        max_selections=2,
        key="persona_multiselect", # Explicit key for multiselect
        help="Choose up to two personas to define the AI assistant's behavior. Content from these files will be combined.",
        on_change=update_token_warning,
        disabled=st.session_state.chat_initialized
    )

    # User Agent Prompt
    user_config = AGENT_CONFIG[USER_NAME]
    user_display_name_for_label = st.session_state[AGENT_DISPLAY_NAMES_KEY].get(USER_NAME, USER_NAME)
    st.text_area(
        f"System Prompt for {user_display_name_for_label} (Editable):",
        key=user_config["key"],
        height=100, # Reduced height
        disabled=st.session_state.chat_initialized,
        help=f"Define background for {user_display_name_for_label}. The 'DisplayName' can be set in '{USER_SYS_MSG_FILE}'.",
        on_change=update_token_warning
    )
    
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
    
    update_token_warning() # Initial call

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
    user_agent_prompt_content = st.session_state.get(USER_EDIT_KEY, "")

    if not selected_personas :
        st.sidebar.warning("Please select at least one AI Persona.")
    elif not task_prompt:
         st.sidebar.warning("Please enter an initial task or question.")
    elif not st.session_state.chat_initialized and user_agent_prompt_content and st.session_state.config:
        st.session_state.processing = True; st.session_state.error_message = None
        st.session_state.total_input_tokens = 0
        st.session_state.total_output_tokens = 0
        
        # Ensure AGENT_DISPLAY_NAMES_KEY is initialized for setup_chat
        if AGENT_DISPLAY_NAMES_KEY not in st.session_state:
            st.session_state[AGENT_DISPLAY_NAMES_KEY] = {}
        # Get current display name for User, it should have been set by initialize_editable_prompts_and_personas
        current_user_display_name = st.session_state[AGENT_DISPLAY_NAMES_KEY].get(USER_NAME, USER_NAME)
        
        # The display names dict will be populated further in setup_chat for the AI_ASSISTANT_NAME
        current_display_names_for_setup = {USER_NAME: current_user_display_name}


        try:
            with st.spinner("Brewing conversations..."):
                st.session_state.manager, st.session_state.user_agent = setup_chat(
                    llm_provider=st.session_state.config.get("llm_provider", VERTEX_AI),
                    model_name=st.session_state.config.get("model_name", "gemini-1.5-pro-002"),
                    content_text=content_text, # This is the group context
                    user_agent_prompt=user_agent_prompt_content,
                    selected_persona_files=selected_personas,
                    agent_display_names=current_display_names_for_setup # Pass the dict to be populated
                )
                initial_messages, next_agent = initiate_chat_task(
                    st.session_state.user_agent,
                    st.session_state.manager,
                    task_prompt,
                    system_content_for_group=content_text # Pass content_text as system_content_for_group
                )
                st.session_state.messages = initial_messages
                st.session_state.next_agent = next_agent
                st.session_state.chat_initialized = True

                st.session_state.total_input_tokens += estimate_tokens(task_prompt)
                if content_text:
                    st.session_state.total_input_tokens += estimate_tokens(content_text)
                
                # Estimate system prompt tokens (user + loaded personas)
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

# --- Main Chat Area ---
chat_container = st.container()
with chat_container:
    if st.session_state.chat_initialized and st.session_state.manager:
        display_messages(st.session_state.messages)
        next_agent_code_name = st.session_state.next_agent.name if st.session_state.next_agent else None
        
        # Use AGENT_DISPLAY_NAMES_KEY from session state as it's the most up-to-date
        display_next_agent_name = st.session_state.get(AGENT_DISPLAY_NAMES_KEY, {}).get(next_agent_code_name, next_agent_code_name)


        if next_agent_code_name and not st.session_state.processing:
            if next_agent_code_name == USER_NAME: # User's turn
                with st.form(key=f'user_input_form_{len(st.session_state.messages)}'): # Unique key for form
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
            else: # AI Persona's Turn (AI_ASSISTANT_NAME)
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
                        
                        # Default to User after AI turn. This could also be determined by select_speaker if AI's response implies another AI should speak.
                        # For simplicity, after AI assistant speaks, it's User's turn.
                        st.session_state.next_agent = st.session_state.user_agent 
                        # st.session_state.next_agent = next_ag_after_ai # Use if custom_speaker_selection can pick AI again
                        
                    except Exception as e:
                        st.session_state.error_message = f"Error during {display_next_agent_name}'s turn: {e}"; st.session_state.next_agent = st.session_state.user_agent # Fallback to user
                        logger.error(f"{display_next_agent_name} turn error: {traceback.format_exc()}")
                st.session_state.processing = False
                update_token_warning(); st.rerun()
        elif not next_agent_code_name and st.session_state.chat_initialized and not st.session_state.processing:
             st.success("Chat finished or awaiting next step.") # Or "Chat Ended."

if st.session_state.chat_initialized or st.session_state.error_message: # Show clear button if chat started or error
     if st.sidebar.button("Clear Chat / Reset", key="clear_chat"):
         keys_to_clear = list(default_values.keys()) 
         # Add UI-specific keys that need reset if not in default_values
         keys_to_clear.extend([USER_EDIT_KEY, AGENT_DISPLAY_NAMES_KEY, "persona_multiselect"]) 
         
         for key in keys_to_clear:
             if key in st.session_state: 
                 # For lists/dicts, specifically clear them or reset to default empty if needed
                 if key == SELECTED_PERSONAS_KEY: st.session_state[key] = []
                 elif key == AGENT_DISPLAY_NAMES_KEY: st.session_state[key] = {}
                 elif key == "messages": st.session_state[key] = []
                 else: del st.session_state[key]

         # Re-initialize after clearing to set defaults
         for key, value in default_values.items(): # Ensure defaults are back
            if key not in st.session_state: st.session_state[key] = value
         
         initialize_editable_prompts_and_personas() # Re-init prompts and selected personas list
         logger.info("Chat state cleared and prompts re-initialized.")
         st.session_state.total_input_tokens = 0 # Explicitly reset
         st.session_state.total_output_tokens = 0
         update_token_warning(); st.rerun()

