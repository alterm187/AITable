import streamlit as st
import logging
import traceback
import os
import json
from typing import Tuple, Optional, Dict, List, Union # Added Union

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
    read_system_message, 
    load_personas
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
USER_NAME = "User" 
AI_ASSISTANT_NAME = "AI_Assistant"

USER_SYS_MSG_FILE = "User.md"
PERSONA_DIR = "." 

CONTENT_INJECTION_MARKER = "## Content" 
MAX_MESSAGES_DISPLAY = 50
CONTENT_TEXT_KEY = "content_text_input" 
TASK_PROMPT_KEY = "initial_prompt_input"
USER_EDIT_KEY = "user_editable_prompt" 
SELECTED_PERSONAS_KEY = "selected_personas_key"
AGENT_DISPLAY_NAMES_KEY = "agent_display_names_for_chat_display" # Renamed for clarity
PERSONA_DISPLAY_NAMES_MAP_KEY = "persona_display_names_map_for_setup"

AGENT_CONFIG = {
    USER_NAME: {"file": USER_SYS_MSG_FILE, "key": USER_EDIT_KEY},
}

CONTEXT_LIMIT = 1_000_000
WARNING_THRESHOLD = 0.85

# --- Helper Functions ---

def list_persona_files(persona_dir: str) -> List[Tuple[str, str]]:
    persona_options: List[Tuple[str, str]] = []
    try:
        # Only include files starting with "Persona" and ending with ".md"
        candidate_files = [f for f in os.listdir(persona_dir) if f.startswith("Persona") and f.endswith(".md") and f != USER_SYS_MSG_FILE]
        candidate_files = sorted(candidate_files)
        for f_name in candidate_files:
            file_path_for_read = os.path.join(persona_dir, f_name)
            try:
                display_name_from_file, _ = read_system_message(file_path_for_read)
                if display_name_from_file and display_name_from_file.strip():
                    display_name = display_name_from_file.strip()
                else:
                    display_name = os.path.splitext(f_name)[0]
                persona_options.append((f_name, display_name))
            except Exception as e:
                logger.warning(f"Could not extract display name from '{file_path_for_read}': {e}. Using filename.")
                display_name = os.path.splitext(f_name)[0]
                persona_options.append((f_name, display_name))
        return persona_options
    except FileNotFoundError:
        logger.error(f"Persona directory '{persona_dir}' not found.")
        return []
    except Exception as e:
        logger.error(f"Error listing persona files in '{persona_dir}': {e}")
        return []

def initialize_prompts_and_personas():
    if AGENT_DISPLAY_NAMES_KEY not in st.session_state:
        st.session_state[AGENT_DISPLAY_NAMES_KEY] = {}
    if PERSONA_DISPLAY_NAMES_MAP_KEY not in st.session_state:
        st.session_state[PERSONA_DISPLAY_NAMES_MAP_KEY] = {}

    user_config = AGENT_CONFIG[USER_NAME]
    if user_config["key"] not in st.session_state:
        try:
            display_name, system_content = read_system_message(user_config["file"])
            st.session_state[user_config["key"]] = system_content
            actual_display_name = display_name or USER_NAME
            # This map is for *displaying* messages in the chat UI
            st.session_state[AGENT_DISPLAY_NAMES_KEY][USER_NAME] = actual_display_name
            logger.info(f"Loaded system message for '{USER_NAME}' (Display: '{actual_display_name}') from '{user_config['file']}'.")
        except Exception as e:
            st.session_state[user_config["key"]] = ""
            st.session_state[AGENT_DISPLAY_NAMES_KEY][USER_NAME] = USER_NAME
            logger.error(f"Failed to load system message for {USER_NAME}: {e}")
            st.sidebar.warning(f"Could not load system message for User agent from {user_config['file']}.")

    if SELECTED_PERSONAS_KEY not in st.session_state:
        st.session_state[SELECTED_PERSONAS_KEY] = []

def update_token_warning():
    content_text = st.session_state.get(CONTENT_TEXT_KEY, "")
    task_text = st.session_state.get(TASK_PROMPT_KEY, "")
    user_prompt = st.session_state.get(USER_EDIT_KEY, "")
    selected_persona_files = st.session_state.get(SELECTED_PERSONAS_KEY, [])
    persona_content = load_personas(PERSONA_DIR, selected_persona_files) if selected_persona_files else ""
    
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
            st.session_state.token_warning_placeholder.warning(f"Inputs for new chat approaching limit. Est: ~{total_estimated_for_new_chat_tokens:,}")
        else:
            st.session_state.token_warning_placeholder.empty()

def setup_chat(
    llm_provider: str, model_name: str, content_text: Optional[str],
    user_agent_prompt: str, selected_persona_files: List[str],
    # This map is for groupchat internal logic (speaker selection)
    group_chat_agent_display_config: Dict[str, Union[str, List[str]]]
) -> Tuple[GroupChatManager, UserProxyAgent]:
    logger.info(f"Setting up chat. Selected personas: {selected_persona_files}")
    if llm_provider == VERTEX_AI:
        try:
            creds = st.secrets.get("gcp_credentials", {})
            llm_config = LLMConfiguration(VERTEX_AI, model_name, project_id=creds.get('project_id'), location="us-central1", vertex_credentials=creds)
        except Exception as e: logger.error(f"Vertex AI credential error: {e}", exc_info=True); raise
    else: raise ValueError(f"Unsupported LLM provider: {llm_provider}")
    if not llm_config.get_config(): raise ValueError("Invalid LLM configuration.")

    agents_list = []
    if not user_agent_prompt: raise ValueError(f"User agent prompt empty (from {USER_SYS_MSG_FILE}).")
    user_agent = create_agent(USER_NAME, llm_config, user_agent_prompt, "user_proxy")
    agents_list.append(user_agent)

    ai_assistant_system_message = load_personas(PERSONA_DIR, selected_persona_files) or "You are a helpful AI assistant."
    ai_assistant = create_agent(AI_ASSISTANT_NAME, llm_config, ai_assistant_system_message, "assistant")
    agents_list.append(ai_assistant)

    # Construct the AI Assistant's display name for chat UI (AGENT_DISPLAY_NAMES_KEY)
    # This is separate from group_chat_agent_display_config which is for speaker selection
    persona_display_names_for_ui = st.session_state.get(PERSONA_DISPLAY_NAMES_MAP_KEY, {})
    assistant_ui_name_parts = [persona_display_names_for_ui.get(f, os.path.splitext(f)[0]) for f in selected_persona_files]
    final_assistant_ui_name = " & ".join(assistant_ui_name_parts) if assistant_ui_name_parts else AI_ASSISTANT_NAME
    st.session_state[AGENT_DISPLAY_NAMES_KEY][AI_ASSISTANT_NAME] = final_assistant_ui_name
    logger.info(f"AI Assistant UI Display Name: '{final_assistant_ui_name}'")

    # Ensure User agent has an entry in AGENT_DISPLAY_NAMES_KEY if not already by initialize_prompts_and_personas
    if USER_NAME not in st.session_state[AGENT_DISPLAY_NAMES_KEY]:
        st.session_state[AGENT_DISPLAY_NAMES_KEY][USER_NAME] = group_chat_agent_display_config.get(USER_NAME, USER_NAME) # Fallback to code name

    try:
        # create_groupchat now uses group_chat_agent_display_config for its internal logic
        groupchat = create_groupchat(agents_list, group_chat_agent_display_config, max_round=50)
        manager = create_groupchat_manager(groupchat, llm_config)
    except Exception as e: logger.error(f"GroupChat/Manager creation error: {e}", exc_info=True); raise
    return manager, user_agent

def estimate_tokens(text: str) -> int: return len(text or "") // 4

def display_messages(messages):
    num_messages = len(messages)
    start_index = max(0, num_messages - MAX_MESSAGES_DISPLAY)
    if num_messages > MAX_MESSAGES_DISPLAY: st.warning(f"Displaying last {MAX_MESSAGES_DISPLAY} of {num_messages} messages.")
    initial_task_prompt = st.session_state.get(TASK_PROMPT_KEY, "").strip()

    for msg in messages[start_index:]:
        internal_sender_name = msg.get("name", "System")
        # Use AGENT_DISPLAY_NAMES_KEY for chat display
        sender_display_name = st.session_state.get(AGENT_DISPLAY_NAMES_KEY, {}).get(internal_sender_name, internal_sender_name)
        content = msg.get("content", "")
        if isinstance(content, list): content = "\n".join(item["text"] for item in content if isinstance(item, dict) and "text" in item)
        elif not isinstance(content, str): content = str(content)
        if internal_sender_name == USER_NAME and content.strip() == initial_task_prompt and messages.index(msg) > 0: continue
        avatar = {"User": "🧑", "AI_Assistant": "🤖"}.get(internal_sender_name, "⚙️")
        with st.chat_message("user" if internal_sender_name == USER_NAME else "assistant", avatar=avatar):
            if internal_sender_name == AI_ASSISTANT_NAME:
                st.markdown(content)
            else:
                st.markdown(f"**{sender_display_name}:**\\n{content}")

st.title("AI table discussion")

default_values = {
    "chat_initialized": False, "processing": False, "error_message": None,
    "config": None, "manager": None, "user_agent": None, "messages": [], "next_agent": None,
    TASK_PROMPT_KEY: "", CONTENT_TEXT_KEY: "", "total_input_tokens": 0, "total_output_tokens": 0,
    SELECTED_PERSONAS_KEY: [], PERSONA_DISPLAY_NAMES_MAP_KEY: {},
    AGENT_DISPLAY_NAMES_KEY: {}
}
for key, value in default_values.items():
    if key not in st.session_state: st.session_state[key] = value

initialize_prompts_and_personas()
if not st.session_state.config: st.session_state.config = {"llm_provider": VERTEX_AI, "model_name": "gemini-2.5-pro-preview-05-06"}

with st.sidebar.expander("Configure AI Agents & Task", expanded=True):
    st.caption("Select AI personas, User agent (User.md), and initial task.")
    available_persona_options = list_persona_files(PERSONA_DIR)
    if not available_persona_options:
        st.warning(f"No persona files (Persona*.md) found in '{PERSONA_DIR}'. AI uses default prompt.")
        persona_filenames_for_options = []
        st.session_state[PERSONA_DISPLAY_NAMES_MAP_KEY] = {}
    else:
        persona_filenames_for_options = [option[0] for option in available_persona_options]
        st.session_state[PERSONA_DISPLAY_NAMES_MAP_KEY] = {fn: dn for fn, dn in available_persona_options}

    st.session_state[SELECTED_PERSONAS_KEY] = st.multiselect(
        "Select AI Personas (1 or 2):",
        options=persona_filenames_for_options, 
        default=st.session_state.get(SELECTED_PERSONAS_KEY, []),
        format_func=lambda fn: st.session_state[PERSONA_DISPLAY_NAMES_MAP_KEY].get(fn, fn),
        max_selections=2, key="persona_multiselect",
        help="Choose 1 or 2 personas for the AI assistant.",
        on_change=update_token_warning, disabled=st.session_state.chat_initialized
    )
    user_ui_display_name = st.session_state[AGENT_DISPLAY_NAMES_KEY].get(USER_NAME, USER_NAME)
    #st.markdown(f"**{user_ui_display_name} Prompt:** Loaded from `{USER_SYS_MSG_FILE}`.")
    
    st.session_state.token_info_placeholder = st.sidebar.empty()
    st.session_state.token_warning_placeholder = st.sidebar.empty()
    content_text_input = st.sidebar.text_area("General Context (Optional):", height=100, key=CONTENT_TEXT_KEY, disabled=st.session_state.chat_initialized, on_change=update_token_warning)
    initial_prompt_input = st.sidebar.text_area("Initial Task or Question:", height=150, key=TASK_PROMPT_KEY, disabled=st.session_state.chat_initialized, on_change=update_token_warning)
    update_token_warning()

if st.sidebar.button("🚀 Start Chat", key="start_chat", disabled=st.session_state.chat_initialized or not st.session_state.get(TASK_PROMPT_KEY, "").strip() or not st.session_state.get(SELECTED_PERSONAS_KEY) or not st.session_state.config):
    task_prompt = st.session_state.get(TASK_PROMPT_KEY, "").strip()
    content_text = st.session_state.get(CONTENT_TEXT_KEY, "").strip()
    selected_persona_files = st.session_state.get(SELECTED_PERSONAS_KEY, [])
    user_agent_prompt_content = st.session_state.get(USER_EDIT_KEY, "")

    if not selected_persona_files: st.sidebar.warning("Please select at least one AI Persona.")
    elif not task_prompt: st.sidebar.warning("Please enter an initial task.")
    elif not user_agent_prompt_content: st.sidebar.warning(f"User agent prompt from {USER_SYS_MSG_FILE} is missing.")
    else:
        st.session_state.processing = True; st.session_state.error_message = None
        st.session_state.total_input_tokens = 0; st.session_state.total_output_tokens = 0

        # Prepare display name config for GroupChat (speaker selection)
        group_chat_agent_config = {}
        # User agent display name (can be from User.md or default)
        user_agent_display_name_for_groupchat = st.session_state[AGENT_DISPLAY_NAMES_KEY].get(USER_NAME, USER_NAME)
        group_chat_agent_config[USER_NAME] = user_agent_display_name_for_groupchat

        # AI Assistant display name(s)/aliases for GroupChat
        persona_map = st.session_state.get(PERSONA_DISPLAY_NAMES_MAP_KEY, {})
        individual_persona_display_names = [persona_map.get(f, os.path.splitext(f)[0]) for f in selected_persona_files]
        
        if len(individual_persona_display_names) > 1:
            combined_name = " & ".join(individual_persona_display_names)
            # Aliases include combined name + individual names
            group_chat_agent_config[AI_ASSISTANT_NAME] = [combined_name] + individual_persona_display_names
        elif len(individual_persona_display_names) == 1:
            group_chat_agent_config[AI_ASSISTANT_NAME] = individual_persona_display_names[0]
        else: # Should not happen due to check, but as a fallback
            group_chat_agent_config[AI_ASSISTANT_NAME] = AI_ASSISTANT_NAME 
        logger.info(f"GroupChat speaker selection config for {AI_ASSISTANT_NAME}: {group_chat_agent_config[AI_ASSISTANT_NAME]}")

        try:
            with st.spinner("Brewing conversations..."):
                st.session_state.manager, st.session_state.user_agent = setup_chat(
                    llm_provider=st.session_state.config["llm_provider"],
                    model_name=st.session_state.config["model_name"],
                    content_text=content_text, user_agent_prompt=user_agent_prompt_content,
                    selected_persona_files=selected_persona_files, 
                    group_chat_agent_display_config=group_chat_agent_config
                )
                initial_messages, next_agent = initiate_chat_task(st.session_state.user_agent, st.session_state.manager, task_prompt, content_text)
                st.session_state.messages = initial_messages; st.session_state.next_agent = next_agent
                st.session_state.chat_initialized = True
                st.session_state.total_input_tokens += sum(estimate_tokens(text) for text in [task_prompt, content_text, user_agent_prompt_content, load_personas(PERSONA_DIR, selected_persona_files)])
                for msg in initial_messages:
                    if msg.get("name") != USER_NAME: st.session_state.total_output_tokens += estimate_tokens(msg.get("content", ""))
        except Exception as e: st.session_state.error_message = f"Setup/Initiation failed: {e}"; st.session_state.chat_initialized = False; logger.error(f"Chat start error: {traceback.format_exc()}")
        finally: st.session_state.processing = False
        update_token_warning(); st.rerun()

if st.session_state.error_message: st.error(st.session_state.error_message)

chat_container = st.container()
with chat_container:
    if st.session_state.chat_initialized and st.session_state.manager:
        display_messages(st.session_state.messages)
        next_agent_code_name = st.session_state.next_agent.name if st.session_state.next_agent else None
        # Use AGENT_DISPLAY_NAMES_KEY for chat display purposes
        display_next_agent_name = st.session_state.get(AGENT_DISPLAY_NAMES_KEY, {}).get(next_agent_code_name, next_agent_code_name)

        if next_agent_code_name and not st.session_state.processing:
            if next_agent_code_name == USER_NAME:
                with st.form(key=f'user_input_form_{len(st.session_state.messages)}'):
                    col1, col2 = st.columns([0.9, 0.1])
                    with col1: user_input = st.text_area("Enter message:", height=80, key=f"user_input_{len(st.session_state.messages)}", label_visibility="collapsed")
                    with col2: send_button_pressed = st.form_submit_button("➤")
                    if send_button_pressed:
                        if user_input.strip():
                            st.session_state.processing = True; st.session_state.error_message = None
                            st.session_state.total_input_tokens += estimate_tokens(user_input)
                            with st.spinner("Sending..."):
                                try:
                                    new_msgs, next_ag = send_user_message(st.session_state.manager, st.session_state.user_agent, user_input)
                                    st.session_state.messages.extend(new_msgs); st.session_state.next_agent = next_ag
                                except Exception as e: st.session_state.error_message = f"Send error: {e}"; logger.error(f"Send error: {traceback.format_exc()}")
                            st.session_state.processing = False; update_token_warning(); st.rerun()
                        else: st.warning("Empty message.")
            else: 
                st.markdown(f"**Running turn for:** {display_next_agent_name}...")
                st.session_state.processing = True; st.session_state.error_message = None
                with st.spinner(f"Thinking... {display_next_agent_name} is responding..."):
                    try:
                        new_msgs, next_ag_after_ai = run_agent_step(st.session_state.manager, st.session_state.next_agent)
                        st.session_state.messages.extend(new_msgs)
                        for msg in new_msgs:
                            if msg.get("name") != USER_NAME: st.session_state.total_output_tokens += estimate_tokens(msg.get("content", ""))
                        st.session_state.next_agent = st.session_state.user_agent # Default to user after AI turn
                    except Exception as e: st.session_state.error_message = f"Error during {display_next_agent_name}'s turn: {e}"; st.session_state.next_agent = st.session_state.user_agent; logger.error(f"{display_next_agent_name} error: {traceback.format_exc()}")
                st.session_state.processing = False; update_token_warning(); st.rerun()
        elif not next_agent_code_name and st.session_state.chat_initialized and not st.session_state.processing:
             st.success("Chat finished or awaiting next step.")

if st.session_state.chat_initialized or st.session_state.error_message:
     if st.sidebar.button("Clear Chat / Reset", key="clear_chat"):
         keys_to_clear = list(default_values.keys())
         keys_to_clear.extend([USER_EDIT_KEY, "persona_multiselect"])
         for key in keys_to_clear:
             if key in st.session_state:
                 if key == SELECTED_PERSONAS_KEY: st.session_state[key] = []
                 elif key == PERSONA_DISPLAY_NAMES_MAP_KEY: st.session_state[key] = {}
                 elif key == AGENT_DISPLAY_NAMES_KEY: st.session_state[key] = {}
                 elif key == "messages": st.session_state[key] = []
                 else: del st.session_state[key]
         for key, value in default_values.items():
            if key not in st.session_state: st.session_state[key] = value
         initialize_prompts_and_personas()
         logger.info("Chat state cleared and prompts re-initialized.")
         st.session_state.total_input_tokens = 0; st.session_state.total_output_tokens = 0
         update_token_warning(); st.rerun()
