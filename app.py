import streamlit as st
import logging
import traceback
import os
import json
from typing import Tuple, Optional, Dict

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
    read_system_message # Updated signature: returns (display_name, content)
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---

USER_NAME = "User" # Internal code name for the user agent
PERSONA1_NAME = "Persona1"
PERSONA2_NAME = "Persona2"

USER_SYS_MSG_FILE = "User.md"
PERSONA1_DEFAULT_SYS_MSG_FILE = "Persona1.md"
PERSONA2_DEFAULT_SYS_MSG_FILE = "Persona2.md"

CONTENT_INJECTION_MARKER = "## Content" # Renamed from POLICY_INJECTION_MARKER
MAX_MESSAGES_DISPLAY = 50
CONTENT_TEXT_KEY = "content_text_input" # Renamed from POLICY_TEXT_KEY
TASK_PROMPT_KEY = "initial_prompt_input"

PERSONA1_EDIT_KEY = "persona1_editable_prompt"
PERSONA2_EDIT_KEY = "persona2_editable_prompt"
USER_EDIT_KEY = "user_editable_prompt"

AGENT_DISPLAY_NAMES_KEY = "agent_display_names"

AGENT_CONFIG = {
    PERSONA1_NAME: {"file": PERSONA1_DEFAULT_SYS_MSG_FILE, "key": PERSONA1_EDIT_KEY},
    PERSONA2_NAME: {"file": PERSONA2_DEFAULT_SYS_MSG_FILE, "key": PERSONA2_EDIT_KEY},
    USER_NAME: {"file": USER_SYS_MSG_FILE, "key": USER_EDIT_KEY},
}

CONTEXT_LIMIT = 1_000_000
WARNING_THRESHOLD = 0.85

# --- Helper Functions ---

def estimate_tokens(text: str) -> int:
    return len(text or "") // 4

def initialize_editable_prompts():
    if AGENT_DISPLAY_NAMES_KEY not in st.session_state:
        st.session_state[AGENT_DISPLAY_NAMES_KEY] = {}

    for agent_code_name, config in AGENT_CONFIG.items():
        if config["key"] not in st.session_state or agent_code_name not in st.session_state[AGENT_DISPLAY_NAMES_KEY]:
            try:
                display_name, system_content = read_system_message(config["file"])
                st.session_state[config["key"]] = system_content
                # Store display name, defaulting to agent_code_name if not parsed from file
                actual_display_name = display_name or agent_code_name
                st.session_state[AGENT_DISPLAY_NAMES_KEY][agent_code_name] = actual_display_name
                logger.info(f"Loaded prompt for '{agent_code_name}' (Display: '{actual_display_name}') into state ('{config['key']}').")
            except Exception as e:
                st.session_state[config["key"]] = f"Error loading from {config['file']}: {e}"
                st.session_state[AGENT_DISPLAY_NAMES_KEY][agent_code_name] = agent_code_name # Fallback
                logger.error(f"Failed to load prompt for {agent_code_name} from {config['file']}: {e}")

def update_token_warning():
    # ... (token warning logic remains the same)
    content_text = st.session_state.get(CONTENT_TEXT_KEY, "") # Renamed from policy_text and POLICY_TEXT_KEY
    task_text = st.session_state.get(TASK_PROMPT_KEY, "")
    persona1_prompt = st.session_state.get(PERSONA1_EDIT_KEY, "")
    persona2_prompt = st.session_state.get(PERSONA2_EDIT_KEY, "")
    user_prompt = st.session_state.get(USER_EDIT_KEY, "")

    content_tokens = estimate_tokens(content_text) # Renamed from policy_tokens
    task_tokens = estimate_tokens(task_text)
    persona1_tokens = estimate_tokens(persona1_prompt)
    persona2_tokens = estimate_tokens(persona2_prompt)
    user_tokens = estimate_tokens(user_prompt)

    total_system_prompt_tokens = persona1_tokens + persona2_tokens + user_tokens
    total_input_tokens = content_tokens + task_tokens # Renamed from policy_tokens
    total_estimated_tokens = total_input_tokens + total_system_prompt_tokens

    if hasattr(st.session_state, 'token_info_placeholder') and st.session_state.token_info_placeholder:
        st.session_state.token_info_placeholder.caption(f"Estimated Input Tokens: ~{total_estimated_tokens:,} / {CONTEXT_LIMIT:,}")

    if hasattr(st.session_state, 'token_warning_placeholder') and st.session_state.token_warning_placeholder:
        if total_estimated_tokens > CONTEXT_LIMIT * WARNING_THRESHOLD:
            st.session_state.token_warning_placeholder.warning(f"Inputs approaching context limit ({WARNING_THRESHOLD*100:.0f}%). Total: ~{total_estimated_tokens:,}")
        else:
            st.session_state.token_warning_placeholder.empty()

def setup_chat(
    llm_provider: str,
    model_name: str,
    content_text: Optional[str], # Renamed from policy_text
    agent_prompts: Dict[str, str], # agent_code_name: system_message_content
    agent_display_names: Dict[str, str] # agent_code_name: display_name
    ) -> Tuple[GroupChatManager, UserProxyAgent]:

    logger.info(f"Setting up chat with provider: {llm_provider}, model: {model_name}")

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

    agents_dict = {}
    try:
        for agent_code_name, config_entry in AGENT_CONFIG.items():
            system_message_content = agent_prompts.get(agent_code_name)
            if not system_message_content:
                _, system_message_content = read_system_message(config_entry["file"]) # Fallback
                if not system_message_content: raise ValueError(f"Could not load system message for {agent_code_name}.")


            agent_type = "user_proxy" if agent_code_name == USER_NAME else "assistant"
            agents_dict[agent_code_name] = create_agent(
                name=agent_code_name, # Internal code name
                llm_config=llm_config,
                system_message_content=system_message_content,
                agent_type=agent_type)
        logger.info("Agents created successfully.")
    except Exception as e: logger.error(f"Agent creation error: {e}", exc_info=True); raise

    user_agent = agents_dict[USER_NAME]
    chat_participants = list(agents_dict.values())

    try:
        # Pass agent_display_names to create_groupchat
        groupchat = create_groupchat(chat_participants, agent_display_names, max_round=50)
        manager = create_groupchat_manager(groupchat, llm_config)
        logger.info("GroupChat and Manager created.")
    except Exception as e: logger.error(f"GroupChat/Manager creation error: {e}", exc_info=True); raise

    return manager, user_agent

def display_messages(messages):
    num_messages = len(messages)
    start_index = max(0, num_messages - MAX_MESSAGES_DISPLAY)
    if num_messages > MAX_MESSAGES_DISPLAY: st.warning(f"Displaying last {MAX_MESSAGES_DISPLAY} of {num_messages} messages.")

    for msg in messages[start_index:]:
        internal_sender_name = msg.get("name", "System") # This is the agent code name
        sender_display_name = st.session_state.get(AGENT_DISPLAY_NAMES_KEY, {}).get(internal_sender_name, internal_sender_name)
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = [item["text"] if isinstance(item, dict) and "text" in item else str(item) for item in content]
            content = "\n".join(parts)
        elif not isinstance(content, str): content = str(content)

        avatar_map = {USER_NAME: "🧑", PERSONA1_NAME: "🧑‍🏫", PERSONA2_NAME: "🧐"}
        avatar = avatar_map.get(internal_sender_name, "⚙️")

        with st.chat_message("user" if internal_sender_name == USER_NAME else "assistant", avatar=avatar):
            # Display name and content without extra quotes or redundant names
            st.markdown(f"**{sender_display_name}:**\n{content}")

# --- Streamlit App UI ---
st.title("🤖 Chat with agentT and agentH")

# --- Initialization ---
default_values = {
    "chat_initialized": False, "processing": False, "error_message": None,
    "config": None, "manager": None, "user_agent": None,
    "messages": [], "next_agent": None,
    TASK_PROMPT_KEY: "", CONTENT_TEXT_KEY: "", # Renamed from POLICY_TEXT_KEY
}
for key, value in default_values.items():
    if key not in st.session_state: st.session_state[key] = value

initialize_editable_prompts() # Populates AGENT_DISPLAY_NAMES_KEY and editable prompts

if not st.session_state.config:
    try:
        st.session_state.config = {"llm_provider": VERTEX_AI, "model_name": "gemini-1.5-pro-002"}
    except Exception as e: st.sidebar.error(f"Config load failed: {e}"); st.stop()

with st.sidebar.expander("Configure AI Personas & Task", expanded=True):
    st.caption("Define the AI personas and the initial task.")
    for agent_code_name, config_info in AGENT_CONFIG.items():
        # Use display name from session state for the label
        display_name_for_label = st.session_state[AGENT_DISPLAY_NAMES_KEY].get(agent_code_name, agent_code_name)
        st.text_area(
            f"System Prompt for {display_name_for_label} (Editable):",
            key=config_info["key"],
            height=150,
            disabled=st.session_state.chat_initialized,
            help=f"Define background for {display_name_for_label}. The 'DisplayName' is read from the first line of the .md file.",
            on_change=update_token_warning
        )
    st.session_state.token_info_placeholder = st.sidebar.empty()
    st.session_state.token_warning_placeholder = st.sidebar.empty()

    content_text_input = st.sidebar.text_area( # Renamed variable for clarity, though not strictly necessary
        "Enter Content Text (Optional):", height=100, key=CONTENT_TEXT_KEY, # Changed label and key
        disabled=st.session_state.chat_initialized, on_change=update_token_warning)
    initial_prompt_input = st.sidebar.text_area(
        "Enter the Initial Task or Question:", height=150, key=TASK_PROMPT_KEY,
        disabled=st.session_state.chat_initialized, on_change=update_token_warning)
    update_token_warning()

if st.sidebar.button("🚀 Start Chat", key="start_chat",
                    disabled=st.session_state.chat_initialized or
                             not st.session_state.get(TASK_PROMPT_KEY, "").strip() or
                             not st.session_state.config):
    task_prompt = st.session_state.get(TASK_PROMPT_KEY, "").strip()
    if not st.session_state.chat_initialized and task_prompt and st.session_state.config:
        st.session_state.processing = True; st.session_state.error_message = None
        try:
            with st.spinner("Brewing conversations..."):
                current_agent_prompts = {name: st.session_state.get(cfg["key"], "") for name, cfg in AGENT_CONFIG.items()}
                # Get the display names from session state
                current_display_names = st.session_state[AGENT_DISPLAY_NAMES_KEY]

                st.session_state.manager, st.session_state.user_agent = setup_chat(
                    llm_provider=st.session_state.config.get("llm_provider", VERTEX_AI),
                    model_name=st.session_state.config.get("model_name", "gemini-1.5-pro-002"),
                    content_text=st.session_state.get(CONTENT_TEXT_KEY, ""), # Renamed from policy_text and POLICY_TEXT_KEY
                    agent_prompts=current_agent_prompts,
                    agent_display_names=current_display_names # Pass display names here
                )
                initial_messages, next_agent = initiate_chat_task(
                    st.session_state.user_agent,
                    st.session_state.manager,
                    task_prompt,
                    system_content_for_group=st.session_state.get(CONTENT_TEXT_KEY, "") # New argument
                )
                st.session_state.messages = initial_messages
                st.session_state.next_agent = next_agent
                st.session_state.chat_initialized = True
        except Exception as e:
            st.session_state.error_message = f"Setup/Initiation failed: {e}"; st.session_state.chat_initialized = False
            logger.error(f"Chat start error: {traceback.format_exc()}")
        finally: st.session_state.processing = False
        st.rerun()

if st.session_state.error_message: st.error(st.session_state.error_message)

chat_container = st.container()
with chat_container:
    if st.session_state.chat_initialized and st.session_state.manager:
        display_messages(st.session_state.messages) # display_messages uses AGENT_DISPLAY_NAMES_KEY from session_state
        next_agent_code_name = st.session_state.next_agent.name if st.session_state.next_agent else None
        # Use display name for whose turn it is
        display_next_agent_name = st.session_state[AGENT_DISPLAY_NAMES_KEY].get(next_agent_code_name, next_agent_code_name)

        if next_agent_code_name and not st.session_state.processing:
            if next_agent_code_name == USER_NAME: # Still use code name for this check
                st.markdown(f"**Your turn (as {display_next_agent_name}):**")
                with st.form(key=f'user_input_form_{len(st.session_state.messages)}'):
                    user_input = st.text_input("Enter your message:", key=f"user_input_{len(st.session_state.messages)}")
                    if st.form_submit_button("✉️ Send Message"):
                        if user_input.strip():
                            st.session_state.processing = True; st.session_state.error_message = None
                            with st.spinner("Sending message..."):
                                try:
                                    new_msgs, next_ag = send_user_message(st.session_state.manager, st.session_state.user_agent, user_input)
                                    st.session_state.messages.extend(new_msgs); st.session_state.next_agent = next_ag
                                except Exception as e: st.session_state.error_message = f"Send error: {e}"; logger.error(f"Send error: {traceback.format_exc()}")
                            st.session_state.processing = False; st.rerun()
                        else: st.warning("Please enter a message.")
            else: # AI Persona's Turn
                st.markdown(f"**Running turn for:** {display_next_agent_name}...")
                st.session_state.processing = True; st.session_state.error_message = None
                with st.spinner(f"Thinking... {display_next_agent_name} is responding..."):
                    try:
                        new_msgs, _ = run_agent_step(st.session_state.manager, st.session_state.next_agent)
                        st.session_state.messages.extend(new_msgs)
                        # Force next speaker to be the user
                        st.session_state.next_agent = st.session_state.user_agent
                    except Exception as e:
                        st.session_state.error_message = f"Error during {display_next_agent_name}'s turn: {e}"; st.session_state.next_agent = None
                        logger.error(f"{display_next_agent_name} turn error: {traceback.format_exc()}")
                st.session_state.processing = False; st.rerun()
        elif not next_agent_code_name and st.session_state.chat_initialized and not st.session_state.processing:
             st.success("Chat finished or awaiting next step.")

if st.session_state.chat_initialized or st.session_state.error_message:
     if st.sidebar.button("Clear Chat / Reset", key="clear_chat"):
         # Clear all relevant session state keys
         keys_to_clear = list(default_values.keys()) + [PERSONA1_EDIT_KEY, PERSONA2_EDIT_KEY, USER_EDIT_KEY, AGENT_DISPLAY_NAMES_KEY]
         for key in keys_to_clear:
             if key in st.session_state: del st.session_state[key]
         # Re-initialize after clearing
         initialize_editable_prompts()
         logger.info("Chat state cleared and prompts re-initialized.")
         update_token_warning(); st.rerun()
