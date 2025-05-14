import logging
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChatManager, Agent, GroupChat
from typing import List, Optional, Sequence, Tuple, Dict, Union
import json
import re
import functools # Added for functools.partial
import os # Added for listing persona files

from LLMConfiguration import LLMConfiguration, logger # Assuming LLMConfiguration also has a logger

# Configure logging specifically for this module if not inheriting
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Use a named logger for this module to avoid conflicts if LLMConfiguration has its own basicConfig
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.DEBUG) # Or whatever level is appropriate

# --- NEW FUNCTION ---
def load_personas(persona_dir: str, persona_filenames: List[str]) -> str:
    """
    Loads and concatenates the content of specified persona markdown files.

    Args:
        persona_dir: The directory where persona files are stored.
        persona_filenames: A list of persona filenames (e.g., ["Persona1.md", "Persona3.md"]).

    Returns:
        A single string containing the concatenated content of the persona files.
        Returns an empty string if no files are specified or if errors occur.
    """
    combined_system_message = []
    if not persona_filenames:
        module_logger.info("No persona filenames provided to load_personas.")
        return ""

    for filename in persona_filenames:
        try:
            # It's good practice to ensure filenames are not traversing directories unexpectedly
            safe_filename = os.path.basename(filename)
            if not safe_filename.endswith(".md"):
                module_logger.warning(f"Skipping non-markdown file: {safe_filename}")
                continue

            filepath = os.path.join(persona_dir, safe_filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                if content:
                    # Optional: Remove DisplayName if present, or keep it as part of the persona
                    # For now, we'll take the whole content.
                    # lines = content.splitlines()
                    # if lines and re.match(r"^DisplayName:\s*(.+)$", lines[0].strip(), re.IGNORECASE):
                    #    content = "\n".join(lines[1:]).strip()
                    combined_system_message.append(content)
                    module_logger.info(f"Successfully loaded persona: {safe_filename}")
                else:
                    module_logger.warning(f"Persona file {safe_filename} is empty.")
        except FileNotFoundError:
            module_logger.error(f"Persona file not found: {filename} in directory {persona_dir}")
        except Exception as e:
            module_logger.error(f"Error reading persona file {filename}: {e}")
    
    return "\n---\n".join(combined_system_message) # Separate personas clearly

# --- MODIFIED FUNCTION ---
def create_agent(
    name: str, llm_config: LLMConfiguration,
    system_message_content: Optional[str] = None, # This will now come from load_personas
    agent_type="assistant"
    ) -> autogen.Agent:
    # Use a more robust default if system_message_content is empty or None
    system_message_for_agent = system_message_content.strip() if system_message_content and system_message_content.strip() else "You are a helpful assistant. Your primary goal is to assist users effectively."
    
    # The rest of the function remains the same
    if not system_message_for_agent: # Should not happen with the default above, but as a safeguard
         raise ValueError(f"System message is empty for agent {name}")
    config = llm_config.get_config()
    if not config: raise ValueError(f"Invalid LLM configuration for agent {name}")

    if agent_type == "user_proxy":
        return UserProxyAgent(name=name, system_message=system_message_for_agent, human_input_mode="NEVER", code_execution_config=False, llm_config=config, default_auto_reply="", is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"))
    else:
        return AssistantAgent(name=name, system_message=system_message_for_agent, human_input_mode="NEVER", llm_config=config, is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"))

# --- Deprecated or to be re-evaluated ---
def read_system_message(filename: str) -> Tuple[Optional[str], str]:
    # This function's role might change. It could be used by load_personas to process individual files
    # or be deprecated if DisplayName is handled differently or not at all from these files.
    # For now, let's assume load_personas handles file reading directly.
    default_system_message = "You are a helpful assistant."
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            full_content = file.read()
            lines = full_content.splitlines()
            display_name = None
            system_message_content = full_content.strip()
            if lines:
                match = re.match(r"^DisplayName:\s*(.+)$", lines[0].strip(), re.IGNORECASE)
                if match:
                    display_name = match.group(1).strip()
                    system_message_content = "\n".join(lines[1:]).strip()
                    module_logger.info(f"Parsed DisplayName '{display_name}' from {filename}")
                else:
                    module_logger.info(f"No DisplayName found in first line of {filename}.")
            else:
                 module_logger.warning(f"System message file {filename} is empty. Using default.")
                 return None, default_system_message # Return None for display_name if file empty
            # Ensure system_message_content is not empty after stripping DisplayName
            if not system_message_content and display_name: # Had display name but no content after
                 module_logger.warning(f"System message content is empty in {filename} after parsing DisplayName. Using default for content.")
                 return display_name, default_system_message
            elif not system_message_content and not display_name: # File had only whitespace or was empty
                 return None, default_system_message

            return display_name, system_message_content
    except FileNotFoundError:
        module_logger.error(f"System message file not found: {filename}")
        return None, default_system_message
    except Exception as e:
        module_logger.error(f"Error reading system message file {filename}: {e}")
        return None, default_system_message


def custom_speaker_selection(
    last_speaker: Agent, 
    groupchat: GroupChat, 
    agent_display_names: Dict[str, str]
) -> Agent:
    module_logger.debug(f"--- Entering custom_speaker_selection (with DisplayNames) ---")
    module_logger.debug(f"Last speaker: {last_speaker.name if last_speaker else 'None'}")
    module_logger.debug(f"Available agent code names: {[a.name for a in groupchat.agents]}")
    module_logger.debug(f"Agent display names map: {agent_display_names}")

    user_agent = next((agent for agent in groupchat.agents if isinstance(agent, UserProxyAgent)), None)
    if not user_agent:
        module_logger.error("No UserProxyAgent (User) found!")
        # Attempt to return the first agent if no user agent, or raise error if no agents
        if groupchat.agents:
            module_logger.warning("Returning the first available agent as UserProxyAgent is missing.")
            return groupchat.agents[0]
        else:
            # This case should ideally be prevented by checks before calling (e.g., in create_groupchat)
            raise ValueError("No agents in groupchat and UserProxyAgent is missing!")


    if not groupchat.messages:
        module_logger.info("No messages yet, User by default.")
        return user_agent

    last_message_obj = groupchat.messages[-1]
    message_content = str(last_message_obj.get('content', '')).strip()
    module_logger.debug(f"Checking message content for selection: '{message_content[:150]}{'...' if len(message_content) > 150 else ''}'")

    if message_content.rstrip().endswith("TERMINATE"):
        module_logger.info("TERMINATE detected. Selecting User.")
        return user_agent

    lower_message_content = message_content.lower()
    last_mention_index = -1
    agent_to_select = None

    # Iterate through agents to find mentions based on their display names
    for agent in groupchat.agents:
        # Skip UserProxyAgent for mention-based selection to avoid self-reply loops on generic terms
        if isinstance(agent, UserProxyAgent): 
            continue

        agent_code_name = agent.name
        display_name = agent_display_names.get(agent_code_name)
        if not display_name: # If no display name, this agent cannot be mentioned by display name
            module_logger.warning(f"Display name not found for agent code name '{agent_code_name}'. Skipping this agent for mention check.")
            continue
        
        pattern = display_name.lower()
        if not pattern: continue # Skip if display name is empty

        # Find the last occurrence of the pattern
        current_idx = -1
        start_search_from = 0
        final_found_idx = -1
        while True:
            found_idx = lower_message_content.find(pattern, start_search_from)
            if found_idx == -1:
                break
            final_found_idx = found_idx # Keep track of the last found index
            start_search_from = found_idx + len(pattern) # Continue search after this find
        
        current_idx = final_found_idx # This is the index of the last occurrence

        if current_idx > last_mention_index:
            last_mention_index = current_idx
            agent_to_select = agent
            module_logger.debug(f"Mention of '{display_name}' (for '{agent_code_name}') at index {current_idx}. Tentatively selected.")
        elif current_idx == last_mention_index and current_idx != -1:
            # If two different display names are mentioned starting at the same last position,
            # prefer the longer one (more specific mention).
            current_selected_display_name = agent_display_names.get(agent_to_select.name, "") if agent_to_select else ""
            if len(pattern) > len(current_selected_display_name.lower()):
                agent_to_select = agent
                module_logger.debug(f"Mention of '{display_name}' at same index {current_idx}, but is longer. Switched.")
    
    # Final decision based on mentions
    selected_agent_name = "None" # For logging
    if agent_to_select:
        selected_agent_name = agent_to_select.name
        # If the selected agent is the one who just spoke, it implies they might be asking a question
        # or trying to pass the turn. In many scenarios, it's better to return to the User/orchestrator.
        if agent_to_select == last_speaker:
            module_logger.info(f"Speaker '{last_speaker.name}' mentioned themselves (DisplayName: '{agent_display_names.get(last_speaker.name)}'). Defaulting to User.")
            next_speaker = user_agent
        else:
            module_logger.info(f"Next speaker by mention: '{agent_to_select.name}' (DisplayName: '{agent_display_names.get(agent_to_select.name)}').")
            next_speaker = agent_to_select
    else:
        # If no specific agent was mentioned, default to the UserProxyAgent.
        module_logger.info(f"No specific agent display name mentioned in the last message. Defaulting to User.")
        next_speaker = user_agent

    module_logger.debug(f"--- Exiting custom_speaker_selection (Selected: {next_speaker.name if next_speaker else 'None'}) ---")
    return next_speaker

def create_groupchat(agents: Sequence[Agent], agent_display_names: Dict[str, str], max_round: int = 50) -> GroupChat:
     if not any(isinstance(agent, UserProxyAgent) for agent in agents):
         raise ValueError("GroupChat requires at least one UserProxyAgent.")

     # Use functools.partial to adapt custom_speaker_selection for GroupChat
     # This correctly passes the agent_display_names to your custom selection logic
     speaker_selection_func = functools.partial(custom_speaker_selection, agent_display_names=agent_display_names)

     return GroupChat(
         agents=list(agents), # Ensure it's a list
         messages=[],
         max_round=max_round,
         speaker_selection_method=speaker_selection_func, # Use the custom function
         allow_repeat_speaker=False, # Consider if this should be True or False based on desired flow
     )

def create_groupchat_manager(groupchat: GroupChat, llm_config: LLMConfiguration) -> GroupChatManager:
    config = llm_config.get_config()
    if not config: raise ValueError("Invalid LLM configuration for GroupChatManager.")
    # Ensure the manager also uses a termination message condition
    return GroupChatManager(
        groupchat=groupchat, 
        llm_config=config,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE")
    )

def initiate_chat_task(
    user_agent: UserProxyAgent, 
    manager: GroupChatManager, 
    initial_prompt: str,
    system_content_for_group: Optional[str] = None # New parameter
    ) -> Tuple[List[Dict], Optional[Agent]]:
    manager.groupchat.reset() # Clears previous messages
    
    final_prompt = initial_prompt.strip()

    # The 'system_content_for_group' is now less relevant if personas define agent behavior.
    # However, it could still be used for a general group instruction or context NOT part of a persona.
    # If it's meant to be part of the persona, it should be in the persona files.
    if system_content_for_group and system_content_for_group.strip():
        formatted_content = (
            f"General Context for this Discussion (provided by the user, distinct from individual assistant personas):\n"
            f"---\n"
            f"{system_content_for_group.strip()}\n"
            f"---\n"
            f"User's Initial Task:\n"
            f"{final_prompt}"
        )
        final_prompt = formatted_content # Prepend if provided
        module_logger.info(f"Prepended 'system_content_for_group' to the initial_prompt.")

    # The first message to kick off the conversation from the user
    initial_message = {"role": "user", "content": final_prompt, "name": user_agent.name}
    manager.groupchat.messages.append(initial_message) # Add to history
    module_logger.info(f"Initial message from {user_agent.name} (potentially with context) added to manager history.")

    # Let the speaker selection logic pick the first AI responder
    # The GroupChatManager's 'run_chat' usually handles this, but for step-by-step, we select first.
    # Pass the user_agent as the 'last_speaker' to select_speaker, as the user initiated.
    next_speaker = manager.groupchat.select_speaker(user_agent, manager.groupchat)
    module_logger.info(f"Initiate chat: First AI speaker selected by custom logic: {next_speaker.name if next_speaker else 'None'}")

    # Return the initial message (as it's part of the history) and the next speaker
    return [initial_message], next_speaker


def run_agent_step(manager: GroupChatManager, speaker: Agent) -> Tuple[List[Dict], Optional[Agent]]:
    newly_added_messages = [] # Store messages generated in this step
    next_speaker = None
    user_agent = next((agent for agent in manager.groupchat.agents if isinstance(agent, UserProxyAgent)), None) # Find user_agent

    try:
        module_logger.info(f"--- Running step for agent: {speaker.name} ---")
        
        # The 'sender' for generate_reply should be the manager or an orchestrator agent
        # if the manager itself is not an agent (which is typical for GroupChatManager).
        # AutoGen's GroupChatManager handles the flow, so we pass `manager` which acts as the orchestrator.
        # The `messages` argument should be the current history.
        messages_context = manager.groupchat.messages
        
        len_before_reply = len(messages_context) # Messages before this agent replies
        
        # Generate reply using the speaker agent
        reply = speaker.generate_reply(messages=messages_context, sender=manager) # Sender is manager
        
        # Messages after the agent might have added its reply (or not)
        # Some agents might directly append to groupchat.messages or return the message
        messages_after_reply = manager.groupchat.messages 
        len_after_reply = len(messages_after_reply)
        num_new_messages = len_after_reply - len_before_reply

        if num_new_messages > 0:
            # If messages were added directly to groupchat.messages by the agent's `send` or `receive`
            newly_added_messages = messages_after_reply[len_before_reply:]
            # module_logger.info(f"Agent {speaker.name} added {num_new_messages} message(s) directly to chat history.")
        elif reply is not None:
            # If agent returns a reply (string or dict) and doesn't add it itself, we add it.
            reply_content = None
            if isinstance(reply, dict): # Standard message dict
                reply_content = reply.get("content")
            elif isinstance(reply, str): # Simple string reply
                reply_content = reply
            
            if reply_content is not None:
                # Determine role based on agent type for the message
                role = "user" if isinstance(speaker, UserProxyAgent) else "assistant"
                manual_message = {"role": role, "content": reply_content, "name": speaker.name}
                
                # Add this reply to the group chat's message history
                manager.groupchat.messages.append(manual_message)
                newly_added_messages = [manual_message] # This is the new message
                module_logger.info(f"Manually added message from {speaker.name} (Role: {role}).")
            else:
                module_logger.warning(f"Could not extract content from reply by {speaker.name}. Reply was: {reply}")
        else:
             # Agent generated no reply (e.g. None)
             module_logger.info(f"Agent {speaker.name} generated no reply (returned None).")

        # Select the next speaker based on the updated history and last speaker
        next_speaker = manager.groupchat.select_speaker(speaker, manager.groupchat)
        module_logger.info(f"Step for {speaker.name} done. Next speaker selected: {next_speaker.name if next_speaker else 'None'}")

    except Exception as e:
        module_logger.error(f"Error during agent step for {speaker.name}: {e}", exc_info=True)
        # Fallback: if an error occurs, set next speaker to UserProxyAgent if available
        if user_agent:
            next_speaker = user_agent
            module_logger.info("Error occurred. Defaulting next speaker to User.")
        else:
            # This is a critical state; no user agent to fall back to.
            module_logger.error("User agent not found for error fallback. Chat cannot reliably continue.")
            # Depending on desired robustness, could raise or try to pick any agent.
            # For now, next_speaker remains None or as set before error if any.
    
    return newly_added_messages, next_speaker


def send_user_message(
    manager: GroupChatManager, 
    user_agent: UserProxyAgent, 
    user_message: str
    ) -> Tuple[List[Dict], Optional[Agent]]:
    # If the user sends an empty message, it might mean they just want the AI to continue.
    # The speaker selection logic should handle who speaks next.
    if not user_message or not user_message.strip():
        module_logger.info("User sent an empty message. Triggering next speaker selection.")
        # Determine the last actual speaker to correctly select the next one.
        # If history is empty or last speaker was user, this logic might need adjustment
        # based on how `select_speaker` handles it.
        last_actual_speaker = user_agent # Default to user if no history or tricky situation
        if manager.groupchat.messages:
             try:
                 last_msg_name = manager.groupchat.messages[-1].get('name')
                 # Ensure agent exists before setting, otherwise default to user_agent
                 if last_msg_name: last_actual_speaker = manager.groupchat.agent_by_name(last_msg_name) or user_agent
             except Exception: # Catch any error in getting agent by name
                 pass # Stick with user_agent as last_actual_speaker
        
        next_speaker = manager.groupchat.select_speaker(last_actual_speaker, manager.groupchat)
        return [], next_speaker # No new message from user, just return next speaker

    # Construct the message from the user
    message_dict = {"role": "user", "content": user_message.strip(), "name": user_agent.name}
    
    # Add user's message to the group chat history
    manager.groupchat.messages.append(message_dict)
    
    # Select the next speaker after the user's message
    # The user_agent is passed as the 'last_speaker' to the selection method
    next_speaker = manager.groupchat.select_speaker(user_agent, manager.groupchat)
    module_logger.info(f"User message sent. Selected next speaker: {next_speaker.name if next_speaker else 'None'}")
    
    return [message_dict], next_speaker
