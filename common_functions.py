import logging
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChatManager, Agent, GroupChat
from typing import List, Optional, Sequence, Tuple, Dict, Union
import json
import re
import functools # Added for functools.partial

from LLMConfiguration import logger # Assuming LLMConfiguration also has a logger

# Configure logging specifically for this module if not inheriting
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Use a named logger for this module to avoid conflicts if LLMConfiguration has its own basicConfig
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.DEBUG) # Or whatever level is appropriate

def read_system_message(filename: str) -> Tuple[Optional[str], str]:
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
                    system_message_content = "
".join(lines[1:]).strip()
                    module_logger.info(f"Parsed DisplayName '{display_name}' from {filename}")
                else:
                    module_logger.info(f"No DisplayName found in first line of {filename}.")
            else:
                 module_logger.warning(f"System message file {filename} is empty. Using default.")
                 return None, default_system_message
            return display_name, system_message_content
    except FileNotFoundError:
        module_logger.error(f"System message file not found: {filename}")
        return None, default_system_message
    except Exception as e:
        module_logger.error(f"Error reading system message file {filename}: {e}")
        return None, default_system_message

def create_agent(
    name: str, llm_config: LLMConfiguration,
    system_message_content: Optional[str] = None,
    agent_type="assistant"
    ) -> autogen.Agent:
    system_message_for_agent = system_message_content.strip() if system_message_content else "You are a helpful assistant."
    if not system_message_for_agent:
         raise ValueError(f"System message is empty for agent {name}")
    config = llm_config.get_config()
    if not config: raise ValueError(f"Invalid LLM configuration for agent {name}")

    if agent_type == "user_proxy":
        return UserProxyAgent(name=name, system_message=system_message_for_agent, human_input_mode="NEVER", code_execution_config=False, llm_config=config, default_auto_reply="", is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"))
    else:
        return AssistantAgent(name=name, system_message=system_message_for_agent, human_input_mode="NEVER", llm_config=config, is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"))

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
        return groupchat.agents[0] if groupchat.agents else ValueError("No agents in groupchat!")

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

    for agent in groupchat.agents:
        agent_code_name = agent.name
        display_name = agent_display_names.get(agent_code_name)
        if not display_name:
            module_logger.warning(f"Display name not found for agent code name '{agent_code_name}'. Skipping this agent for mention check.")
            continue
        
        pattern = display_name.lower()
        if not pattern: continue

        current_idx = -1
        start_search_from = 0
        final_found_idx = -1
        while True:
            found_idx = lower_message_content.find(pattern, start_search_from)
            if found_idx == -1:
                break
            final_found_idx = found_idx
            start_search_from = found_idx + len(pattern) # Start search after the current find
        
        current_idx = final_found_idx

        if current_idx > last_mention_index:
            last_mention_index = current_idx
            agent_to_select = agent
            module_logger.debug(f"Mention of '{display_name}' (for '{agent_code_name}') at index {current_idx}. Selected.")
        elif current_idx == last_mention_index and current_idx != -1:
            current_selected_display_name = agent_display_names.get(agent_to_select.name, "")
            if len(pattern) > len(current_selected_display_name.lower()):
                agent_to_select = agent
                module_logger.debug(f"Mention of '{display_name}' at same index {current_idx}, but is longer. Switched.")
    
    selected_agent_name = "None"
    if agent_to_select:
        selected_agent_name = agent_to_select.name
        if agent_to_select == last_speaker:
            module_logger.info(f"Speaker '{last_speaker.name}' mentioned themselves ('{agent_display_names.get(last_speaker.name)}'). Defaulting to User.")
            next_speaker = user_agent
        else:
            module_logger.info(f"Next speaker by mention: '{agent_to_select.name}' (DisplayName: '{agent_display_names.get(agent_to_select.name)}').")
            next_speaker = agent_to_select
    else:
        module_logger.info(f"No specific agent display name mentioned. Defaulting to User.")
        next_speaker = user_agent

    module_logger.debug(f"--- Exiting custom_speaker_selection (Selected: {next_speaker.name if next_speaker else 'None'}) ---")
    return next_speaker

def create_groupchat(agents: Sequence[Agent], agent_display_names: Dict[str, str], max_round: int = 50) -> GroupChat:
     if not any(isinstance(agent, UserProxyAgent) for agent in agents):
         raise ValueError("GroupChat requires at least one UserProxyAgent.")

     # Use functools.partial to adapt custom_speaker_selection for GroupChat
     speaker_selection_func = functools.partial(custom_speaker_selection, agent_display_names=agent_display_names)

     return GroupChat(
         agents=list(agents),
         messages=[],
         max_round=max_round,
         speaker_selection_method=speaker_selection_func,
         allow_repeat_speaker=False,
     )

def create_groupchat_manager(groupchat: GroupChat, llm_config: LLMConfiguration) -> GroupChatManager:
    config = llm_config.get_config()
    if not config: raise ValueError("Invalid LLM configuration for GroupChatManager.")
    return GroupChatManager(groupchat=groupchat, llm_config=config, is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"))

def initiate_chat_task(
    user_agent: UserProxyAgent, 
    manager: GroupChatManager, 
    initial_prompt: str,
    system_content_for_group: Optional[str] = None # New parameter
    ) -> Tuple[List[Dict], Optional[Agent]]:
    manager.groupchat.reset()
    
    final_prompt = initial_prompt.strip()
    if system_content_for_group and system_content_for_group.strip():
        formatted_content = (
            f"Contextual Content provided by the User (this is not part of the direct task, but general context for all assistants):
"
            f"---
"
            f"{system_content_for_group.strip()}
"
            f"---

"
            f"Original Task from User:
"
            f"{final_prompt}"
        )
        final_prompt = formatted_content
        module_logger.info(f"Prepended system_content_for_group to the initial_prompt.")

    initial_message = {"role": "user", "content": final_prompt, "name": user_agent.name}
    manager.groupchat.messages.append(initial_message)
    module_logger.info(f"Initial message from {user_agent.name} (potentially with context) added to manager history.")

    # Let speaker selection logic (which now uses display names) pick the first AI responder
    next_speaker = manager.groupchat.select_speaker(user_agent, manager.groupchat)
    module_logger.info(f"Initiate chat: First AI speaker selected by custom logic: {next_speaker.name if next_speaker else 'None'}")

    return [initial_message], next_speaker

def run_agent_step(manager: GroupChatManager, speaker: Agent) -> Tuple[List[Dict], Optional[Agent]]:
    newly_added_messages = []
    next_speaker = None
    user_agent = next((agent for agent in manager.groupchat.agents if isinstance(agent, UserProxyAgent)), None)
    try:
        module_logger.info(f"--- Running step for agent: {speaker.name} ---")
        messages_context = manager.groupchat.messages
        len_before_reply = len(messages_context)
        reply = speaker.generate_reply(messages=messages_context, sender=manager)
        messages_after_reply = manager.groupchat.messages
        len_after_reply = len(messages_after_reply)
        num_new_messages = len_after_reply - len_before_reply

        if num_new_messages > 0:
            newly_added_messages = messages_after_reply[len_before_reply:]
        elif reply is not None:
            reply_content = reply.get("content") if isinstance(reply, dict) else str(reply) if isinstance(reply, str) else None
            if reply_content is not None:
                role = "user" if isinstance(speaker, UserProxyAgent) else "assistant"
                manual_message = {"role": role, "content": reply_content, "name": speaker.name}
                manager.groupchat.messages.append(manual_message)
                newly_added_messages = [manual_message]
                module_logger.info(f"Manually added message from {speaker.name} (Role: {role}).")
            else:
                module_logger.warning(f"Could not extract content from reply by {speaker.name}.")
        else:
             module_logger.info(f"Agent {speaker.name} generated no reply.")

        next_speaker = manager.groupchat.select_speaker(speaker, manager.groupchat)
        module_logger.info(f"Step for {speaker.name} done. Next: {next_speaker.name if next_speaker else 'None'}")
    except Exception as e:
        module_logger.error(f"Error during agent step for {speaker.name}: {e}", exc_info=True)
        if user_agent: next_speaker = user_agent
        else: module_logger.error("User agent not found for error fallback.")
    return newly_added_messages, next_speaker

def send_user_message(manager: GroupChatManager, user_agent: UserProxyAgent, user_message: str) -> Tuple[List[Dict], Optional[Agent]]:
    if not user_message or not user_message.strip():
        last_actual_speaker = user_agent
        if manager.groupchat.messages:
             try:
                 last_msg_name = manager.groupchat.messages[-1].get('name')
                 if last_msg_name: last_actual_speaker = manager.groupchat.agent_by_name(last_msg_name) or user_agent
             except Exception: pass
        next_speaker = manager.groupchat.select_speaker(last_actual_speaker, manager.groupchat)
        return [], next_speaker

    message_dict = {"role": "user", "content": user_message.strip(), "name": user_agent.name}
    manager.groupchat.messages.append(message_dict)
    next_speaker = manager.groupchat.select_speaker(user_agent, manager.groupchat)
    module_logger.info(f"User message sent. Selected next: {next_speaker.name if next_speaker else 'None'}")
    return [message_dict], next_speaker
