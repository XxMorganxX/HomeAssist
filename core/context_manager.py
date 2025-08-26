import config
from openai import OpenAI
import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Any

class ContextManager:
    """
    Pure context data provider. Returns exactly what should be sent as context
    for different use cases by querying the conversation oracle (ConversationManager).
    
    Responsibilities:
    - Provide context data for different use cases (tool selection, response generation)
    - Manage context summaries and storage
    - Return ready-to-use context arrays
    
    Does NOT:
    - Make decisions about WHEN to generate context
    - Implement context strategy logic
    - Trigger context updates automatically
    """
    
    def __init__(self):
        pass
    
    def get_context_for_tools(self, conversation_manager, max_messages: int = 6) -> List[Dict[str, str]]:
        """
        Get context optimized for tool selection.
        
        Args:
            conversation_manager: The conversation oracle
            max_messages: Maximum recent messages to include
            
        Returns:
            List of messages ready for API (system prompt + recent messages)
        """
        all_messages = conversation_manager.get_messages()
        
        # For realtime API: no system prompt in messages, add it
        if hasattr(conversation_manager, 'system_prompt'):
            system_message = {"role": "system", "content": conversation_manager.system_prompt}
            if len(all_messages) <= max_messages:
                return [system_message] + all_messages
            else:
                recent_messages = all_messages[-max_messages:]
                return [system_message] + recent_messages
        else:
            # Traditional API: system prompt already in messages
            if len(all_messages) <= max_messages + 1:  # +1 for system prompt
                return all_messages
            else:
                system_prompt = all_messages[0]
                recent_messages = all_messages[-max_messages:]
                return [system_prompt] + recent_messages
    
    def get_context_for_response(self, conversation_manager, use_summary: bool = True) -> List[Dict[str, str]]:
        """
        Get context optimized for response generation.
        Uses sliding window + summary if conversation is long enough.
        
        Args:
            conversation_manager: The conversation oracle
            use_summary: Whether to use summary + sliding window for long conversations
            
        Returns:
            List of messages ready for API
        """
        all_messages = conversation_manager.get_messages()
        
        # Check if we should use summary context
        min_messages = getattr(config, 'CONTEXT_SUMMARY_MIN_MESSAGES', 10)
        window_size = getattr(config, 'REALTIME_SLIDING_WINDOW_SIZE', 6)
        
        # Short conversation or summary disabled - return full context
        if not use_summary or len(all_messages) <= min_messages:
            return self._add_system_prompt_if_needed(conversation_manager, all_messages)
        
        # Long conversation - use summary + sliding window
        try:
            summary_data = self.get_summary_data()
            if summary_data and summary_data.get('User_Summary') and summary_data.get('Response_Summary'):
                return self._create_summarized_context(conversation_manager, all_messages, summary_data, window_size)
            else:
                # No valid summary available, return recent messages only
                return self._create_windowed_context(conversation_manager, all_messages, window_size)
        except Exception:
            # Fallback to windowed context if summary fails
            return self._create_windowed_context(conversation_manager, all_messages, window_size)
    
    def get_full_context(self, conversation_manager) -> List[Dict[str, str]]:
        """
        Get complete conversation context (all messages).
        
        Args:
            conversation_manager: The conversation oracle
            
        Returns:
            Complete conversation ready for API
        """
        all_messages = conversation_manager.get_messages()
        return self._add_system_prompt_if_needed(conversation_manager, all_messages)
    
    def generate_summary(self, conversation_manager) -> Dict[str, str]:
        """
        Generate a new conversation summary using AI.
        
        Args:
            conversation_manager: The conversation oracle
            
        Returns:
            Dict with User_Summary and Response_Summary
        """
        client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
        conversation = conversation_manager.get_messages()
        
        # Format conversation for summarization (exclude system messages)
        conv_text = "\\n".join([
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in conversation if msg.get('role') != 'system'
        ])
        
        # Include previous summary for continuity
        previous_summary = self.get_summary_data()
        conv_text = f"Previous User Summary: {previous_summary.get('User_Summary', '')}\\n\\nPrevious Response Summary: {previous_summary.get('Response_Summary', '')}\\n\\n{conv_text}"
        
        system_instruction = self._get_summary_instructions()
        user_content = f"Summarize this conversation:\\n\\n{conv_text}"
        
        # Debug output if enabled
        if getattr(config, 'DISPLAY_CONTEXT', False):
            self._debug_summary_call(conversation, system_instruction, user_content)
        
        response = client.chat.completions.create(
            model=config.CONTEXT_SUMMARY_MODEL,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"},
            max_tokens=450,
            temperature=0.3
        )
        
        return json.loads(response.choices[0].message.content)
    
    def save_summary(self, conversation_manager, summary_data: Dict[str, str]) -> None:
        """
        Save conversation summary to persistent storage.
        
        Args:
            conversation_manager: The conversation oracle
            summary_data: Dict with User_Summary and Response_Summary
        """
        conversation = conversation_manager.get_messages()
        # Count messages excluding system prompt
        conversation_msgs = [msg for msg in conversation if msg.get('role') != 'system']
        total_user = sum(1 for msg in conversation_msgs if msg.get("role") == "user")
        total_assistant = sum(1 for msg in conversation_msgs if msg.get("role") == "assistant")
        
        full_summary = {
            "conversation": {
                "timestamp": datetime.now().isoformat(),
                "message_count": len(conversation_msgs),
                "total_user_messages": total_user,
                "total_assistant_messages": total_assistant,
                "User_Summary": summary_data.get("User_Summary", ""),
                "Response_Summary": summary_data.get("Response_Summary", ""),
                "System_Prompt": config.SYSTEM_PROMPT[:200] + "..." if len(config.SYSTEM_PROMPT) > 200 else config.SYSTEM_PROMPT
            }
        }
        
        os.makedirs(os.path.dirname(config.session_summary_file), exist_ok=True)
        with open(config.session_summary_file, 'w') as f:
            json.dump(full_summary, f, indent=2)
    
    def get_summary_data(self) -> Dict[str, Any]:
        """
        Get existing summary data from storage.
        
        Returns:
            Dict with summary data or empty dict if no summary exists
        """
        try:
            if os.path.exists(config.session_summary_file):
                with open(config.session_summary_file, 'r') as f:
                    session_summary = json.load(f)
                return session_summary.get("conversation", {})
            return {}
        except Exception:
            return {}
    
    def update_summary(self, conversation_manager) -> None:
        """
        Generate a new summary and save it.
        
        Args:
            conversation_manager: The conversation oracle
        """
        try:
            summary_data = self.generate_summary(conversation_manager)
            self.save_summary(conversation_manager, summary_data)
        except Exception as e:
            print(f"❌ Error updating summary: {e}")
            raise
    
    # Private helper methods
    
    def _add_system_prompt_if_needed(self, conversation_manager, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Add system prompt if using realtime API (where it's not in messages)."""
        if hasattr(conversation_manager, 'system_prompt'):
            # Realtime API - add system prompt
            system_message = {"role": "system", "content": conversation_manager.system_prompt}
            return [system_message] + messages
        else:
            # Traditional API - system prompt already in messages
            return messages
    
    def _create_summarized_context(self, conversation_manager, messages: List[Dict[str, str]], 
                                 summary_data: Dict[str, Any], window_size: int) -> List[Dict[str, str]]:
        """Create context with summary + sliding window."""
        # Create summary message
        user_summary = summary_data.get('User_Summary', '')
        response_summary = summary_data.get('Response_Summary', '')
        summary_text = f"CONVERSATION SUMMARY - User Actions: {user_summary} | Assistant Actions: {response_summary}"
        
        if getattr(config, 'REALTIME_SUMMARY_AS_SYSTEM_MESSAGE', True):
            summary_message = {"role": "system", "content": summary_text}
        else:
            summary_message = {"role": "user", "content": f"[Context Summary] {summary_text}"}
        
        # Get recent messages
        recent_messages = messages[-window_size:] if len(messages) > window_size else messages
        
        # Combine: system prompt + summary + recent messages
        if hasattr(conversation_manager, 'system_prompt'):
            system_message = {"role": "system", "content": conversation_manager.system_prompt}
            return [system_message, summary_message] + recent_messages
        else:
            # For traditional API, assume first message is system prompt
            if messages and messages[0].get('role') == 'system':
                return [messages[0], summary_message] + recent_messages
            else:
                return [summary_message] + recent_messages
    
    def _create_windowed_context(self, conversation_manager, messages: List[Dict[str, str]], 
                               window_size: int) -> List[Dict[str, str]]:
        """Create context with just recent messages."""
        recent_messages = messages[-window_size:] if len(messages) > window_size else messages
        return self._add_system_prompt_if_needed(conversation_manager, recent_messages)
    
    def _get_summary_instructions(self) -> str:
        """Get instructions for AI summary generation."""
        return """
        You are tasked with generating a summary of the session conversation history. 
        You will be given a conversation history, and you will need to generate a summary 
        of the conversation in two main groups:
        - User Summary
        - Response Summary

        The User Summary is a summary of the user's questions and actions.
        The Response Summary is a summary of the assistant's responses.
        
        Respond in valid JSON format with this structure:
        {
            "User_Summary": "Brief summary of user questions and actions",
            "Response_Summary": "Brief summary of assistant responses and help provided"
        }
        """
    
    def _debug_summary_call(self, conversation: List[Dict], system_instruction: str, user_content: str) -> None:
        """Debug output for summary generation."""
        print("=" * 60)
        print("🔍 CONTEXT SUMMARY API CALL DEBUG")
        print("=" * 60)
        print(f"Model: {config.CONTEXT_SUMMARY_MODEL}")
        print(f"Max tokens: 450")
        print(f"Temperature: 0.3")
        print(f"Response format: json_object")
        print("-" * 60)
        print("SYSTEM MESSAGE:")
        print(system_instruction)
        print("-" * 60)
        print("USER MESSAGE:")
        print(user_content)
        print("-" * 60)
        print("RAW CONVERSATION DATA:")
        print(f"Total conversation messages: {len(conversation)}")
        for i, msg in enumerate(conversation):
            print(f"  {i+1}. {msg.get('role', 'unknown')}: {msg.get('content', 'no content')[:100]}...")
        print("=" * 60)