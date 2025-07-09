# games/common/utils/conversation.py

from typing import List, Dict, Any, Optional
from datetime import datetime

from ..base.models import BaseConversationTurn


class ConversationManager:
    """Utility class for managing conversation turns and state"""
    
    @staticmethod
    def create_turn(turn_number: int, speaker: str, message: str) -> BaseConversationTurn:
        """Create a conversation turn"""
        return BaseConversationTurn(
            turn_number=turn_number,
            speaker=speaker,
            message=message,
            timestamp=datetime.now()
        )
    
    @staticmethod
    def add_turn_to_conversation(conversation: List[Dict[str, Any]], 
                               speaker: str, message: str) -> BaseConversationTurn:
        """Add a turn to an existing conversation list"""
        turn_number = len(conversation) + 1
        turn = ConversationManager.create_turn(turn_number, speaker, message)
        conversation.append(turn.model_dump())
        return turn
    
    @staticmethod
    def get_last_turn(conversation: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Get the last turn in the conversation"""
        return conversation[-1] if conversation else None
    
    @staticmethod
    def get_turns_by_speaker(conversation: List[Dict[str, Any]], 
                           speaker: str) -> List[Dict[str, Any]]:
        """Get all turns by a specific speaker"""
        return [turn for turn in conversation if turn.get("speaker") == speaker]
    
    @staticmethod
    def format_conversation_for_display(conversation: List[Dict[str, Any]]) -> str:
        """Format conversation for human-readable display"""
        formatted = []
        for turn in conversation:
            speaker = turn.get("speaker", "Unknown")
            message = turn.get("message", "")
            turn_num = turn.get("turn_number", "?")
            formatted.append(f"Turn {turn_num} - {speaker.title()}: {message}")
        return "\n".join(formatted)
    
    @staticmethod
    def get_conversation_statistics(conversation: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the conversation"""
        if not conversation:
            return {"total_turns": 0, "speakers": [], "turn_distribution": {}}
        
        speakers = set()
        turn_distribution = {}
        total_words = 0
        
        for turn in conversation:
            speaker = turn.get("speaker", "Unknown")
            message = turn.get("message", "")
            
            speakers.add(speaker)
            turn_distribution[speaker] = turn_distribution.get(speaker, 0) + 1
            total_words += len(message.split())
        
        return {
            "total_turns": len(conversation),
            "speakers": list(speakers),
            "turn_distribution": turn_distribution,
            "total_words": total_words,
            "average_words_per_turn": total_words / len(conversation) if conversation else 0
        }