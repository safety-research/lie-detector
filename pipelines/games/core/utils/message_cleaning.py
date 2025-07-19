# games/core/utils/message_cleaning.py

import re
from typing import List, Optional


class MessageCleaner:
    """Utility for cleaning agent messages to ensure natural spoken communication"""

    # Patterns to remove
    STAGE_DIRECTION_PATTERNS = [
        r'\([^)]*\)',  # Parenthetical actions: (smiles), (nervously)
        r'\[[^\]]*\]',  # Bracketed actions: [laughs], [pauses]
        r'\*[^*]*\*',  # Asterisk actions: *sighs*, *thinks*
        r'<[^>]*>',  # Angle bracket actions: <whispers>, <gestures>
    ]

    # Patterns that indicate non-speech content
    NON_SPEECH_INDICATORS = [
        r'^(I |He |She |They |We |One |The )\w+ly ',  # "I nervously say..."
        r'^\w+ing,',  # "Smiling, I respond..."
        r'^With \w+,',  # "With hesitation, ..."
    ]

    @staticmethod
    def clean_message(text: str, speaker_name: Optional[str] = None) -> str:
        """Remove all non-speech elements from a message"""
        if not text:
            return ""

        cleaned = text.strip()

        # Remove stage directions
        for pattern in MessageCleaner.STAGE_DIRECTION_PATTERNS:
            cleaned = re.sub(pattern, '', cleaned)

        # Remove multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)

        # Split into lines and process each
        lines = cleaned.split('\n')
        clean_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip lines that are pure stage direction
            if line.startswith('(') and line.endswith(')'):
                continue
            if line.startswith('[') and line.endswith(']'):
                continue
            if line.startswith('*') and line.endswith('*'):
                continue

            # Remove speaker labels if present
            if speaker_name and ':' in line:
                parts = line.split(':', 1)
                if parts[0].strip().lower() == speaker_name.lower():
                    line = parts[1].strip()
                elif any(name in parts[0] for name in ['Detective', 'Suspect', 'Blue', 'Red']):
                    line = parts[1].strip()

            # Remove common non-speech prefixes
            for pattern in MessageCleaner.NON_SPEECH_INDICATORS:
                if re.match(pattern, line):
                    # Try to extract just the spoken part
                    quote_match = re.search(r'"([^"]+)"', line)
                    if quote_match:
                        line = quote_match.group(1)
                        break

            # Final cleanup
            line = line.strip(' "\'')
            if line:
                clean_lines.append(line)

        # Join lines and do final cleanup
        result = ' '.join(clean_lines) if len(clean_lines) <= 1 else '\n'.join(clean_lines)

        # Remove any remaining artifacts
        result = re.sub(r'\s+([,.!?])', r'\1', result)  # Fix spacing before punctuation
        result = re.sub(r'([,.!?])\s*\1+', r'\1', result)  # Remove duplicate punctuation

        return result.strip()

    @staticmethod
    def is_valid_speech(text: str) -> bool:
        """Check if text represents valid spoken communication"""
        cleaned = text.strip()

        # Must have actual content
        if not cleaned or len(cleaned) < 3:
            return False

        # Should not be purely stage direction
        if (cleaned.startswith('(') and cleaned.endswith(')')) or \
                (cleaned.startswith('[') and cleaned.endswith(']')) or \
                (cleaned.startswith('*') and cleaned.endswith('*')):
            return False

        # Should contain actual words
        if not re.search(r'[a-zA-Z]{2,}', cleaned):
            return False

        return True

    @staticmethod
    def extract_speech_only(messages: List[str]) -> List[str]:
        """Extract only valid speech from a list of messages"""
        return [
            MessageCleaner.clean_message(msg)
            for msg in messages
            if MessageCleaner.is_valid_speech(msg)
        ]