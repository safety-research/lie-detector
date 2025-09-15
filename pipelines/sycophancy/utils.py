def chunk_conversation(text):
    """
    Split a conversation string into human and assistant sections.
    Returns a list of dictionaries with 'role' and 'content' keys.
    """
    chunks = []
    current_chunk = None
    current_content = []

    lines = text.strip().split('\n')

    for line in lines:
        # Check if this line starts a new section
        if line.strip().startswith('Human:'):
            # Save previous chunk if exists
            if current_chunk:
                chunks.append({
                    'role': current_chunk,
                    'content': '\n'.join(current_content).strip()
                })
            # Start new human chunk
            current_chunk = 'human'
            current_content = [line[6:].strip()]  # Remove 'Human:' prefix

        elif line.strip().startswith('Assistant:') or line.strip().startswith('A:'):
            # Save previous chunk if exists
            if current_chunk:
                chunks.append({
                    'role': current_chunk,
                    'content': '\n'.join(current_content).strip()
                })
            # Start new assistant chunk
            current_chunk = 'assistant'
            if line.strip().startswith('Assistant:'):
                current_content = [line[10:].strip()]  # Remove 'Assistant:' prefix
            else:
                current_content = [line[2:].strip()]  # Remove 'A:' prefix

        else:
            # Continue current chunk
            if current_chunk:
                current_content.append(line)

    # Don't forget the last chunk
    if current_chunk:
        chunks.append({
            'role': current_chunk,
            'content': '\n'.join(current_content).strip()
        })

    return chunks


def format_feedback_conversation(prompt):
    """Format feedback conversation messages into a single string."""
    conversation_parts = []

    for message in prompt:
        msg_type = message.get('type', '')
        content = message.get('content', '')

        if msg_type == 'human':
            conversation_parts.append(f"Human: {content}")
        elif msg_type in ['assistant', 'ai']:
            conversation_parts.append(f"Assistant: {content}")

    return "\n\n".join(conversation_parts)