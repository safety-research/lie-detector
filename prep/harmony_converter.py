"""
Harmony Format Converter for Lie Detection Dataset

This script converts reasoning-enhanced samples to OpenAI Harmony format
using the official openai-harmony library and prepares tokenized data for Together AI.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import tiktoken

try:
    from openai_harmony import (
        Author,
        Conversation,
        DeveloperContent,
        HarmonyEncodingName,
        Message,
        Role,
        SystemContent,
        ToolDescription,
        load_harmony_encoding,
        ReasoningEffort
    )
except ImportError:
    print("Please install openai-harmony: pip install openai-harmony")
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HarmonyConverter:
    """Converts lie detection samples to Harmony format with tokenization."""

    def __init__(self, reasoning_effort: str = "high", max_length: int = 4096):
        """
        Initialize the Harmony converter.

        Args:
            reasoning_effort: One of "high", "medium", "low"
            max_length: Maximum sequence length for tokenization
        """
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.reasoning_effort = self._get_reasoning_effort(reasoning_effort)
        self.max_length = max_length

        # Initialize tiktoken for tokenization
        # Using o200k_harmony encoding for Harmony format
        try:
            self.tokenizer = tiktoken.get_encoding("o200k_harmony")
        except:
            # Fallback to o200k_base if harmony encoding not available
            logger.warning("o200k_harmony encoding not found, using o200k_base")
            self.tokenizer = tiktoken.get_encoding("o200k_base")

    def _get_reasoning_effort(self, effort: str) -> ReasoningEffort:
        """Convert string to ReasoningEffort enum."""
        mapping = {
            "high": ReasoningEffort.HIGH,
            "medium": ReasoningEffort.MEDIUM,
            "low": ReasoningEffort.LOW
        }
        return mapping.get(effort.lower(), ReasoningEffort.HIGH)

    def create_system_message(self, include_date: bool = True) -> SystemContent:
        """
        Create the system message for Harmony format.

        Args:
            include_date: Whether to include current date

        Returns:
            SystemContent object
        """
        system_content = SystemContent.new().with_reasoning_effort(self.reasoning_effort)

        if include_date:
            current_date = datetime.now().strftime("%Y-%m-%d")
            system_content = system_content.with_conversation_start_date(current_date)

        return system_content

    def create_developer_message(self, instructions: str = None) -> DeveloperContent:
        """
        Create the developer message with instructions.

        Args:
            instructions: Custom instructions for the model

        Returns:
            DeveloperContent object
        """
        if instructions is None:
            instructions = (
                "You are a helpful AI assistant trained to detect deception in conversations. "
                "Analyze conversations carefully and determine if lies were told."
            )

        return DeveloperContent.new().with_instructions(instructions)

    def convert_sample_to_harmony(self, sample: Dict[str, Any]) -> Conversation:
        """
        Convert a single enhanced sample to Harmony format.

        Args:
            sample: Enhanced sample with reasoning

        Returns:
            Harmony Conversation object
        """
        messages_list = []

        # Add system message
        system_content = self.create_system_message()
        messages_list.append(Message.from_role_and_content(Role.SYSTEM, system_content))

        # Add developer message
        developer_content = self.create_developer_message()
        messages_list.append(Message.from_role_and_content(Role.DEVELOPER, developer_content))

        # Add conversation messages
        conversation_messages = sample.get('messages', [])
        for msg in conversation_messages:
            role_str = msg.get('role', 'user')
            content = msg.get('content', '')

            # Map roles to Harmony Role enum
            if role_str == 'user':
                role = Role.USER
            elif role_str == 'assistant':
                role = Role.ASSISTANT
            elif role_str == 'system':
                # System prompts in conversation become developer messages
                role = Role.DEVELOPER
            else:
                role = Role.USER  # Default fallback

            # For assistant messages, use final channel for regular responses
            if role == Role.ASSISTANT:
                message = Message.from_role_and_content(role, content)
                message = message.with_channel("final")
            else:
                message = Message.from_role_and_content(role, content)

            messages_list.append(message)

        # Add lie detection prompt
        lie_prompt = sample.get('lie_detection_prompt',
                                'Did you lie? Answer (A) or (B). A: Yes, I lied  B: No, I was truthful.')
        messages_list.append(Message.from_role_and_content(Role.USER, lie_prompt))

        # Add reasoning in analysis channel (if present)
        reasoning = sample.get('reasoning', {})
        if reasoning:
            # Analysis channel - the chain of thought
            analysis_content = reasoning.get('analysis', '')
            if analysis_content:
                analysis_message = Message.from_role_and_content(
                    Role.ASSISTANT,
                    analysis_content
                ).with_channel("analysis")
                messages_list.append(analysis_message)

            # Final channel - the conclusion and answer
            final_content = reasoning.get('final', '')
            if final_content:
                final_message = Message.from_role_and_content(
                    Role.ASSISTANT,
                    final_content
                ).with_channel("final")
                messages_list.append(final_message)

        # Create conversation from messages
        return Conversation.from_messages(messages_list)

    def render_and_tokenize(self, conversation: Conversation) -> Tuple[List[int], List[int], Dict[str, Any]]:
        """
        Render conversation and tokenize for training.

        Args:
            conversation: Harmony Conversation object

        Returns:
            Tuple of (input_ids, labels, metadata)
        """
        # Get tokens using the harmony encoding
        tokens = self.encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)

        # The tokens from harmony library are the full conversation tokens
        # We need to split them for training:
        # - input_ids: Full sequence including the response
        # - labels: Masked version where we only predict the assistant response

        # Find where the last assistant message starts
        # This is a simplified approach - in practice you might need more sophisticated splitting

        # For training, we want to predict the final assistant responses (analysis + final channels)
        # So we mask everything before the last assistant messages

        # Convert tokens to list if needed
        if not isinstance(tokens, list):
            tokens = list(tokens)

        # Create labels (same as input_ids but with masking)
        # We'll use -100 as the ignore token for loss calculation
        labels = tokens.copy()

        # Find the position where we start predicting
        # This would be the start of the reasoning response
        # For now, we'll mask the first 80% of tokens (you can adjust this)
        mask_until = int(len(tokens) * 0.8)
        for i in range(mask_until):
            labels[i] = -100

        # Truncate or pad to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            labels = labels[:self.max_length]
        else:
            # Pad with the pad token (typically 0 or a special pad token ID)
            pad_token_id = 0  # You might need to adjust this based on the tokenizer
            padding_length = self.max_length - len(tokens)
            tokens = tokens + [pad_token_id] * padding_length
            labels = labels + [-100] * padding_length  # Don't compute loss on padding

        metadata = {
            'original_length': len(tokens),
            'truncated': len(tokens) > self.max_length,
            'mask_position': mask_until
        }

        return tokens, labels, metadata

    def tokenize_text_fallback(self, text: str) -> Tuple[List[int], List[int], str]:
        """
        Fallback tokenization using tiktoken directly.

        Args:
            text: Text to tokenize

        Returns:
            Tuple of (input_ids, labels, text)
        """
        # Encode the text
        tokens = self.tokenizer.encode(text)

        # Find where the assistant response starts (for label masking)
        # Look for the last occurrence of assistant channel markers
        text_lower = text.lower()
        assistant_start = max(
            text_lower.rfind("<|start|>assistant<|channel|>analysis"),
            text_lower.rfind("<|start|>assistant<|channel|>final"),
            text_lower.rfind("<|start|>assistant")
        )

        # Estimate token position for masking
        if assistant_start > 0:
            # Rough estimate: characters before assistant response / average chars per token
            chars_before = assistant_start
            avg_chars_per_token = len(text) / len(tokens) if tokens else 4
            mask_until = int(chars_before / avg_chars_per_token)
        else:
            # Default: mask 80% of tokens
            mask_until = int(len(tokens) * 0.8)

        # Create labels with masking
        labels = tokens.copy()
        for i in range(min(mask_until, len(labels))):
            labels[i] = -100

        # Truncate or pad
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            labels = labels[:self.max_length]
        else:
            pad_token_id = 0
            padding_length = self.max_length - len(tokens)
            tokens = tokens + [pad_token_id] * padding_length
            labels = labels + [-100] * padding_length

        return tokens, labels, text

    def process_file(self, input_file: Path, output_dir: Path,
                     save_parquet: bool = True, save_jsonl: bool = True) -> int:
        """
        Process a file of enhanced samples to Harmony format with tokenization.

        Args:
            input_file: Path to input JSONL file
            output_dir: Directory for output files
            save_parquet: Whether to save Parquet format
            save_jsonl: Whether to save JSONL format

        Returns:
            Number of samples processed
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine file type
        file_type = 'train' if 'train' in input_file.name else 'val'

        tokenized_samples = []
        text_samples = []
        metadata_list = []

        # Process each sample
        with open(input_file, 'r') as f:
            for i, line in enumerate(f):
                if i % 100 == 0:
                    logger.info(f"Processing sample {i}...")

                sample = json.loads(line)

                try:
                    # Convert to Harmony format
                    conversation = self.convert_sample_to_harmony(sample)

                    # Render and tokenize
                    input_ids, labels, token_metadata = self.render_and_tokenize(conversation)

                    # Build text representation from original sample for debugging
                    text_parts = []

                    # Add system message
                    text_parts.append(f"<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nReasoning: {self.reasoning_effort}<|end|>")

                    # Add developer message
                    text_parts.append(f"<|start|>developer<|message|>You are a helpful AI assistant trained to detect deception in conversations. Analyze conversations carefully and determine if lies were told.<|end|>")

                    # Add conversation messages from original sample
                    messages = sample.get('messages', [])
                    for msg in messages:
                        role = msg.get('role', 'user')
                        content = msg.get('content', '')
                        if role == 'assistant':
                            text_parts.append(f"<|start|>{role}<|channel|>final<|message|>{content}<|end|>")
                        else:
                            text_parts.append(f"<|start|>{role}<|message|>{content}<|end|>")

                    # Add lie detection prompt
                    lie_prompt = sample.get('lie_detection_prompt',
                                           'Did you lie? Answer (A) or (B). A: Yes, I lied  B: No, I was truthful.')
                    text_parts.append(f"<|start|>user<|message|>{lie_prompt}<|end|>")

                    # Add reasoning if present
                    reasoning = sample.get('reasoning', {})
                    if reasoning.get('analysis'):
                        text_parts.append(f"<|start|>assistant<|channel|>analysis<|message|>{reasoning['analysis']}<|end|>")
                    if reasoning.get('final'):
                        text_parts.append(f"<|start|>assistant<|channel|>final<|message|>{reasoning['final']}<|end|>")

                    text = "\n".join(text_parts)

                except Exception as e:
                    logger.warning(f"Failed to process sample {i} with harmony library: {e}")
                    logger.warning("Falling back to text tokenization")

                    # Fallback: construct text and tokenize directly
                    messages = sample.get('messages', [])
                    text_parts = []

                    # Add basic system and developer messages
                    text_parts.append(f"<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nReasoning: high<|end|>")
                    text_parts.append(f"<|start|>developer<|message|>You are a helpful AI assistant trained to detect deception.<|end|>")

                    # Add conversation
                    for msg in messages:
                        role = msg.get('role', 'user')
                        content = msg.get('content', '')
                        if role == 'assistant':
                            text_parts.append(f"<|start|>{role}<|channel|>final<|message|>{content}<|end|>")
                        else:
                            text_parts.append(f"<|start|>{role}<|message|>{content}<|end|>")

                    # Add lie detection and reasoning
                    lie_prompt = sample.get('lie_detection_prompt', 'Did you lie?')
                    text_parts.append(f"<|start|>user<|message|>{lie_prompt}<|end|>")

                    reasoning = sample.get('reasoning', {})
                    if reasoning.get('analysis'):
                        text_parts.append(f"<|start|>assistant<|channel|>analysis<|message|>{reasoning['analysis']}<|end|>")
                    if reasoning.get('final'):
                        text_parts.append(f"<|start|>assistant<|channel|>final<|message|>{reasoning['final']}<|return|>")

                    text = "\n".join(text_parts)
                    input_ids, labels, text = self.tokenize_text_fallback(text)
                    token_metadata = {'fallback': True}

                # Extract metadata
                metadata = sample.get('metadata', {})
                metadata.update(token_metadata)
                # Ensure sample_id is always a string
                if 'sample_id' not in metadata:
                    metadata['sample_id'] = f'sample_{i}'
                else:
                    metadata['sample_id'] = str(metadata['sample_id'])

                # Create attention mask (1 for real tokens, 0 for padding)
                attention_mask = [1 if token_id != 0 else 0 for token_id in input_ids]

                # Create tokenized sample for training (only required columns for Together AI)
                tokenized_sample = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels
                }

                tokenized_samples.append(tokenized_sample)
                metadata_list.append(metadata)

                # Create text sample for JSONL output
                text_sample = {
                    'text': text,
                    'metadata': metadata
                }
                text_samples.append(text_sample)

        logger.info(f"Processed {len(tokenized_samples)} samples")

        # Save JSONL format (text only, for debugging)
        if save_jsonl:
            jsonl_file = output_dir / f"{file_type}_harmony.jsonl"
            with open(jsonl_file, 'w') as f:
                for sample in text_samples:
                    f.write(json.dumps(sample) + '\n')
            logger.info(f"Text samples: {len(text_samples)}")
            logger.info(f"Saved JSONL to {jsonl_file}")

        # Save Parquet format for training (with tokenized data)
        if save_parquet:
            # Create DataFrame with tokenized data
            df = pd.DataFrame(tokenized_samples)

            # Save Parquet file with tokenized data
            parquet_file = output_dir / f"{file_type}_harmony.parquet"
            df.to_parquet(parquet_file, engine='pyarrow', compression='snappy', index=False)
            logger.info(f"Saved tokenized Parquet to {parquet_file}")

            # Log sample of data structure
            logger.info(f"Parquet columns: {df.columns.tolist()}")
            logger.info(f"Sample input_ids length: {len(df.iloc[0]['input_ids']) if len(df) > 0 else 0}")

            # Also save metadata separately
            meta_df = pd.DataFrame(metadata_list)
            meta_parquet = output_dir / f"{file_type}_metadata.parquet"
            meta_df.to_parquet(meta_parquet, engine='pyarrow', compression='snappy', index=False)
            logger.info(f"Saved metadata to {meta_parquet}")

        return len(tokenized_samples)


def main():
    parser = argparse.ArgumentParser(
        description='Convert reasoning-enhanced samples to Harmony format with tokenization')
    parser.add_argument('input_file', type=str,
                        help='Path to enhanced samples file (e.g., train_with_reasoning.jsonl)')
    parser.add_argument('--output-dir', type=str,
                        help='Output directory (default: same directory with /harmony suffix)')
    parser.add_argument('--reasoning-effort', type=str, default='high',
                        choices=['high', 'medium', 'low'],
                        help='Reasoning effort level')
    parser.add_argument('--max-length', type=int, default=4096,
                        help='Maximum sequence length for tokenization')
    parser.add_argument('--no-parquet', action='store_true',
                        help='Skip Parquet output')
    parser.add_argument('--no-jsonl', action='store_true',
                        help='Skip JSONL output')

    args = parser.parse_args()

    # Setup paths
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent / 'harmony'

    # Initialize converter
    converter = HarmonyConverter(
        reasoning_effort=args.reasoning_effort,
        max_length=args.max_length
    )

    # Process file
    logger.info(f"Converting {input_path} to Harmony format with tokenization...")
    num_processed = converter.process_file(
        input_file=input_path,
        output_dir=output_dir,
        save_parquet=not args.no_parquet,
        save_jsonl=not args.no_jsonl
    )

    # Save conversion report
    report = {
        'timestamp': datetime.now().isoformat(),
        'input_file': str(input_path),
        'output_dir': str(output_dir),
        'num_samples': num_processed,
        'reasoning_effort': args.reasoning_effort,
        'max_length': args.max_length,
        'formats_created': []
    }

    if not args.no_jsonl:
        report['formats_created'].append('jsonl')
    if not args.no_parquet:
        report['formats_created'].append('parquet (tokenized)')

    report_file = output_dir / 'conversion_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Conversion complete! Processed {num_processed} samples")
    logger.info(f"Output saved to {output_dir}")
    logger.info(f"Report saved to {report_file}")


if __name__ == "__main__":
    main()