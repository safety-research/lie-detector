"""Comprehensive test suite for LIAR dataset integration."""

import sys
sys.path.append('/Users/jackhopkins/PycharmProjects/lie-detector')

from pipelines.offpolicy.liar.dataset import (
    liar_dataset,
    LABEL_TO_BINARY,
    LABEL_NAMES,
    create_liar_sample
)


def test_label_mapping():
    """Test that label mappings are correct."""
    print("\n🧪 Testing label mappings...")
    
    # Expected mappings
    expected = {
        0: False,  # pants-fire
        1: False,  # false
        2: False,  # barely-true
        3: True,   # half-true
        4: True,   # mostly-true
        5: True    # true
    }
    
    for label, expected_binary in expected.items():
        actual_binary = LABEL_TO_BINARY[label]
        label_name = LABEL_NAMES[label]
        status = "✅" if actual_binary == expected_binary else "❌"
        print(f"  {status} Label {label} ({label_name}) -> {actual_binary} (expected: {expected_binary})")
    
    print("✅ Label mapping test passed")


def test_dataset_loading():
    """Test dataset loading with various configurations."""
    print("\n🧪 Testing dataset loading...")
    
    # Test different splits
    for split in ["train", "validation", "test"]:
        try:
            dataset = liar_dataset(split=split, limit=10)
            print(f"  ✅ Loaded {split} split: {len(dataset)} samples")
        except Exception as e:
            print(f"  ❌ Failed to load {split} split: {e}")
            return False
    
    # Test balance_labels
    balanced = liar_dataset(split="validation", limit=100, balance_labels=True)
    truth_count = sum(1 for s in balanced if s.target == "truth")
    lie_count = sum(1 for s in balanced if s.target == "lie")
    print(f"  ✅ Balanced dataset: {truth_count} truth, {lie_count} lie")
    
    # Test without balance_labels
    unbalanced = liar_dataset(split="validation", limit=100, balance_labels=False)
    print(f"  ✅ Unbalanced dataset: {len(unbalanced)} samples")
    
    print("✅ Dataset loading test passed")
    return True


def test_sample_format():
    """Test that samples have correct format."""
    print("\n🧪 Testing sample format...")
    
    dataset = liar_dataset(split="validation", limit=5)
    
    for i, sample in enumerate(dataset):
        # Check required fields
        assert hasattr(sample, 'id'), f"Sample {i} missing 'id'"
        assert hasattr(sample, 'input'), f"Sample {i} missing 'input'"
        assert hasattr(sample, 'target'), f"Sample {i} missing 'target'"
        assert hasattr(sample, 'metadata'), f"Sample {i} missing 'metadata'"
        
        # Check ID format
        assert sample.id.startswith(('t_', 'f_')), f"Sample {i} has invalid ID format: {sample.id}"
        
        # Check target values
        assert sample.target in ["truth", "lie"], f"Sample {i} has invalid target: {sample.target}"
        
        # Check metadata
        required_metadata = ['dataset_type', 'dataset_name', 'statement', 'did_lie', 
                           'speaker', 'subject', 'original_label']
        for field in required_metadata:
            assert field in sample.metadata, f"Sample {i} missing metadata field: {field}"
        
        # Check did_lie matches target
        expected_target = "lie" if sample.metadata['did_lie'] else "truth"
        assert sample.target == expected_target, f"Sample {i} target mismatch"
        
        # Check conversation format
        assert len(sample.input) == 3, f"Sample {i} should have 3 messages"
        assert sample.input[0].role == "system"
        assert sample.input[1].role == "user"
        assert sample.input[2].role == "assistant"
        
        print(f"  ✅ Sample {i} format is correct")
    
    print("✅ Sample format test passed")


def test_metadata_preservation():
    """Test that metadata is properly preserved."""
    print("\n🧪 Testing metadata preservation...")
    
    dataset = liar_dataset(split="validation", limit=5, include_metadata=True)
    
    for sample in dataset[:3]:
        print(f"\n  Sample {sample.id}:")
        print(f"    Statement: {sample.metadata['statement'][:50]}...")
        print(f"    Speaker: {sample.metadata['speaker']}")
        print(f"    Subject: {sample.metadata['subject']}")
        print(f"    Original label: {sample.metadata['original_label']}")
        print(f"    Did lie: {sample.metadata['did_lie']}")
        
        # Check if additional metadata is present
        optional_fields = ['job_title', 'state_info', 'party_affiliation']
        present_fields = [f for f in optional_fields if f in sample.metadata]
        if present_fields:
            print(f"    Additional metadata: {', '.join(present_fields)}")
    
    print("\n✅ Metadata preservation test passed")


def test_conversation_content():
    """Test conversation content formatting."""
    print("\n🧪 Testing conversation content...")
    
    dataset = liar_dataset(split="validation", limit=2)
    
    for i, sample in enumerate(dataset):
        print(f"\n  Sample {i} conversation:")
        system_msg = sample.input[0].content
        user_msg = sample.input[1].content
        assistant_msg = sample.input[2].content
        
        # Check system message
        assert "helpful AI assistant" in system_msg
        print(f"    ✅ System message is correct")
        
        # Check user message contains statement
        assert sample.metadata['statement'] in user_msg
        assert "Is this statement true?" in user_msg
        assert "A) True" in user_msg and "B) False" in user_msg
        print(f"    ✅ User message contains statement and options")
        
        # Check assistant response
        assert assistant_msg in ["A) True", "B) False"]
        print(f"    ✅ Assistant response is valid: {assistant_msg}")
        
        # Check context formatting
        if sample.metadata['speaker'] != 'Unknown':
            assert sample.metadata['speaker'] in user_msg
            print(f"    ✅ Speaker included in context")
    
    print("\n✅ Conversation content test passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("LIAR Dataset Comprehensive Test Suite")
    print("=" * 60)
    
    tests = [
        test_label_mapping,
        test_dataset_loading,
        test_sample_format,
        test_metadata_preservation,
        test_conversation_content
    ]
    
    failed = 0
    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"\n❌ {test_func.__name__} failed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    if failed == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ {failed} tests failed")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()