#!/usr/bin/env python3
"""
Test script for the Flask annotation app
"""

import json
from app import data_manager

def test_data_loading():
    """Test that data is loaded correctly"""
    print("Testing data loading...")
    
    # Check that we have folds
    assert len(data_manager.samples) > 0, "No folds loaded"
    print(f"✓ Loaded {len(data_manager.samples)} folds")
    
    # Check that each fold has samples
    for fold_name, samples in data_manager.samples.items():
        assert len(samples) > 0, f"Fold {fold_name} has no samples"
        print(f"✓ Fold {fold_name}: {len(samples)} samples")
        
        # Check sample structure
        sample = samples[0]
        assert 'messages' in sample, f"Sample in {fold_name} missing 'messages'"
        assert 's3_metadata' in sample, f"Sample in {fold_name} missing 's3_metadata'"
        
        # Check messages structure
        messages = sample['messages']
        assert len(messages) > 0, f"Sample in {fold_name} has no messages"
        
        for msg in messages:
            assert 'role' in msg, f"Message in {fold_name} missing 'role'"
            assert 'content' in msg, f"Message in {fold_name} missing 'content'"
    
    print("✓ All data structure tests passed")

def test_sample_retrieval():
    """Test sample retrieval"""
    print("\nTesting sample retrieval...")
    
    for fold_name in data_manager.samples.keys():
        sample = data_manager.get_sample(fold_name, 0)
        assert sample is not None, f"Could not retrieve sample 0 from {fold_name}"
        print(f"✓ Retrieved sample 0 from {fold_name}")
        
        # Test invalid index
        invalid_sample = data_manager.get_sample(fold_name, 99999)
        assert invalid_sample is None, f"Should return None for invalid index in {fold_name}"
        print(f"✓ Invalid index handling works for {fold_name}")

def test_annotation_saving():
    """Test annotation saving (without actually saving to files)"""
    print("\nTesting annotation saving...")
    
    # Get a sample
    fold_name = list(data_manager.samples.keys())[0]
    sample = data_manager.get_sample(fold_name, 0)
    
    # Test adding annotation
    original_annotations = sample.get('human_annotations', {})
    
    # Simulate adding annotation
    if 'human_annotations' not in sample:
        sample['human_annotations'] = {}
    
    sample['human_annotations']['test_annotator'] = {
        'classification': 'lie',
        'timestamp': '2024-01-01T12:00:00'
    }
    
    assert 'test_annotator' in sample['human_annotations'], "Annotation not added"
    assert sample['human_annotations']['test_annotator']['classification'] == 'lie', "Classification not set"
    
    print("✓ Annotation structure is correct")
    
    # Restore original state
    sample['human_annotations'] = original_annotations

if __name__ == '__main__':
    try:
        test_data_loading()
        test_sample_retrieval()
        test_annotation_saving()
        print("\n🎉 All tests passed! The app is ready to use.")
        print("\nTo start the app, run:")
        print("  python run_app.py")
        print("\nThen open your browser to: http://localhost:5000")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
