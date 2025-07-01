#!/usr/bin/env python3
# games/alibi/example_usage.py

"""
Example usage of the Alibi Deception Game

This script demonstrates how to:
1. Generate alibi scenarios
2. Create and run alibi investigation tasks
3. Analyze results from interrogations
"""

import asyncio
from inspect_ai import eval

from games.alibi.scenario import generate_alibi_scenarios, generate_specific_incident_scenarios
from games.alibi.tasks import create_alibi_task, create_theft_alibi_task, create_fraud_alibi_task
from games.alibi.models import IncidentType


def main():
    """Main function demonstrating alibi game usage"""
    
    print("🔍 Alibi Deception Game - Example Usage\n")
    
    # 1. Generate scenarios
    print("1. Generating alibi scenarios...")
    mixed_scenarios = generate_alibi_scenarios(num_scenarios=5)
    theft_scenarios = generate_specific_incident_scenarios(IncidentType.THEFT, num_scenarios=3)
    
    print(f"   - Generated {len(mixed_scenarios)} mixed scenarios")
    print(f"   - Generated {len(theft_scenarios)} theft scenarios")
    
    # Display example scenario
    example = mixed_scenarios[0]
    print(f"\n📋 Example Scenario:")
    print(f"   Incident: {example.incident.description}")
    print(f"   Location: {example.incident.location}")
    print(f"   Time: {example.incident.time_frame.date}, {example.incident.time_frame.start_time}-{example.incident.time_frame.end_time}")
    print(f"   Suspect: {example.suspect_profile.name}")
    print(f"   Detective: {example.detective_profile.name}")
    
    # Show suspect's alibi vs reality
    alibi_element = example.suspect_profile.alibi_story[0]
    actual_element = example.suspect_profile.actual_whereabouts[0]
    is_lying = not alibi_element.is_truthful
    
    print(f"\n🎭 Deception Analysis:")
    print(f"   Suspect Claims: {alibi_element.location} - {alibi_element.activity}")
    print(f"   Actually Was: {actual_element.location} - {actual_element.activity}")
    print(f"   Is Lying: {'Yes' if is_lying else 'No'}")
    
    # 2. Create tasks
    print(f"\n2. Creating evaluation tasks...")
    mixed_task = create_alibi_task(mixed_scenarios[:3], "example-mixed-alibi")
    theft_task = create_theft_alibi_task(num_scenarios=2)
    
    print(f"   - Created mixed alibi task with {len(mixed_scenarios[:3])} scenarios")
    print(f"   - Created theft-focused task with 2 scenarios")
    
    # 3. Run evaluation (commented out to avoid actual API calls in example)
    print(f"\n3. Running evaluations...")
    print("   [Evaluation code - uncomment to run with actual model]")
    results = eval(
        tasks=[mixed_task, theft_task],
        model='openai/gpt-4o-mini',
        log_dir='../../logs'
    )
    
    # Example of what results would contain
    print(f"\n📊 Expected Results Structure:")
    print("   - investigation_effectiveness: How well detective caught lies")
    print("   - deception_quality: How convincing suspect's lies were")  
    print("   - alibi_consistency: Timeline and story coherence")
    print("   - lies_detected: Number of lies caught by detective")
    print("   - total_turns: Length of interrogation")
    print("   - detective_confidence: Final assessment of guilt")
    
    print(f"\n✅ Example completed successfully!")
    print("   Uncomment evaluation code and add model credentials to run actual interrogations.")


def analyze_scenario_types():
    """Analyze the distribution of scenario types"""
    print("\n📈 Analyzing Scenario Generation...")
    
    scenarios = generate_alibi_scenarios(num_scenarios=20)
    
    # Count incident types
    incident_counts = {}
    guilty_count = 0
    innocent_count = 0
    
    for scenario in scenarios:
        incident_type = scenario.incident.incident_type.value
        incident_counts[incident_type] = incident_counts.get(incident_type, 0) + 1
        
        # Check if suspect is actually guilty
        if scenario.suspect_profile.alibi_story:
            is_guilty = not scenario.suspect_profile.alibi_story[0].is_truthful
            if is_guilty:
                guilty_count += 1
            else:
                innocent_count += 1
    
    print("Incident Type Distribution:")
    for incident_type, count in incident_counts.items():
        print(f"   {incident_type}: {count}")
    
    print(f"\nGuilt Distribution:")
    print(f"   Guilty Suspects: {guilty_count}")
    print(f"   Innocent Suspects: {innocent_count}")
    print(f"   Guilt Rate: {guilty_count / len(scenarios):.1%}")


if __name__ == "__main__":
    main()
    analyze_scenario_types()