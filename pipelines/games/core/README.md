# Common Game Infrastructure

This directory contains shared logic for all deception games in the lie-detector project.

## Important Constraint: Tanuki and Class Inheritance

**Tanuki-decorated functions require flat class definitions without inheritance.** This means:

1. Classes used as parameters or return types for `@tanuki.patch` functions CANNOT inherit from base classes
2. All fields must be explicitly defined in each class
3. We use template definitions and composition patterns instead of inheritance

## Structure

### `/base` - Base Classes and Templates
- **models.py**: Base classes that CAN be inherited (for non-Tanuki models)
- **model_templates.py**: Field definitions to copy into Tanuki-compatible models
- **agents.py**: Base agent framework (uses composition, not inheritance)
- **solvers.py**: Base solver framework
- **scorers.py**: Base scorer framework  
- **tasks.py**: Base task creation utilities

### `/utils` - Shared Utilities
- **conversation.py**: Conversation management utilities
- **state_management.py**: State store utilities
- **evaluation.py**: Common evaluation helpers (parse_response_to_json, etc.)

### `/scenario` - Scenario Generation
- **base_generator.py**: Common scenario generation patterns
- **persona_utils.py**: Shared persona generation utilities

## Usage Patterns

### For Tanuki-Compatible Models

Instead of:
```python
class PersonaProfile(BasePersonaProfile):  # ❌ Won't work with Tanuki
    extra_field: str
```

Do:
```python
class PersonaProfile(BaseModel):  # ✅ Flat definition
    name: str
    background: str
    personality_traits: List[str]
    extra_field: str
```

### For Non-Tanuki Models

These CAN use inheritance:
```python
class GameState(BaseGameState):  # ✅ OK for non-Tanuki models
    custom_field: str
```

### Sharing Logic via Composition

Use utility classes and functions instead of inheritance:

```python
# In your game's agent
from games.core.utils.conversation import ConversationManager


class MyAgent:
    def add_turn(self, message: str):
        ConversationManager.add_turn_to_conversation(
            self.game_state.conversation,
            "blue",
            message
        )
```

## Benefits Despite Constraints

1. **Consistent Patterns**: All games follow similar structures
2. **Shared Utilities**: Common functionality without inheritance
3. **Template Reference**: Clear field definitions to copy
4. **Reduced Duplication**: Scorers, solvers, and tasks still share base logic
5. **Better Testing**: Shared test utilities and patterns