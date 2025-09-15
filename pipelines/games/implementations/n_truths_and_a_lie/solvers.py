# games/implementations/n_truths_and_a_lie/solvers.py

from typing import Dict
import random

from inspect_ai.agent import AgentState
from inspect_ai.solver import TaskState, Generate, solver
from inspect_ai.util import store_as

from pipelines.games.core.base.solvers import BaseGameSolver, create_base_solver
from pipelines.games.implementations.n_truths_and_a_lie.agents.blue import blue_agent
from pipelines.games.implementations.n_truths_and_a_lie.agents.red import red_agent
from pipelines.games.implementations.n_truths_and_a_lie.models import (
    GameConfig, GameState, BlueAgentState, RedAgentState,
    PersonaProfile, FactSet, Fact, FactType
)

# Import tanuki functions
from pipelines.games.implementations.n_truths_and_a_lie.scenario.examples import (
    generate_themed_persona, generate_truth_fact, generate_lie_fact,
    generate_supporting_details
)


class NTruthsSolver(BaseGameSolver):
    """Solver for N-Truths-and-a-Lie game with lazy generation"""

    def get_config_class(self):
        return GameConfig

    def get_game_state_class(self):
        return GameState

    def create_agents(self, config: GameConfig):
        return {
            "blue": blue_agent(config),
            "red": red_agent(config)
        }

    def get_opening_message(self, config: GameConfig) -> str:
        # N-Truths game doesn't use a pre-set opening message
        return ""

    def should_continue_game(self, game_state: GameState, config: GameConfig) -> bool:
        return not game_state.game_complete and len(game_state.interrogation) < config.interrogation_turns_limit

    def generate_scenario_details(self, base_config: dict) -> GameConfig:
        """Generate the full scenario details lazily from base configuration"""

        # Extract parameters
        theme = base_config.get('theme', 'professional_background')
        truths_per_game = base_config.get('truths_per_game', 3)
        lies_per_game = base_config.get('lies_per_game', 1)

        params = base_config.get('generation_params', {})
        expertise_level = params.get('expertise_level', 'intermediate')
        authenticity = params.get('authenticity', 'very_realistic')

        print(f"[Solver] Generating N-Truths scenario for theme: {theme}")

        # Generate persona
        persona_data = generate_themed_persona(theme, expertise_level, authenticity)

        persona = PersonaProfile(
            name=persona_data.name,
            background=persona_data.background,
            personality_traits=persona_data.personality_traits,
            communication_style=persona_data.communication_style
        )

        # Define categories based on theme
        theme_categories = {
            'professional_background': ["education", "work", "achievements", "skills"],
            'travel_experiences': ["destinations", "adventures", "cultural_experiences", "mishaps"],
            'personal_quirks': ["habits", "fears", "collections", "abilities"],
            'family_history': ["ancestors", "family_traditions", "relatives", "heritage"],
            'hidden_talents': ["artistic", "athletic", "intellectual", "unusual"]
        }

        categories = theme_categories.get(theme, ["general"])
        used_categories = set()
        facts = []

        # Generate truths
        for _ in range(truths_per_game):
            available_categories = [c for c in categories if c not in used_categories]
            if not available_categories:
                available_categories = categories  # Reset if we've used all

            category = random.choice(available_categories)
            used_categories.add(category)

            difficulty = random.choice(['easy', 'medium', 'hard'])
            truth_data = generate_truth_fact(theme, category, persona_data.background, difficulty)

            # Generate additional supporting details
            extra_details = generate_supporting_details(truth_data.content, "moderate")
            all_details = truth_data.supporting_details + extra_details[:2]  # Limit total details

            truth_fact = Fact(
                content=truth_data.content,
                fact_type=FactType.TRUTH,
                category=truth_data.category,
                supporting_details=all_details
            )
            facts.append(truth_fact)

        # Generate lies
        for _ in range(lies_per_game):
            available_categories = [c for c in categories if c not in used_categories]
            if not available_categories:
                available_categories = categories

            category = random.choice(available_categories)
            used_categories.add(category)

            lie_type = random.choice(['impossible', 'exaggerated', 'timing_impossible', 'inconsistent'])
            lie_data = generate_lie_fact(theme, category, persona_data.background, lie_type)

            lie_fact = Fact(
                content=lie_data.content,
                fact_type=FactType.LIE,
                category=lie_data.category,
                contradicting_evidence=lie_data.why_implausible
            )
            facts.append(lie_fact)

        # Shuffle facts
        random.shuffle(facts)

        # Create fact set
        fact_set = FactSet(
            facts=facts,
            num_truths=truths_per_game,
            num_lies=lies_per_game,
            theme=theme
        )

        # Create game config
        config = GameConfig(
            game_id=base_config.get('game_id', 'generated_ntal'),
            blue_persona=persona,
            fact_set=fact_set,
            interrogation_turns_limit=base_config.get('interrogation_turns_limit', 15),
            allow_followup_questions=True,
            game_context=base_config.get('game_context', {
                "setting": "casual conversation",
                "mood": "engaging and curious",
                "purpose": "building connections through storytelling"
            })
        )

        return config

    async def solve(self, state: TaskState, generate: Generate) -> TaskState:
        # Get game configuration from metadata
        base_config = state.metadata.get("game_config", state.metadata.get("config", {}))

        # Check if we have a full config or need to generate
        if 'blue_persona' not in base_config:
            # Generate full scenario details lazily
            print("[Solver] Generating N-Truths scenario details on-demand...")
            config = self.generate_scenario_details(base_config)
            # Update metadata with full config
            state.metadata["game_config"] = config.model_dump()
        else:
            # Use existing full config
            config = GameConfig(**base_config)

        # Initialize agents
        blue = blue_agent(config)
        red = red_agent(config)

        # Initialize game state
        game_state = store_as(GameState)
        game_state.config = config.model_dump()

        # Initialize agent states
        agent_states = {
            "blue": AgentState(messages=[]),
            "red": AgentState(messages=[])
        }

        # Phase 1: Blue presents facts (opening statement)
        game_state.current_phase = "opening"
        game_state.active_speaker = "blue"

        agent_states["blue"] = await blue(agent_states["blue"])

        # Phase 2: Red interrogates Blue
        while not game_state.game_complete:
            if game_state.active_speaker == "red":
                agent_states["red"] = await red(agent_states["red"])
            else:
                agent_states["blue"] = await blue(agent_states["blue"])

            # Check if we've hit the turn limit
            if len(game_state.interrogation) >= config.interrogation_turns_limit:
                game_state.game_complete = True
                game_state.current_phase = "complete"

        # Store agent traces
        self.store_agent_traces(state, agent_states, game_state)

        # Store final states for analysis
        blue_state = store_as(BlueAgentState, instance="blue")
        red_state = store_as(RedAgentState, instance="red")

        # Prepare metadata for scoring
        state.metadata["game_summary"] = {
            "total_turns": len(game_state.interrogation),
            "opening_statement": game_state.opening_statement,
            "lies_told": len(blue_state.lies_told),
            "facts_explored": len(red_state.facts_explored),
            "questions_asked": len(red_state.questions_asked),
            "lie_index": red_state.known_lie_index,
            "config": config.model_dump()
        }

        return state


# Create the solver
def n_truths_solver():
    return create_base_solver(NTruthsSolver)