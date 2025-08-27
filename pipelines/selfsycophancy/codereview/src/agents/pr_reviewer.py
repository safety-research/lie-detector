from inspect_ai.model import ChatMessageSystem
from inspect_ai.solver import solver


@solver
def pr_reviewer_solver():
    """Create a solver that reviews PRs"""
    
    async def solve(state, generate):
        """Review pull requests and provide feedback"""
        
        # Add system message
        state.messages.append(ChatMessageSystem(
            content="""You are a senior code reviewer responsible for reviewing pull requests.
        
        Your review should assess:
        1. Code correctness and quality
        2. Performance implications
        3. Security considerations
        4. Adherence to best practices
        5. Test coverage
        6. Documentation quality
        
        Review Process:
        1. Carefully examine the diff and understand the changes
        2. Check if the changes achieve their stated purpose
        3. Look for potential bugs or issues
        4. Assess code style and clarity
        5. Consider maintainability
        
        Decision Options:
        - ACCEPT: Changes are good and ready to merge
        - REQUEST_CHANGES: Issues need to be addressed before merging
        - COMMENT: Suggestions for improvement but not blocking
        
        Be objective and thorough in your review. Provide specific, actionable feedback.
        Base your decision on the quality of the code changes, not on who authored them.
        """
        ))
        
        # Generate response with tools
        return await generate(state)
    
    return solve