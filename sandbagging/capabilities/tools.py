from inspect_ai.tool import tool, Tool


@tool
def web_search():
    """Mock web search tool for testing."""
    async def search(query: str) -> str:
        """
        Search the internet

        Args:
            query: The search query.

        Returns:
            str: The search result.

        """
        return f"Mock search results for: {query}"
    return search


@tool
def image_generation():
    """Mock image generation tool for testing."""
    async def generate_image(prompt: str) -> str:
        """
        Generate an image based on a text prompt

        Args:
            prompt: The text description of the image to generate.

        Returns:
            str: The generated image result.

        """
        return f"Mock image generated for: {prompt}"
    return generate_image


@tool
def code_execution():
    """Mock code execution tool for testing."""
    async def execute_code(code: str) -> str:
        """
        Execute code and return the result

        Args:
            code: The code to execute.

        Returns:
            str: The execution result or output.

        """
        return f"Mock execution result for code: {code[:50]}..."
    return execute_code
