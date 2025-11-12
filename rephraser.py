from textwrap import dedent
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools

# Create a Rephraser Agent that uses DuckDuckGo tool for corrections
rephraser_agent = Agent(
    model=Gemini(id="gemini-2.0-flash-001"),
    instructions=dedent("""\
        You are a highly skilled rephraser and corrector. 
        Your task is to rephrase input text to improve clarity, grammar, and flow,
        while preserving the original meaning fully.

        If the text contains errors, misspellings, wrong terminology or ambiguous terms, 
        use the DuckDuckGo search tool to find the best possible correction 
        or clarification. Use the search results to fix the text accordingly.

        Maintain context and avoid changing any factual meaning.
        Provide your response as a polished and corrected version of the input text.
        """),
    tools=[DuckDuckGoTools()],
    markdown=True,
)

def rephrase_text_with_correction(text: str):
    """Function to pass text to the rephraser agent and get polished text back."""
    rephraser_agent.print_response(text)


