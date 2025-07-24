# llm_wrapper.py

from ollama import Client

class LLMWrapper:
    def __init__(
        self,
        model_name: str = "llama3.2",
        host: str = "http://localhost:11434",
        verbose: bool = True
    ):
        """
        Initialize an Ollama LLM with customizable settings.

        Args:
            model_name (str): Name of the local Ollama model to use.
            host (str): Address of the Ollama server.
            verbose (bool): Whether to print initialization status.
        """
        self.model_name = model_name
        self.client = Client(host=host)

        if verbose:
            print(f"[LLMWrapper] Initialized Ollama model '{self.model_name}' at {host}")

    def run_prompt(self, prompt: str) -> str:
        """
        Send a prompt to the LLM and return the string response.

        Args:
            prompt (str): The text prompt to send to the model.

        Returns:
            str: The model's textual response.
        """
        response = self.client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]
