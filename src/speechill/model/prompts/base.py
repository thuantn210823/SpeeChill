"""
Prompt Module for Turn-Taking Detection

Handles prompt templating, random selection for diversity,
and prompt embedding for LLM input.
"""

import random
import yaml
from typing import List, Dict, Optional
from pathlib import Path

class PromptConfig:
    """Base configuration for prompt modules."""

    def __init__(
        self,
        prompts: Dict[str, List[str]],
        random_selection: bool = True,
        add_special_tokens: bool = True,
        prompts_file: Optional[str] = None
    ):
        self.prompts = prompts
        self.random_selection = random_selection
        self.add_special_tokens = add_special_tokens
        self.prompts_file = prompts_file


class BasePromptModule:
    """Base class for prompt modules."""

    def __init__(self,         
                 prompts: Dict[str, List[str]] = {},
                 random_selection: bool = True,
                 add_special_tokens: bool = True,
                 prompts_file: Optional[str] = None, 
                 tokenizer=None):
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.random_selection = random_selection
        self.add_special_tokens = add_special_tokens

        # Load prompts from file if specified
        if prompts_file:
            self.load_prompts_from_file(prompts_file)

    def load_prompts_from_file(self, prompts_file: str):
        """Load prompts from YAML file."""
        with open(prompts_file, 'r', encoding='utf-8') as f:
            loaded_prompts = yaml.safe_load(f)
            self.prompts.update(loaded_prompts)

    def get_prompt(self, task_name: str) -> str:
        """Get a prompt for the given task."""
        if task_name not in self.prompts:
            raise ValueError(f"Task '{task_name}' not configured in prompts")

        prompt_list = self.prompts[task_name]
        if self.random_selection:
            return random.choice(prompt_list)
        else:
            return prompt_list[0]

    def embed_prompt(self, prompt: str, tokenizer=None):
        """Tokenize and embed a prompt string."""
        tok = tokenizer or self.tokenizer
        if tok is None:
            raise ValueError("No tokenizer provided")
        if isinstance(prompt, str):
            return tok([prompt], return_tensors='pt', padding = True)
        elif isinstance(prompt, list):
            return tok(prompt, return_tensors="pt", padding = True)

    def get_prompt(self, task_name: str) -> str:
        """Get a prompt for the given task."""
        if task_name not in self.prompts:
            raise ValueError(f"Task '{task_name}' not configured in prompts")

        prompt_list = self.prompts[task_name]
        if self.random_selection:
            return random.choice(prompt_list)
        else:
            return prompt_list[0]