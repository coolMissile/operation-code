"""LLM API client for code analysis and generation."""

import json
import os
from typing import List, Dict, Optional, Any


class LLMClient:
    """Unified client for LLM APIs (DeepSeek/Claude)."""

    def __init__(self, provider: str = "deepseek", model: str = "deepseek-chat"):
        self.provider = provider
        self.model = model
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = "https://api.deepseek.com"

    def analyze_code(self, code: str, instructions: str) -> str:
        """Send code to LLM for analysis."""
        prompt = f"""{instructions}

Here is the code to analyze:
```python
{code}
```

Provide a structured analysis in JSON format."""
        return self._call_llm(prompt)

    def extract_algorithm(self, code: str) -> Dict[str, Any]:
        """Extract structured algorithm description from code."""
        prompt = f"""You are an expert in combinatorial optimization algorithms.
Analyze the following TSP solver code and extract its core algorithm logic.

Focus on:
1. What representation does it use (permutation, heatmap, etc.)?
2. What search strategy (constructive, iterative improvement, population-based, learning-based)?
3. How does it initialize solutions?
4. What neighborhood/operator does it use (2-opt, swap, insertion, LK-move)?
5. How does it select/survive solutions?
6. What's the core innovation or key mechanism?
7. What's the acceptance criterion?

Return a JSON object with these fields:
- representation: string
- search_type: "constructive" | "iterative_improvement" | "population_based" | "learning_based" | "exact" | "hybrid"
- initialization: string
- neighborhood: string
- selection: string
- acceptance: string
- population_strategy: string
- termination: string
- core_innovation: string (2-3 sentences)
- pseudocode: string (brief pseudocode of main loop)
- key_operators: list of strings

Code:
```python
{code}
```"""
        response = self._call_llm(prompt)
        return self._parse_json(response)

    def generate_candidate(self, design_vector: Dict[str, str],
                          parents_info: List[Dict[str, Any]],
                          tsp_instance_info: Dict[str, Any]) -> str:
        """Generate a candidate solution description based on design vector."""
        prompt = f"""You are an expert algorithm designer. Given the following design
specification and reference algorithms, design a new TSP solving approach.

Design Vector:
{json.dumps(design_vector, indent=2)}

Parent Algorithms:
{json.dumps(parents_info, indent=2)}

Problem Context:
{json.dumps(tsp_instance_info, indent=2)}

Describe the new algorithm in detail, including:
1. How it represents solutions
2. The search/improvement strategy
3. Step-by-step procedure
4. Why this combination might be effective

Then provide a Python implementation skeleton."""
        return self._call_llm(prompt)

    def _call_llm(self, prompt: str) -> str:
        """Make API call to the LLM."""
        import requests
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
        }
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"# LLM call failed: {e}"

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response."""
        import re
        # Try to find JSON block
        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if match:
            text = match.group(1)
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return {"raw": text}
