import re
import time
from typing import Optional, Sequence, Mapping
from llama_index.core.prompts.base import Prompt

from llama_index.core.evaluation.base import EvaluationResult

class ContextRelevanceEvaluator:
    def __init__(self, llm, threshold=0.7):
        self._llm = llm
        self._threshold = threshold

    def evaluate(
        self,
        query: Optional[str],
        response: Optional[str],
        contexts: Optional[Sequence[str]],
        reference: Optional[str],
    ) -> Mapping[str, EvaluationResult]:
        if query is None or response is None or contexts is None:
            raise ValueError("query, response, and contexts must be provided")

        options = self._extract_options_from_query(query)
        selected_options = [options[letter] for letter in response if letter in options]

        # Compute the expressions outside the f-string
        context_text = "\n".join(contexts)
        selected_answers_text = ', '.join(selected_options)

        prompt_template = Prompt(template="""
        Question:
        {query}

        Context:
        {context_text}

        Selected Answer(s):
        {selected_answers_text}

        Evaluate how relevant the selected answer(s) are to the context provided.

        Provide a score between 0 and 1, where:
        - 1 means the selected answers are highly relevant to the context.
        - 0 means the selected answers are not relevant to the context.
        Also, provide a brief explanation.

        Response format:
        Score: <score>
        Explanation: <your explanation>
        """)

        prompt_args = {
            "query": query,
            "context_text": context_text,
            "selected_answers_text": selected_answers_text
        }

        eval_response = self._llm.predict(prompt_template, **prompt_args).strip()

        # Parse the response
        lines = eval_response.strip().split('\n')
        score_line = next((line for line in lines if line.startswith('Score:')), None)
        explanation_lines = [line for line in lines if line.startswith('Explanation:') or not line.startswith('Score:')]
        if score_line is None:
            raise ValueError("Failed to parse score from LLM response.")
        score = float(score_line.replace('Score:', '').strip())
        explanation = '\n'.join(explanation_lines).replace('Explanation:', '').strip()

        eval_result = EvaluationResult(
            query=query,
            response=response,
            contexts=contexts,
            passing=score >= self._threshold,
            score=score,
            feedback=explanation,
        )

        return {'context_relevance': eval_result}

    def _extract_options_from_query(self, query: str) -> dict:
        pattern = r'([A-D])\.\s*(.+)'
        options = dict(re.findall(pattern, query))
        return options
