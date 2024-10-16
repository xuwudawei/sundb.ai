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
        
        You are an expert evaluator specializing in assessing the relevance of selected answers to the provided context.

        **Objective:**

        - Evaluate how accurately and thoroughly the selected answer(s) address the question, based on the given context.

        **Instructions:**

        1. **Understand the Context:**
        - Carefully read and comprehend the context provided.
        - Identify key information, facts, and details relevant to the question.

        2. **Analyze the Selected Answers:**
        - Review each selected answer option.
        - Determine how each option relates to the context.

        3. **Evaluate Relevance:**
        - Assess whether each selected option is supported by the context.
        - Consider both direct and indirect references.

        4. **Provide a Detailed Evaluation:**
        - **Assign a score between 0.00 and 1.00 (two decimal places), where:**
            - **1.00** indicates the selected answers are fully relevant and supported by the context.
            - **0.00** indicates the selected answers are not relevant or contradict the context.
        - **Provide a concise explanation (one or two sentences) justifying the score.**

        **Constraints:**

        - **Base your evaluation solely on the provided context and selected answers.**
        - **Do not include any external information or assumptions.**
        - **Do not mention irrelevant options.**

        **Response Format (strictly adhere to this format):**
        Score: <score> 
        Explanation: <your explanation>


        **Question:**

        {query}

        **Context:**

        {context_text}

        **Selected Answer(s):**

        {selected_answers_text}

        **Your Evaluation:**

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
