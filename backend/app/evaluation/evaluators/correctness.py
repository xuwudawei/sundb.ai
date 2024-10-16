import time
from typing import Optional, Sequence, Mapping
from llama_index.core.evaluation.base import EvaluationResult
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric


max_retries = 3
retry_delay = 2

class CorrectnessEvaluator:
    def __init__(self, model="gpt-4o", threshold=0.7) -> None:
        self._model = model
        self._threshold = threshold
        self._correctness_metric = AnswerRelevancyMetric(
            threshold=self._threshold, model=self._model, include_reason=True
        )
        
        self._prompt_template =  """
You are an expert evaluator specializing in assessing the correctness of answers to multiple-choice questions.

**Objective:**

- Determine whether the response contains the correct option(s) based on the reference answer.

**Instructions:**

1. **Analyze the Question and Options:**
   - Carefully read the question and all available options.
   - Understand the nuances and specifics of each option.

2. **Compare with Reference Answer:**
   - Thoroughly compare the response to the reference answer.
   - Determine if the selected options in the response match the correct options.

3. **Provide a Detailed Evaluation:**
   - **Assign a score between 0.00 and 1.00 (two decimal places), where:**
     - **1.00** indicates the response contains all correct options and no incorrect ones.
     - **0.00** indicates the response is completely incorrect.
   - **Provide a concise explanation (one or two sentences) justifying the score.**

**Constraints:**

- **Do not include any external information or assumptions.**
- **Focus solely on the response and reference answer.**

**Response Format (strictly adhere to this format):**
Score: <score> 
Explanation: <your explanation>
**Question:**

{input}

**Reference Answer:**

{expected_output}

**Response:**

{actual_output}

**Your Evaluation:**

"""

    def evaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        reference: Optional[str] = None,
    ) -> Mapping[str, EvaluationResult]:
        test_case = LLMTestCase(
            input=query,
            actual_output=response.strip(),
            expected_output=reference.strip(),
            retrieval_context=contexts,
        )

        evaluation_results = None
        for attempt in range(max_retries):
            try:
                evaluation_results = evaluate(
                    test_cases=[test_case],
                    metrics=[self._correctness_metric],
                    print_results=False,
                    show_indicator=False,
                    hyperparameters={
                        "model": self._model,
                        "prompt template": self._prompt_template
                    },
                    
                )
                break  # Exit loop if successful
            except ValueError as e:
                print(f"Caught ValueError: {e}")
                print(f"Retrying {attempt + 1}/{max_retries}...")
                time.sleep(retry_delay)

        if not evaluation_results:
            print("No evaluation results were returned.")
            return {}

        

        metrics_results = {}

        # Access test_results from evaluation_results
        if hasattr(evaluation_results, 'test_results'):
            test_results = evaluation_results.test_results
        elif isinstance(evaluation_results, dict) and 'test_results' in evaluation_results:
            test_results = evaluation_results['test_results']
        else:
            print("Unexpected structure of evaluation_results.")
            return {}

        if not test_results:
            print("test_results is None or empty.")
            return {}

        for test_result in test_results:
            for metric_data in test_result.metrics_data:
                metrics_results[metric_data.name] = EvaluationResult(
                    query=query,
                    response=response,
                    contexts=contexts,
                    passing=metric_data.success,
                    score=metric_data.score or 0.0,
                    feedback=metric_data.reason or metric_data.error,
                )

        return metrics_results
