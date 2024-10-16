from deepeval.metrics import ContextualRelevancyMetric
from typing import Optional, Sequence, Mapping
from llama_index.core.evaluation.base import EvaluationResult
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
import time


max_retries = 3
retry_delay = 2

class ContextCorrectnessEvaluator:
    def __init__(self, model="gpt-4o", threshold=0.7) -> None:
        self._model = model
        self._threshold = threshold
        self._context_correctness_metric = ContextualRelevancyMetric(
            threshold=self._threshold, model=self._model, include_reason=True
        )

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
            expected_output=reference.strip() if reference else None,
            retrieval_context=contexts,
        )

        evaluation_results = []
        for attempt in range(max_retries):
            try:
                evaluation_results = evaluate(
                    test_cases=[test_case],
                    metrics=[self._context_correctness_metric],
                    print_results=False,
                    show_indicator=False,
                )
                break  # Exit loop if successful
            except ValueError as e:
                print(f"Caught ValueError: {e}")
                print(f"Retrying {attempt + 1}/{max_retries}...")
                time.sleep(retry_delay)

        if not evaluation_results:
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
