import time
import logging
from typing import Any, Optional, Sequence, List, Mapping
from llama_index.core.evaluation.base import EvaluationResult
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    GEval,
    HallucinationMetric
)

logger = logging.getLogger(__name__)
max_retries = 3
retry_delay = 2


class E2ERagEvaluator:
    def __init__(self, model="gpt-4o-mini", threshold=0.7) -> None:
        self._model = model
        self._threshold = threshold

        self._contextual_precision = ContextualPrecisionMetric(
            threshold=self._threshold, model=self._model, include_reason=True
        )
        self._contextual_recall = ContextualRecallMetric(
            threshold=self._threshold, model=self._model, include_reason=True
        )
        self._contextual_relevancy = ContextualRelevancyMetric(
            threshold=self._threshold, model=self._model, include_reason=True
        )
        self._answer_relevancy = AnswerRelevancyMetric(
            threshold=self._threshold, model=self._model, include_reason=True
        )
        self._faithfulness = FaithfulnessMetric(
            threshold=self._threshold, model=self._model, include_reason=True
        )
        self._correctness_g_eval = GEval(
            threshold=self._threshold, 
            model=self._model,
            name="Correctness",
            criteria="Correctness - determine if the actual output is correct according to the answers in the expected output.",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        )
        self._hallucination = HallucinationMetric(
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
            actual_output=response,
            expected_output=reference,
            retrieval_context=contexts,
            context=contexts,
        )

        evaluation_results = []
        for attempt in range(max_retries):
            try:
                evaluation_results = evaluate(
                    test_cases=[test_case],
                    metrics=[
                        self._correctness_g_eval,
                        self._contextual_precision,
                        self._contextual_recall,
                        self._contextual_relevancy,
                        self._answer_relevancy,
                        self._faithfulness,
                        self._hallucination,
                    ],
                    print_results=True,
                    show_indicator=False,
                    run_async=True,
                )
                break  # Exit loop if successful
            except ValueError as e:
                print(f"Caught ValueError: {e}")
                print(f"Retrying {attempt + 1}/{max_retries}...")
                time.sleep(retry_delay)
            except Exception as e:
                # Handle unexpected exceptions
                print(f"An unexpected error occurred: {e}")
                break  # Exit the retry loop to prevent hanging

        if not evaluation_results:
            return {}
        
        print(f"\n\nEvaluation results: {evaluation_results}\n\n")
        print(f"\n\nType of evaluation_results: {type(evaluation_results)}\n\n")

        metrics_results = {}
        # # Access test_results from evaluation_results
        # if hasattr(evaluation_results, 'test_results'):
        #     test_results = evaluation_results.test_results
        # elif isinstance(evaluation_results, list):
        #     test_results = evaluation_results['test_results']
        # else:
        #     print("Unexpected structure of evaluation_results.")
        #     return {}
        if hasattr(evaluation_results, 'test_results'):
            test_results = evaluation_results.test_results
        elif isinstance(evaluation_results, list):
            test_results = evaluation_results  # Assign the list directly
        else:
            print("Unexpected structure of evaluation_results.")
            return {}


        if not test_results:
            print("test_results is None or empty.")
            return {}

        for test_result in test_results:
            for metric_data in test_result.metrics_metadata:
                metrics_results[metric_data.metric] = EvaluationResult(
                    query=query,
                    response=response,
                    contexts=contexts,
                    passing=metric_data.success,
                    score=metric_data.score or 0.0,
                    feedback=metric_data.reason or metric_data.error,
                )

        return metrics_results

