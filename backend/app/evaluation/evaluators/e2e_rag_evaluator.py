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
from deepeval.metrics.ragas import RagasMetric
from deepeval.metrics.ragas import RAGASAnswerRelevancyMetric
from deepeval.metrics.ragas import RAGASFaithfulnessMetric
from deepeval.metrics.ragas import RAGASContextualRecallMetric
from deepeval.metrics.ragas import RAGASContextualPrecisionMetric


logger = logging.getLogger(__name__)
max_retries = 3
retry_delay = 2


class E2ERagEvaluator:
    def __init__(self, model, threshold=0.5) -> None:
        self._model = model
        self._threshold = threshold
        self._contextual_precision = ContextualPrecisionMetric(
            threshold=self._threshold, model=self._model, include_reason=True
        )
        self._contextual_recall = ContextualRecallMetric(
            threshold=self._threshold, model=self._model, include_reason=True
        )
        self._contextual_relevancy = ContextualRelevancyMetric(
            threshold=0.4, model=self._model, include_reason=True, verbose_mode=True
        )
        self._answer_relevancy = AnswerRelevancyMetric(
            threshold=self._threshold, model=self._model, include_reason=True
        )
        self._faithfulness = FaithfulnessMetric(
            threshold=self._threshold, model=self._model, include_reason=True
        )
        self._correctness_g_eval = GEval(
            threshold=0.7, 
            model=self._model,
            name="Correctness",
            criteria="When evaluating the actual output against the expected output, focus solely on the correctness and accuracy of the information provided. Do not consider factors such as the level of detail, brevity, focus, or whether the output directly addresses specific aspects. Ignore any issues related to exceeding concise requirements, divergence in terms of brevity and focus, or the inclusion of detailed explanations. Jusr whether the actual output provides the correct answer as per the expected output, focusing solely on accuracy without regard for phrasing, formatting, structure, or emphasis on specific details.",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            verbose_mode=True,
            evaluation_steps=[
                "Compare the actual output to the expected output, focusing solely on the correctness and accuracy of the information provided.",
                "Identify any discrepancies in factual content between the two outputs, disregarding differences in phrasing, formatting, or structure.",
                "Determine if the actual output accurately conveys the same information as the expected output, without considering factors like level of detail, brevity, or emphasis on specific aspects.",
                "Vague languages are OK",
                "Record the evaluation based only on the accuracy of the information, ignoring all other factors such as style, focus, or additional explanations.",
            ],
        )
        self._hallucination = HallucinationMetric(
            threshold=self._threshold, model=self._model, include_reason=True
        )
        self._ragas_metric = RagasMetric(
            threshold=self._threshold, model=self._model
        )
        self._ragas_answer_relevancy = RAGASAnswerRelevancyMetric(
            threshold=self._threshold, model=self._model
        )
        self._ragas_faithfulness = RAGASFaithfulnessMetric(
            threshold=self._threshold, model=self._model
        )
        self._ragas_contextual_recall = RAGASContextualRecallMetric(
            threshold=self._threshold, model=self._model
        )
        self._ragas_contextual_precision = RAGASContextualPrecisionMetric(
            threshold=self._threshold, model=self._model
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
                        # self._contextual_relevancy,
                        self._answer_relevancy,
                        self._faithfulness,
                        # self._hallucination,
                        self._ragas_metric,
                        self._ragas_answer_relevancy,
                        self._ragas_faithfulness,
                        self._ragas_contextual_recall,
                        self._ragas_contextual_precision,
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

        # for test_result in test_results:
        #     for metric_data in test_result.metrics_metadata:
        #         metrics_results[metric_data.metric] = EvaluationResult(
        #             query=query,
        #             response=response,
        #             contexts=contexts,
        #             passing=metric_data.success,
        #             score=metric_data.score or 0.0,
        #             feedback=metric_data.reason or metric_data.error,
        #         )

        # return metrics_results
        for test_result in test_results:
            # Dynamically find the attribute that contains metrics data
            metrics_data_list = None
            for attr_name in ['metrics_metadata', 'metrics_data', 'metrics']:
                if hasattr(test_result, attr_name):
                    metrics_data_list = getattr(test_result, attr_name)
                    break  # Exit the loop once we've found the attribute
            if metrics_data_list is None:
                print("No metrics data found in test_result.")
                continue  # Skip this test_result if no metrics data is found

            for metric_data in metrics_data_list:
                # Ensure all necessary attributes are present
                metric_name = getattr(metric_data, 'name', None) or getattr(metric_data, 'metric', None)
                if metric_name is None:
                    print("Metric data does not have a 'name' or 'metric' attribute.")
                    continue  # Skip this metric_data if name is missing

                passing = getattr(metric_data, 'success', None)
                score = getattr(metric_data, 'score', 0.0) or 0.0
                feedback = getattr(metric_data, 'reason', None) or getattr(metric_data, 'error', None)

                metrics_results[metric_name] = EvaluationResult(
                    query=query,
                    response=response,
                    contexts=contexts,
                    passing=passing,
                    score=score,
                    feedback=feedback,
                )

        return metrics_results

