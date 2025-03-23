import os
import datetime
import hashlib
from typing import Any, Literal
import logging

import dspy
import requests
from dspy.clients.lm import LM
from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.gemini import Gemini
from llama_index.llms.bedrock import Bedrock
from llama_index.llms.ollama import Ollama
from app.rag.llms.anthropic_vertex import AnthropicVertex


def get_dspy_lm_by_llama_llm(llama_llm: BaseLLM) -> dspy.LM:
    """
    Get the dspy LM by the llama LLM.

    In this project, we use both llama-index and dspy, both of them have their own LLM implementation.
    This function can help us reduce the complexity of the code by converting the llama LLM to the dspy LLM easily.
    """
    if type(llama_llm) is OpenAI:
        return dspy.OpenAI(
            # model=llama_llm.model,
            'openai/'+llama_llm.model,
            max_tokens=llama_llm.max_tokens or 4096,
            api_key=llama_llm.api_key,
            api_base=enforce_trailing_slash(llama_llm.api_base),
        )
    elif type(llama_llm) is OpenAILike:
        return dspy.LM(
            'openai/'+llama_llm.model,
            max_tokens=llama_llm.max_tokens or 6096,
            api_key=llama_llm.api_key,
            temperature=0.0,
            api_base=enforce_trailing_slash(llama_llm.api_base),
            model_type="chat" if llama_llm.is_chat_model else "text",
        )
    elif type(llama_llm) is Gemini:
        # Don't need to configure the api_key again,
        # it has already been configured to `genai` by the llama_llm.
        return dspy.Google(
            model=llama_llm.model.split("models/")[1],
            max_output_tokens=llama_llm.max_tokens or 7900,
        )
    elif type(llama_llm) is Bedrock:
        # Notice: dspy.Bedrock currently does not support configuring access keys through parameters.
        # Using environment variables for configuration risks contaminating global variables.
        os.environ["AWS_ACCESS_KEY_ID"] = llama_llm.aws_access_key_id
        os.environ["AWS_SECRET_ACCESS_KEY"] = llama_llm.aws_secret_access_key
        bedrock = dspy.Bedrock(region_name=llama_llm.region_name)
        if llama_llm.model.startswith("anthropic"):
            return dspy.AWSAnthropic(
                bedrock, model=llama_llm.model, max_new_tokens=llama_llm.max_tokens
            )
        elif llama_llm.model.startswith("meta"):
            return dspy.AWSMeta(
                bedrock, model=llama_llm.model, max_new_tokens=llama_llm.max_tokens
            )
        elif llama_llm.model.startswith("mistral"):
            return dspy.AWSMistral(
                bedrock, model=llama_llm.model, max_new_tokens=llama_llm.max_tokens
            )
        elif llama_llm.model.startswith("amazon"):
            return dspy.AWSModel(
                bedrock, model=llama_llm.model, max_new_tokens=llama_llm.max_tokens
            )
        else:
            raise ValueError(
                "Bedrock model " + llama_llm.model + " is not supported by dspy."
            )
    elif type(llama_llm) is AnthropicVertex:
        raise ValueError("AnthropicVertex is not supported by dspy.")
    elif type(llama_llm) is Ollama:
        return dspy.LM(
            'ollama/'+llama_llm.model,
            base_url=llama_llm.base_url,
            timeout_s=llama_llm.request_timeout,
            temperature=llama_llm.temperature,
            max_tokens=llama_llm.context_window,
            num_ctx=llama_llm.context_window,
        )
    else:
        raise ValueError(f"Got unknown LLM provider: {llama_llm.__class__.__name__}")


def enforce_trailing_slash(url: str):
    if url.endswith("/"):
        return url
    return url + "/"


###################### Dspy Ollama Local #########################################
##################################################################################
# Copy from dspy.OllamaLocal but add `format = json` when sending request to ollama


def post_request_metadata(model_name, prompt):
    """Creates a serialized request object for the Ollama API."""
    timestamp = datetime.datetime.now().timestamp()
    id_string = str(timestamp) + model_name + prompt
    hashlib.sha1().update(id_string.encode("utf-8"))
    id_hash = hashlib.sha1().hexdigest()
    return {
        "id": f"chatcmpl-{id_hash}",
        "object": "chat.completion",
        "created": int(timestamp),
        "model": model_name,
    }


class DspyOllamaLocal(LM):
    """Wrapper around a locally hosted Ollama model (API: https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values and https://github.com/jmorganca/ollama/blob/main/docs/api.md#generate-a-completion).
    Returns dictionary info in the OpenAI API style (https://platform.openai.com/docs/api-reference/chat/object).

    Args:
        model (str, optional): Name of Ollama model. Defaults to "llama2".
        model_type (Literal["chat", "text"], optional): The type of model that was specified. Mainly to decide the optimal prompting strategy. Defaults to "text".
        base_url (str):  Protocol, host name, and port to the served ollama model. Defaults to "http://localhost:11434" as in ollama docs.
        timeout_s (float): Timeout period (in seconds) for the post request to llm.
        **kwargs: Additional arguments to pass to the API.
    """

    def __init__(
        self,
        model: str = "llama2",
        model_type: Literal["chat", "text"] = "text",
        base_url: str = "http://10.0.0.107:11434",
        timeout_s: float = 120,
        temperature: float = 0.0,
        max_tokens: int = 150,
        top_p: int = 1,
        top_k: int = 20,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        n: int = 1,
        num_ctx: int = 1024,
        cache: bool = True,
        cache_in_memory: bool = True,
        callbacks = None,
        num_retries: int = 8,
        **kwargs,
    ):
        super().__init__(model, model_type=model_type, temperature=temperature, max_tokens=max_tokens, 
                        cache=cache, cache_in_memory=cache_in_memory, callbacks=callbacks, 
                        num_retries=num_retries, **kwargs)

        self.provider = "ollama"
        self.base_url = base_url
        self.model_name = model
        self.timeout_s = timeout_s

        self.kwargs.update({
            "top_p": top_p,
            "top_k": top_k,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "n": n,
            "num_ctx": num_ctx,
            **kwargs,
        })

        # Ollama uses num_predict instead of max_tokens
        self.kwargs["num_predict"] = self.kwargs["max_tokens"]

        self.history: list[dict[str, Any]] = []
        self.version = kwargs["version"] if "version" in kwargs else ""

        # Ollama occasionally does not send `prompt_eval_count` in response body.
        # https://github.com/stanfordnlp/dspy/issues/293
        self._prev_prompt_eval_count = 0

    def basic_request(self, prompt: str, **kwargs):
        raw_kwargs = kwargs

        kwargs = {**self.kwargs, **kwargs}

        request_info = post_request_metadata(self.model_name, prompt)
        request_info["choices"] = []
        settings_dict = {
            "model": self.model_name,
            "options": {
                k: v for k, v in kwargs.items() if k not in ["n", "max_tokens"]
            },
            "stream": False,
        }
        if self.model_type == "chat":
            settings_dict["messages"] = [{"role": "user", "content": prompt}]
        else:
            settings_dict["prompt"] = prompt
        settings_dict["format"] = "json"

        urlstr = (
            f"{self.base_url}/api/chat"
            if self.model_type == "chat"
            else f"{self.base_url}/api/generate"
        )
        tot_eval_tokens = 0
        for i in range(kwargs["n"]):
            response = requests.post(urlstr, json=settings_dict, timeout=self.timeout_s)

            # Check if the request was successful (HTTP status code 200)
            if response.status_code != 200:
                # If the request was not successful, print an error message
                print(f"Error: CODE {response.status_code} - {response.text}")

            response_json = response.json()

            text = (
                response_json.get("message").get("content")
                if self.model_type == "chat"
                else response_json.get("response")
            )
            request_info["choices"].append(
                {
                    "index": i,
                    "message": {
                        "role": "assistant",
                        "content": "".join(text),
                    },
                    "finish_reason": "stop",
                },
            )
            tot_eval_tokens += response_json.get("eval_count", 0)
        request_info["additional_kwargs"] = {
            k: v for k, v in response_json.items() if k not in ["response"]
        }

        request_info["usage"] = {
            "prompt_tokens": response_json.get(
                "prompt_eval_count", self._prev_prompt_eval_count
            ),
            "completion_tokens": tot_eval_tokens,
            "total_tokens": response_json.get(
                "prompt_eval_count", self._prev_prompt_eval_count
            )
            + tot_eval_tokens,
        }

        history = {
            "prompt": prompt,
            "response": request_info,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)

        return request_info

    def forward(self, prompt=None, messages=None, **kwargs):
        """Wrapper for requesting completions from the Ollama model."""
        if "model_type" in kwargs:
            del kwargs["model_type"]
            
        # Build the request
        messages = messages or [{"role": "user", "content": prompt}]
        kwargs = {**self.kwargs, **kwargs}
        
        return self.basic_request(prompt, **kwargs)

    def _get_choice_text(self, choice: dict[str, Any]) -> str:
        return choice["message"]["content"]

    def __call__(
        self,
        prompt: str = None,
        messages = None,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Retrieves completions from Ollama.

        Args:
            prompt (str, optional): prompt to send to Ollama
            messages (list, optional): messages to send to Ollama in chat format
            only_completed (bool, optional): return only completed responses and ignores completion due to length. Defaults to True.
            return_sorted (bool, optional): sort the completion choices using the returned probabilities. Defaults to False.

        Returns:
            list[dict[str, Any]]: list of completion choices
        """

        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        response = self.forward(prompt=prompt, messages=messages, **kwargs)

        choices = response["choices"]

        completed_choices = [c for c in choices if c["finish_reason"] != "length"]

        if only_completed and len(completed_choices):
            choices = completed_choices

        completions = [self._get_choice_text(c) for c in choices]

        return completions
