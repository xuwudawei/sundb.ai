from litellm import get_supported_openai_params

params = get_supported_openai_params(model="gpt-4o-mini", custom_llm_provider="openai")

print(params)


print("\n\n")

from litellm import supports_response_schema

print(supports_response_schema(model="gpt-4o-mini", custom_llm_provider="openai"))