DEFAULT_INTENT_GRAPH_KNOWLEDGE = """\
Given a list of prerequisite questions and their relevant knowledge for the user's main question, when conflicts in meaning arise, prioritize the relationship with the higher weight and the more recent version.

Knowledge sub-queries:

{% for sub_query, data in sub_queries.items() %}

Sub-query: {{ sub_query }}

  - Entities:

{% for entity in data['entities'] %}

    - Name: {{ entity.name }}
    - Description: {{ entity.description }}

{% endfor %}

  - Relationships:

{% for relationship in data['relationships'] %}

    - Description: {{ relationship.rag_description }}
    - Weight: {{ relationship.weight }}

{% endfor %}

{% endfor %}
"""

DEFAULT_NORMAL_GRAPH_KNOWLEDGE = """\
Given a list of relationships of a knowledge graph as follows. When there is a conflict in meaning between knowledge relationships, the relationship with the higher `weight` and newer `last_modified_at` value takes precedence.

---------------------
Entities:

{% for entity in entities %}

- Name: {{ entity.name }}
- Description: {{ entity.description }}

{% endfor %}

---------------------

Knowledge relationships:

{% for relationship in relationships %}

- Description: {{ relationship.rag_description }}
- Weight: {{ relationship.weight }}
- Last Modified At: {{ relationship.last_modified_at }}
- Meta: {{ relationship.meta | tojson(indent=2) }}

{% endfor %}
"""

DEFAULT_CONDENSE_QUESTION_PROMPT = """\
---------------------
The prerequisite questions and their relevant knowledge for the user's main question.
---------------------

{{graph_knowledges}}

---------------------

Task:
Given the conversation between the Human and Assistant, along with the follow-up message from the Human, and the provided prerequisite questions and relevant knowledge, refine the Human’s follow-up message into a standalone, detailed question.

Instructions:
1. Focus on the latest query from the Human, ensuring it is given the most weight.
2. Incorporate Key Information:
  - Use the prerequisite questions and their relevant knowledge to add specific details to the follow-up question.
  - Replace ambiguous terms or references in the follow-up question with precise information from the provided knowledge. Example: Replace “latest version” with the actual version number mentioned in the knowledge.
3. Utilize Conversation Context:
  - Incorporate relevant context and background information from the conversation history to enhance the question's specificity.
4. Optimize for Retrieval:
  - Ensure the refined question emphasizes specific and relevant terms to maximize the effectiveness of a vector search for retrieving precise and comprehensive information.
5. Grounded and Factual:
  - Make sure the refined question is grounded in and directly based on the user's follow-up question and the provided knowledge.
  - Do not introduce information that is not supported by the knowledge or conversation history.

Example:

Chat History:

Human: "I'm interested in the performance improvements in the latest version of TiDB."
Assistant: "TiDB version 8.1 was released recently with significant performance enhancements over version 6.5."

Follow-up Question:

"Can you tell me more about these improvements?"

Prerequisite Questions and Relevant Knowledge:

- Prerequisite Question: What is the latest version of TiDB?
- Relevant Knowledge: The latest version of TiDB is 8.1.

...

Refined Standalone Question:

"Can you provide detailed information about the performance improvements introduced in TiDB version 8.1 compared to version 6.5?"

Your Turn:

Chat history:

{{chat_history}}

---------------------

Followup question:

{{question}}

---------------------

Refined standalone question:
"""


DEFAULT_TEXT_QA_PROMPT = """\
---------------------
Knowledge graph information is below
---------------------

{{graph_knowledges}}

---------------------
Context information is below.
---------------------

<<context_str>>

---------------------

Answer Format:

Use markdown footnote syntax (for example: [^1]) to indicate sources you used.
Each footnote must correspond to a unique source. Do not use the same source for multiple footnotes.

### Examples of Correct Footnote Usage (no the unique sources and diverse sources):
[^1]: [TiDB Overview | PingCAP Docs](https://docs.pingcap.com/tidb/stable/overview)
[^2]: [TiDB Architecture | PingCAP Docs](https://docs.pingcap.com/tidb/stable/architecture)

### Examples of Incorrect Footnote Usage (Avoid duplicating the same source for multiple footnotes):
[^1]: [TiDB Introduction | PingCAP Docs](https://docs.pingcap.com/tidb/v5.4/overview)
[^2]: [TiDB Introduction | PingCAP Docs](https://docs.pingcap.com/tidb/v5.4/overview)
[^3]: [TiDB Introduction | PingCAP Docs](https://docs.pingcap.com/tidb/dev/overview)
[^4]: [TiDB Introduction | PingCAP Docs](https://docs.pingcap.com/tidb/stable/overview)

---------------------

Answer Language:

Respond in the language of the original question; for instance, reply in Chinese if the original question was asked in Chinese and in English if it was asked in English!

---------------------

As a customer support assistant, please do not fabricate any knowledge. If you cannot get knowledge from the context, please just directly state "you do not know", rather than constructing nonexistent and potentially fake information!!!

First, analyze the provided context information without assuming prior knowledge. Identify all relevant aspects of knowledge contained within. Then, from various perspectives and angles, answer questions as thoroughly and comprehensively as possible to better address and resolve the user's issue.

The Original questions is:

{{original_question}}

The Refined Question used to search:
<<query_str>>

Answer:
"""

DEFAULT_REFINE_PROMPT = """\
The Original questions is:

{{original_question}}

Refined Question used to search:
<<query_str>>

---------------------
We have provided an existing answer:
---------------------

<<existing_answer>>

---------------------
We have the opportunity to refine the existing answer (only if needed) with some more knowledge graph and context information below.

---------------------
Knowledge graph information is below
---------------------

{{graph_knowledges}}

---------------------
Context information is below.
---------------------

<<context_msg>>

---------------------
Given the new context, refine the original answer to better answer the query. If the context isn't useful, return the original answer.
Refined Answer:
"""
