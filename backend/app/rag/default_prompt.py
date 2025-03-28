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

DEFAULT_CLARIFYING_QUESTION_PROMPT = """\
---------------------
The prerequisite questions and their relevant knowledge for the user's main question.
---------------------

{{graph_knowledges}}

---------------------

Task:
Given the conversation between the user and ASSISTANT, along with the follow-up message from the user, and the provided prerequisite questions and relevant knowledge, determine if the user's question is clear and specific enough for a confident response. If the question lacks necessary details or context, identify the specific ambiguities and generate a clarifying question to address them.

Instructions:
1. Assess Information Sufficiency:
   - Evaluate if the user’s question provides enough detail to generate a precise answer based on the prerequisite questions, relevant knowledge, and conversation history.
   - If the user's question is too vague or lacks key information, identify what additional information would be necessary for clarity.

2. Generate a Clarifying Question:
   - If the question is clear and answerable, return "False" and leave the clarifying question empty ("").
   - If clarification is needed, return "True" and generate a specific question to ask the user, directly addressing the information gap. Avoid general questions; focus on the specific details required for an accurate answer.

3. Use the same language to ask the clarifying question as the user's original question.
   
Example 1:

user: "Does Sundb support foreign keys?"
Relevant Knowledge: Sundb supports foreign keys starting from version 6.6.0.

Response:

- Clarity Needed: True
- Clarifying Question: "Which version of Sundb are you using?"

Example 2:

user: "Does Sundb support nested transaction?"
Relevant Knowledge: Sundb supports nested transaction starting from version 6.2.0.

Response:

- Clarity Needed: True
- Clarifying Question: "Which version of Sundb are you using?"

Your Turn:

Chat history:

{{chat_history}}

---------------------

Follow-up question:

{{question}}

Response:
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
6. Give the language hint for the answer:
  - Add a hint after the question like "(Answer language: English)", or "(Answer language: Chinese)", etc.
  - This language hint should be exactly same with the language of the original question.
  - If the original question has part of other language aside from English, please use the language of another language rather than English. Example: "sundb 安装慢会是哪些原因", it should be Chinese.

Example:

Chat History:

Human: "I'm interested in the performance improvements in the latest version of SunDB."
Assistant: "SunDB version 8.1 was released recently with significant performance enhancements over version 6.5."

Follow-up Question:

"Can you tell me more about these improvements?"

Prerequisite Questions and Relevant Knowledge:

- Prerequisite Question: What is the latest version of SunDB?
- Relevant Knowledge: The latest version of SunDB is 8.1.

...

Refined Standalone Question:

"Can you provide detailed information about the performance improvements introduced in SunDB version 8.1 compared to version 6.5? (Answer language: English)"

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
Current Date: {{current_date}}
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
[^1]: [SunDB Overview | Docs](https://github.com/xuwudawei/sundb.ai)
[^2]: [SunDB Architecture | Docs](https://github.com/xuwudawei/sundb.ai)

### Examples of Incorrect Footnote Usage (Avoid duplicating the same source for multiple footnotes):
[^1]: [SunDB Introduction | Docs](https://github.com/xuwudawei/sundb.ai)
[^2]: [SunDB Introduction | Docs](https://github.com/xuwudawei/sundb.ai)
[^3]: [SunDB Introduction | Docs](https://github.com/xuwudawei/sundb.ai)
[^4]: [SunDB Introduction | Docs](https://github.com/xuwudawei/sundb.ai)

---------------------

Answer Language:

Follow the language of the language hint after the Refined Question.
If the language hint is not provided, use the language that the original questions used.

---------------------

As a sundb database developer support assistant, please do not fabricate any knowledge. If you cannot get knowledge from the context, please just directly state "you do not know", rather than constructing nonexistent and potentially fake information!!!

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
And the answer should use the same language with the question. If the answer has different language with the original question, please translate it to the same language with the question.

Refined Answer:
"""

DEFAULT_FURTHER_QUESTIONS_PROMPT = """\
The chat message content is:

{{chat_message_content}}

---------------------
Task:
Based on the provided chat message, generate 3–5 follow-up questions that are relevant to the content. Each question should explore the topic in greater detail, seek clarification, or introduce new angles for discussion.

Instructions:
1. Build upon the key information, themes, or insights within the provided chat message.
2. Aim for variety in question type (clarifying, probing, or exploratory) to encourage a deeper conversation.
3. Ensure each question logically follows from the context of the provided chat message.
4. Keep questions concise yet insightful to maximize engagement.
5. Use the same language with the chat message content.

Now, generate 3–5 follow-up questions below:
"""

DEFAULT_CONDENSE_ANSWER_PROMPT = """\
Refine the agent-provided answer into the language used by the user (if different) and make stylistic adjustments for better readability.
Do not alter any content; all claims and facts must remain based on the agent-provided answer.

#### Variables:

Chat history:

- Previous chat history:

{{ chat_history }}

- User's followup question:

{{ question }}

---------------------

Agent's answer:

{{ agent_answer }}

---------------------

Now perform your task as follows:

#### Requirements:
1. **Language Detection**:
    - Identify the language used in the user's communication.
    - If it differs from the agent answer's language, translate accordingly.
2. **Stylistic Enhancement**:
    - Improve sentence flow and readability without changing the original meaning.
3. **Content Integrity**:
    - Maintain all original claims and facts from the agent answer.
    - Do not add new information or opinions.

Output the final answer for the user directly.
"""

DEFAULT_GENERATE_GOAL_PROMPT = """\
Given the conversation history between the User and Assistant, along with the latest follow-up question from the User, perform the following tasks:

1. **Language Detection**:
    - Analyze the User's follow-up question to determine the language used.

2. **Goal Generation**:
    - Determine the latest User intent from the follow-up question and the chat history.
    - Reformulate the latest User follow-up question into a clear, standalone question suitable for processing by the agent.
    - Specify the detected language for the answer.
    - Define the desired answer format.
    - Include any additional requirements as needed.

3. **Output**:
    - Produce a goal string in the following format:
      "[Refined Question] (Lang: [Detected Language], Format: [Format], Background: [Specified Goal Scenario])"

**Example**:

Chat history:

[]

---------------------

Follow-up question:

"sundb encryption at rest 会影响数据压缩比例吗？"

---------------------

Goal:

Does encryption at rest in SunDB affect the data compression ratio? (Lang: Chinese, Format: text, Background: TiDB product related consulting.)

**Your Task**:

Chat history:

{{chat_history}}

---------------------

Follow-up question:

{{question}}

---------------------

Goal:
"""