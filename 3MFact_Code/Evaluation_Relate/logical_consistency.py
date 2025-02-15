
from .gpt import *







def evaluate_logical_consistency(document_content, all_evidences):

    evaluate_logical_consistency_prompt = f"""
You will be given the news claim and potentially news image, headline and news post of it. You will then be given one fact-checking reasoning document for this news claim.

Your task is to rate the fact-checking document on one metric. The metric is 'logical_consistency'.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

**Metric Definition:** Logical_consistency refers to the internal coherence and soundness of reasoning presented in the fact-checking document. It ensures that the document's arguments and conclusions flow logically from one another without contradictions or fallacies.

**Evaluation Criteria:**
You should consider the following criteria when evaluating the logical consistency of a fact-checking document:
1. **Absence of Contradictions:**
-Does the document contain any statements that contradict each other?
-Are the explanations and evidence consistent throughout?

2. **Sound Reasoning:**
-Does the document follow a clear and logical path from the claim to the conclusion (true/false/misleading)?
-Are there any logical fallacies present (e.g., circular reasoning, strawman argument)?

3. **Clarity of Argument:**
-Are the arguments presented in a clear and well-structured manner?
-Are the key points and reasoning steps easy to follow?

4. **Relevancy of Information:**
-Does the fact-checking document stay focused on the specific claim being addressed?

**Evaluation Steps:**
1. **Identify Arguments:
-Locate the sections in the fact-checking document that explain why the claim is true, false, or misleading.

2. **Check for Contradictions:
-Carefully analyze the document for any statements that contradict each other or undermine the overall conclusion.

3. **Evaluate Reasoning:
-Analyze the logical flow of the arguments.
-Do the arguments follow a logical path (e.g., deduction, induction) without fallacies?

4. **Assess Clarity:
-Evaluate if the argument is presented clearly and concisely.
-Are the key points and reasoning steps easy to understand?

5. **Analyze Relevance:
-Determine if the information presented directly addresses the news claim being fact-checked.
-Does the document avoid irrelevant information or unnecessary details?

6. **Assign a Score:
-Based on the above analysis, assign a score from 1 to 5:
- **1:** Highly inconsistent. Reasoning is illogical, contradictory, or irrelevant to the claim.
- **2:** Low consistency. Reasoning has minor inconsistencies or lacks a clear logical flow.
- **3:** Moderate consistency. Reasoning is mostly consistent with some minor weaknesses or lack of clarity.
- **4:** High consistency. Reasoning is clear, logical, and avoids major fallacies.
- **5:** Excellent consistency. Reasoning is flawless, well-structured, and addresses potential counter-arguments.

**Additional Notes:**
* The LLM can consider the complexity of the arguments presented. More complex arguments require a higher level of logical consistency to be convincing.
* Even if the conclusion (true/false/misleading) disagrees with the LLM's initial assessment of the claim, as long as the reasoning is sound and consistent, the score should reflect logical coherence.


The claim and related information, along with the content of the fact-checking document, are as follows: 
{json.dumps(document_content, indent=4)}

Here are the relevant evidences for the "fact-checking document":
Here are the relevant evidences for the "fact-checking document":
{json.dumps(all_evidences, indent=4)}

**Output Format:**
- Logical_consistency (score ONLY):
- Judgement reason:
    """

    logical_consistency_answer = gpt4o_mini_analysis(evaluate_logical_consistency_prompt)


    formatted_prompt = f"""
    Please convert the following logical_consistency score analysis content into the specified JSON structure. Ensure the output is in valid JSON format and preserve the original content as much as possible, only modifying the structure to match the specified JSON format.

    The desired JSON structure:
    {{
        "Logical_Consistency": {{
            "Score": "An integer between 0 and 5"
            "Reason": "The detailed reason for the score"
        }}
    }}

    Important notes:
    - The "Score" field should be an integer between 0 and 5.
    - The "Reason" field should provide the justification for the given Logical_Consistency score.

    Logical_Consistency score analysis content to be converted:
    {logical_consistency_answer}

    Please provide the JSON output based on this analysis, following the structure and guidelines specified above.
    """

    formatted_logical_consistency_answer = gpt4o_mini_analysis(formatted_prompt)

    json_logical_consistency_answer = extract_complete_json(formatted_logical_consistency_answer)

    return json_logical_consistency_answer


