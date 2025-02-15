
from .gpt import *







def evaluate_strength_of_evidence(document_content, all_evidences):

    evaluate_strength_of_evidence_prompt = f"""
You will be given the news claim and potentially news image, headline and news post of it. You will then be given one fact-checking reasoning document for this news claim.

Your task is to rate the fact-checking document on one metric. The metric is 'strength_of_evidence'.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

**Metric Definition:** Strength_of_evidence refers to the robustness and sufficiency of the basis for the arguments presented in the fact-checking document. It considers whether there is sufficient evidence to support the points made and evaluates the quality and quantity of the evidence.

**Evaluation Criteria:**
You should consider the following criteria when evaluating the strength of evidence in a fact-checking document:
1. **Robustness of Arguments:**
   - Are the arguments based on strong, credible, and relevant evidence?
   - Is the evidence provided convincing and well-supported?

2. **Sufficiency of Evidence:**
   - Is there enough evidence to substantiate the points made in the document?
   - Are multiple sources and pieces of evidence used to support the arguments?

3. **Quality of Sources:**
   - Are the sources cited reliable, up-to-date, and relevant to the claim?
   - Does the document use a variety of high-quality sources to build its case?

4. **Relevance of Evidence:**
   - Is the evidence directly related to the claim and arguments presented?
   - Does the evidence effectively address and support the key points?

**Evaluation Steps:**

1. **Identify Key Arguments:**
   - Identify Key Evidence:- Analyze the news claim and determine the critical pieces of evidence required to support or refute it.

2. **Assess Robustness:**
   - Evaluate if the arguments are based on strong, credible, and relevant evidence.

3. **Check Sufficiency:**
   - Verify if there is enough evidence to substantiate the points made, using multiple sources if necessary.

4. **Analyze Quality of Sources:**
   - Assess if the sources cited are reliable, up-to-date, and relevant to the claim.

5. **Ensure Relevance:**
   - Check if the evidence is directly related to the claim and effectively supports the key points.

6. **Assign a Score:**
   - Based on the above analysis, assign a score from 0 to 5:
       - **1:** Weak evidence. The document's arguments are based on insufficient, unreliable, or irrelevant evidence.
       - **2:** Low strength. The document includes some evidence, but it is not robust or sufficient to strongly support the arguments.
       - **3:** Moderate strength. The document provides some credible evidence but may lack sufficiency or full relevance.
       - **4:** High strength. The document uses strong, credible, and relevant evidence to support its arguments, with few weaknesses.
       - **5:** Excellent strength. The document's arguments are well-supported by robust, sufficient, and highly relevant evidence from reliable sources.

**Additional Notes:**
* The strength of evidence score should focus on the robustness and sufficiency of the evidence provided, not necessarily the complexity of the claim.
* The LLM should not penalize the document for providing a detailed analysis if the evidence presented is strong and well-substantiated.



The claim and related information, along with the content of the fact-checking document, are as follows: 
{json.dumps(document_content, indent=4)}

Here are the relevant evidences for the "fact-checking document":
Here are the relevant evidences for the "fact-checking document":
{json.dumps(all_evidences, indent=4)}


**Output Format:**
- Strength_of_evidence (score ONLY):
- Judgement reason:

    """

    strength_of_evidence_answer = gpt4o_mini_analysis(evaluate_strength_of_evidence_prompt)


    formatted_prompt = f"""
    Please convert the following strength_of_evidence score analysis content into the specified JSON structure. Ensure the output is in valid JSON format and preserve the original content as much as possible, only modifying the structure to match the specified JSON format.

    The desired JSON structure:
    {{
        "Strength_of_Evidence": {{
            "Score": "An integer between 0 and 5"
            "Reason": "The detailed reason for the score"
        }}
    }}

    Important notes:
    - The "Score" field should be an integer between 0 and 5.
    - The "Reason" field should provide the justification for the given Strength_of_Evidence score.

    Strength_of_Evidence score analysis content to be converted:
    {strength_of_evidence_answer}

    Please provide the JSON output based on this analysis, following the structure and guidelines specified above.
    """

    formatted_strength_of_evidence_answer = gpt4o_mini_analysis(formatted_prompt)

    json_strength_of_evidence_answer = extract_complete_json(formatted_strength_of_evidence_answer)

    return json_strength_of_evidence_answer


