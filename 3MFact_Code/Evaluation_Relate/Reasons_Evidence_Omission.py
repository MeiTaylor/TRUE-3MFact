
from .gpt import *




def evaluate_Omission_of_Reasons_and_Evidence(document_content, all_evidences, ground_truth):

    evaluate_Omission_of_Reasons_and_Evidence_prompt = f"""
You will be given the news claim and potentially news image, headline, and news post of it. You will then be given one fact-checking reasoning document for this news claim and the ground truth reasons and evidence.

Your task is to rate the fact-checking document on one metric. The metric is 'Omission_of_Reasons_and_Evidence'.

Please ensure you understand the instructions clearly. Refer back to these guidelines as needed throughout the evaluation process.

**Metric Definition:** Omission_of_Reasons_and_Evidence refers to the degree to which the fact-checking document fails to include necessary reasons or evidence when compared to the ground truth. It assesses whether the document overlooks critical pieces of evidence or reasoning that would otherwise strengthen its analysis and conclusion.

**Evaluation Criteria:**
When evaluating the fact-checking document for 'Omission of Reasons and Evidence', consider the following criteria:

1. **Comparison with Ground Truth:**
    - Does the document align with the ground truth reasons and evidence?
    - Are there any discrepancies?

2. **Completeness of Evidence:**
    - Does the document include all necessary evidence of the ground truth?
    - Are there any significant omissions?

3. **Thoroughness of Reasons for Conclusion:**
    - Is the reasons thorough and well-explained?
    - Are there any missing key points?

4. **Explores Alternative Evidence:**
    - Does the document consider alternative pieces of evidence that might support or contradict the claim?
    - Does it provide a rationale for why certain alternative evidence was dismissed?


**Evaluation Steps:**

1. **Identify Necessary Evidence:**
    - Analyze the news claim and Determine the critical pieces of evidence required to support or refute it.

2. **Compare with Ground Truth:**
    - Compare the document’s reasons and evidence with the provided ground truth.

3. **Check for Omissions:**
    - Look for any significant omissions of evidence or reasons that would affect the comprehensiveness of the document.

4. **Assess Completeness of Evidence:**
    - Ensure all necessary evidence is included.

5. **Evaluate Thoroughness of Reasons:**
    - Check the thoroughness of the reasons


6. **Assign a Score:**
   - Based on the above analysis, assign a score from 0 to 5:
        - **1:** Major Omissions. Critical evidence and reasons missing.
        - **2:** Significant Omissions. Several key points missing.
        - **3:** Moderate Omissions. Some important aspects missing.
        - **4:** Minor Omissions. Mostly complete with few gaps.
        - **5:** No Significant Omissions. Thorough and complete.

**Additional Notes:**
* Complex claims require more detailed evidence and reasoning.


The claim and related information, along with the content of the fact-checking document, are as follows: 
{json.dumps(document_content, indent=4)}

Here are the relevant evidences for the "fact-checking document":
Here are the relevant evidences for the "fact-checking document":
{json.dumps(all_evidences, indent=4)}

The the ground truth reasons and evidence are as follows:
{json.dumps(ground_truth, indent=4)}

**Output Format:**
- Omission_of_Reasons_and_Evidence (score ONLY):
- Judgement reason:
    """

    logging.info('-'*100)

    logging.info(f"evaluate_Omission_of_Reasons_and_Evidence_prompt: \n {evaluate_Omission_of_Reasons_and_Evidence_prompt}")

    Omission_of_Reasons_and_Evidence_answer = gpt4o_mini_analysis(evaluate_Omission_of_Reasons_and_Evidence_prompt)

    logging.info("-"*50)
    logging.info(f"Omission_of_Reasons_and_Evidence evaluation Result: \n {Omission_of_Reasons_and_Evidence_answer}")


    formatted_prompt = f"""
    Please convert the following Omission_of_Reasons_and_Evidence score analysis content into the specified JSON structure. Ensure the output is in valid JSON format and preserve the original content as much as possible, only modifying the structure to match the specified JSON format.

    The desired JSON structure:
    {{
        "Omission_of_Reasons_and_Evidence": {{
            "Score": "An integer between 0 and 5"
            "Reason": "The detailed reason for the score"
        }}
    }}

    Important notes:
    - The "Score" field should be an integer between 0 and 5.
    - The "Reason" field should provide the justification for the given Omission_of_Reasons_and_Evidence score.

    Omission_of_Reasons_and_Evidence score analysis content to be converted:
    {Omission_of_Reasons_and_Evidence_answer}

    Please provide the JSON output based on this analysis, following the structure and guidelines specified above.
    """

    formatted_Omission_of_Reasons_and_Evidence_answer = gpt4o_mini_analysis(formatted_prompt)

    json_Omission_of_Reasons_and_Evidence_answer = extract_complete_json(formatted_Omission_of_Reasons_and_Evidence_answer)

    return json_Omission_of_Reasons_and_Evidence_answer


