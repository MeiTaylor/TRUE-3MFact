
from .gpt import *







def evaluate_currency(document_content, all_evidences):

    evaluate_currency_prompt = f"""
You will be given the news claim and potentially news image, headline and news post of it. You will then be given one fact-checking reasoning document for this news claim.

Your task is to rate the fact-checking document on one metric. The metric is 'currency'.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

**Metric Definition:** Currency refers to the extent to which the fact-checking document incorporates the latest information and data relevant to the news claim. It assesses whether the document is up-to-date and considers recent developments and current information relevant to the claim.

**Evaluation Criteria:**
You should consider the following criteria when evaluating the currency of a fact-checking document:
1. **Inclusion of Recent Information:**
   - Does the document include the latest information and data relevant to the news claim?
   - Are recent events and developments considered in the analysis?

2. **Timeliness of Evidence:**
   - Does the document use evidence that is timely and reflects the current state of the topic?
   - Are outdated sources or data avoided in the analysis?

**Evaluation Steps:**

1. **Identify Key Developments:**
   - Analyze the news claim and identify the most recent developments and data relevant to it.

2. **Assess Inclusion of Recent Information:**
   - Evaluate if the fact-checking document includes the latest information and data relevant to the claim.

3. **Analyze Timeliness of Evidence:**
   - Assess if the evidence used is timely and reflects the current state of the topic.

4. **Ensure Acknowledgment of Recent Changes:**
   - Check if the document acknowledges and incorporates any recent changes or updates related to the claim.

5. **Assign a Score:**
   - Based on the above analysis, assign a score from 0 to 5:
       - **1:** Not current. The document misses recent information, uses outdated sources, and ignores recent changes.
       - **2:** Low currency. The document includes some recent information but relies heavily on outdated sources or ignores important recent developments.
       - **3:** Moderate currency. The document includes relevant recent information with generally current sources but may miss some recent changes.
       - **4:** High currency. The document incorporates the latest information, uses current sources, and acknowledges recent changes with few inconsistencies.
       - **5:** Excellent currency. The document thoroughly incorporates the latest information and data, uses highly current sources, and fully acknowledges recent developments.

**Additional Notes:**
* The currency score should focus on the timeliness and relevance of the information rather than the complexity of the claim.
* The LLM should not penalize the document for disagreeing with the initial assessment of the claim, as long as it provides a timely and relevant analysis supporting its conclusion.



The claim and related information, along with the content of the fact-checking document, are as follows: 
{json.dumps(document_content, indent=4)}

Here are the relevant evidences for the "fact-checking document":
Here are the relevant evidences for the "fact-checking document":
{json.dumps(all_evidences, indent=4)}


**Output Format:**
- Currency (score ONLY):
- Judgement reason:
    """

    currency_answer = gpt4o_mini_analysis(evaluate_currency_prompt)


    formatted_prompt = f"""
    Please convert the following currency score analysis content into the specified JSON structure. Ensure the output is in valid JSON format and preserve the original content as much as possible, only modifying the structure to match the specified JSON format.

    The desired JSON structure:
    {{
        "Currency": {{
            "Score": "An integer between 0 and 5"
            "Reason": "The detailed reason for the score"
        }}
    }}

    Important notes:
    - The "Score" field should be an integer between 0 and 5.
    - The "Reason" field should provide the justification for the given Currency score.

    Currency score analysis content to be converted:
    {currency_answer}

    Please provide the JSON output based on this analysis, following the structure and guidelines specified above.
    """

    formatted_currency_answer = gpt4o_mini_analysis(formatted_prompt)

    json_currency_answer = extract_complete_json(formatted_currency_answer)

    return json_currency_answer


