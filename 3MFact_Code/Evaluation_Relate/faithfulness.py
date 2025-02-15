
from .gpt import *







def evaluate_faithfulness(document_content, all_evidences):

    evaluate_faithfulness_prompt = f"""
You will be given the news claim and potentially news image, headline and news post of it. You will then be given one fact-checking reasoning document for this news claim.
 
Your task is to rate the fact-checking document on one metric. The metric is 'faithfulness'.
 
Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.


**Metric Definition:** Faithfulness refers to the factual alignment between the fact-checking explanation and the evidence. It considers whether the document accurately reflects the evidence and provides an honest representation of the facts without distortion or misrepresentation..

**Evaluation Criteria:**
You should consider the following criteria when evaluating the faithfulness of a fact-checking document:
1. **Accurate Representation of the Claim:**
   - Does the document accurately represent the main points and arguments presented in the original news claim?
   - Are there any instances where the claim has been distorted or misinterpreted?

2. **Reliability of Sources:**
   - Does the fact-checking document use reliable and credible sources to support its analysis?
   - Are the sources correctly cited and relevant to the claim?

3. **Correct Interpretation of Evidence:**
   - Does the document correctly interpret and present the evidence used to support its conclusions?
   - Are there any instances of cherry-picking or selective use of evidence?

**Evaluation Steps:**

1. **Identify Key Points:**
   - Analyze the news claim and identify the main points and arguments.

2. **Assess Representation Accuracy:**
   - Evaluate if the fact-checking document accurately represents the original claim without distortion and aligns with the evidence provided.

3. **Check Source Reliability:**
   - Verify if the sources used in the document are credible, correctly cited, and relevant.

4. **Analyze Evidence Interpretation:**
   - Assess if the evidence is correctly interpreted and fairly presented without cherry-picking, discrepancies, contradictions, or misrepresentations of the facts in the document.

5. **Assign a Score:**
   - Based on the above analysis, assign a score from 0 to 5:
       - **1:** Not faithful. The document misrepresents the claim, uses unreliable sources, and misinterprets evidence.
       - **2:** Low faithfulness. The document has some accurate points but includes misrepresentations, unreliable sources, or incorrect evidence interpretation.
       - **3:** Moderate faithfulness. The document accurately represents the main points with generally reliable sources but may have minor issues with evidence interpretation.
       - **4:** High faithfulness. The document accurately represents the claim, uses reliable sources, and correctly interprets evidence with few inconsistencies.
       - **5:** Excellent faithfulness. The document faithfully represents the claim, uses highly reliable sources, and correctly interprets evidence consistently.

**Additional Notes:**
* The faithfulness score should focus on accuracy and reliability rather than the complexity of the claim.
* The LLM should not penalize the document for disagreeing with the initial assessment of the claim, as long as it provides a faithful analysis supporting its conclusion.


The claim and related information, along with the content of the fact-checking document, are as follows: 
{json.dumps(document_content, indent=4)}

Here are the relevant evidences for the "fact-checking document":
Here are the relevant evidences for the "fact-checking document":
{json.dumps(all_evidences, indent=4)}

**Output Format:**
- Faithfulness (score ONLY):
- Judgement reason:
    """

    faithfulness_answer = gpt4o_mini_analysis(evaluate_faithfulness_prompt)


    formatted_prompt = f"""
    Please convert the following faithfulness score analysis content into the specified JSON structure. Ensure the output is in valid JSON format and preserve the original content as much as possible, only modifying the structure to match the specified JSON format.

    The desired JSON structure:
    {{
        "Faithfulness": {{
            "Score": "An integer between 0 and 5"
            "Reason": "The detailed reason for the score"
        }}
    }}

    Important notes:
    - The "Score" field should be an integer between 0 and 5.
    - The "Reason" field should provide the justification for the given Faithfulness score.

    Faithfulness score analysis content to be converted:
    {faithfulness_answer}

    Please provide the JSON output based on this analysis, following the structure and guidelines specified above.
    """

    formatted_faithfulness_answer = gpt4o_mini_analysis(formatted_prompt)

    json_faithfulness_answer = extract_complete_json(formatted_faithfulness_answer)

    return json_faithfulness_answer


