
from .gpt import *







def evaluate_fact_hallucination(document_content, all_evidences):

    evaluate_fact_hallucination_prompt = f"""
You will be given the news claim and potentially news image, headline, and news post of it. You will then be given one fact-checking reasoning document for this news claim. 

Your task is to rate the fact-checking document on one metric. The metric is 'Fact_Hallucination'.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

**Metric Definition:** Fact_Hallucination refers to the presence of information in the fact-checking document that is not supported by evidence or is entirely fabricated. It assesses whether the document contains any incorrect or misleading information that cannot be verified by reliable sources.

**Evaluation Criteria:**
You should consider the following criteria when evaluating the fact hallucination in a fact-checking document:
1. **Presence of Unsupported Claims:**
   - Does the document include any statements or claims that are not supported by evidence?
   - Are there any assertions made without proper references or sources?

2. **Accuracy of Information:**
   - Is the information presented in the document factually correct and verifiable?
   - Are there any false or fabricated details?

3. **Verification of Sources:**
   - Are the sources cited in the document reliable and verifiable?
   - Does the document use credible sources to back its claims?

4. **Consistency with Known Facts:**
   - Does the document align with established facts and widely accepted knowledge?
   - Are there any deviations from verified information?

**Evaluation Steps:**

1. **Identify Key Points:**
   - Analyze the news claim and identify the main points and arguments presented in the fact-checking document.

2. **Assess Unsupported Claims:**
   - Evaluate if the document includes any statements or claims that lack supporting evidence.

3. **Check Information Accuracy:**
   - Verify the factual correctness and verifiability of the information presented in the document.

4. **Analyze Source Reliability:**
   - Assess the reliability and credibility of the sources cited in the document.

5. **Ensure Consistency with Known Facts:**
   - Check if the document aligns with established facts and widely accepted knowledge.

6. **Assign a Score:**
   - Based on the above analysis, assign a score from 0 to 5:
       - **1:** High hallucination. The document contains numerous unsupported claims and fabricated information.
       - **2:** Significant hallucination. The document includes some unsupported or incorrect claims, with several factual inaccuracies.
       - **3:** Moderate hallucination. The document is generally accurate but contains a few unsupported or incorrect claims.
       - **4:** Low hallucination. The document is mostly accurate with minor issues, and most claims are well-supported by evidence.
       - **5:** No hallucination. The document is completely accurate, with all claims well-supported by reliable evidence.

**Additional Notes:**
* The fact hallucination score should focus on the presence of unsupported or fabricated information rather than the depth or comprehensiveness of the analysis.
* The LLM should not penalize the document for minor mistakes if the overall information is accurate and well-supported by evidence.



The claim and related information, along with the content of the fact-checking document, are as follows: 
{json.dumps(document_content, indent=4)}

Here are the relevant evidences for the "fact-checking document":
Here are the relevant evidences for the "fact-checking document":
{json.dumps(all_evidences, indent=4)}


**Output Format:**
- Fact_Hallucination (score ONLY):
- Judgement reason:
    """
    logging.info('-'*100)
    logging.info(f"Fact_Hallucination evaluation prompt: \n {evaluate_fact_hallucination_prompt}")


    fact_hallucination_answer = gpt4o_mini_analysis(evaluate_fact_hallucination_prompt)
    logging.info('-'*100)
    logging.info(f"Fact_Hallucination evaluation Result: \n {fact_hallucination_answer}")


    formatted_prompt = f"""
    Please convert the following fact_hallucination score analysis content into the specified JSON structure. Ensure the output is in valid JSON format and preserve the original content as much as possible, only modifying the structure to match the specified JSON format.

    The desired JSON structure:
    {{
        "Fact_Hallucination": {{
            "Score": "An integer between 0 and 5"
            "Reason": "The detailed reason for the score"
        }}
    }}

    Important notes:
    - The "Score" field should be an integer between 0 and 5.
    - The "Reason" field should provide the justification for the given Fact_Hallucination score.

    Fact_Hallucination score analysis content to be converted:
    {fact_hallucination_answer}

    Please provide the JSON output based on this analysis, following the structure and guidelines specified above.
    """

    formatted_fact_hallucination_answer = gpt4o_mini_analysis(formatted_prompt)

    json_fact_hallucination_answer = extract_complete_json(formatted_fact_hallucination_answer)

    return json_fact_hallucination_answer


