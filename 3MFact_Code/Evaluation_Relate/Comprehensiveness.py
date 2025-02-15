
from .gpt import *




def evaluate_comprehensiveness(document_content, all_evidences):

    evaluate_comprehensiveness_prompt = f"""
    You will be given the news claim and potentially news image, headline and news post of it. You will then be given one fact-checking reasoning document for this news claim.

    Your task is to rate the fact-checking document on one metric. The metric is 'Comprehensiveness'.

    Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

    **Metric Definition:** Comprehensiveness refers to the completeness of the fact-checking document in addressing the news claim. It considers whether the document covers all relevant aspects of the claim, explores potential counter-arguments, and provides a well-rounded understanding of the issue.

    **Evaluation Criteria:**
    You should consider the following criteria when evaluating the comprehensiveness of a fact-checking document:
    1. **Addresses All Key Points:** 
        - Does the document address all the main points and arguments presented in the original news claim? 
        - Does it identify any ambiguities or unclear aspects of the claim?

    2. **Considers Context:** 
        - Does the fact-checking document take into account the broader context surrounding the claim (e.g., historical background, current events)?

    3. **Examines Counter-Arguments:** 
        - Does the document acknowledge and address any potential opposing viewpoints or alternative explanations? 
        - Does it present evidence that challenges or strengthens the original claim?

    4. **Provides Depth of Explanation:** 
        - Does the fact-checking document offer a thorough explanation of the reasoning behind its conclusion (true/false/misleading)? 
        - Does it provide sufficient details and evidence to support its claims?

    **Evaluation Steps:**

    1. **Identify Key Points:** 
        - Analyze the news claim and identify the main points, arguments, and any ambiguities.

    2. **Assess Contextual Consideration:** 
        - Evaluate if the fact-checking document acknowledges the broader context surrounding the claim.

    3. **Check for Counter-Arguments:** 
        - Look for sections that address opposing viewpoints or alternative explanations to the claim.

    4. **Analyze Explanation Depth:** 
        - Evaluate if the reasoning behind the conclusion is well-explained and supported by sufficient details and evidence.

    5. **Assign a Score:** 
    - Based on the above analysis, assign a score from 0 to 5:
            - **1:** Not comprehensive. The document misses key points of the claim, lacks context, and ignores counter-arguments.
            - **2:** Low comprehensiveness. The document addresses some aspects of the claim but lacks depth, context, or consideration of counter-arguments.
            - **3:** Moderate comprehensiveness. The document covers the main points of the claim with some context and may address some counter-arguments.
            - **4:** High comprehensiveness. The document thoroughly examines the claim, considers context, and presents arguments and counter-arguments with sufficient explanation.
            - **5:** Excellent comprehensiveness. The document provides a complete and nuanced understanding of the claim, analyzes all relevant aspects in detail, and explores alternative viewpoints comprehensively.

    **Additional Notes:**
    * The comprehensiveness score might be influenced by the complexity of the news claim. A complex claim might require a higher degree of detail and exploration of various perspectives.
    * The LLM should not penalize the document for disagreeing with the initial assessment of the claim, as long as it provides a comprehensive analysis supporting its conclusion.

    The claim and related information, along with the content of the fact-checking document, are as follows: 
    {json.dumps(document_content, indent=4)}

    Here are the relevant evidences for the "fact-checking document":
    Here are the relevant evidences for the "fact-checking document":
    {json.dumps(all_evidences, indent=4)}


    **Output Format:**
    Comprehensiveness (score ONLY):
    Judgement reason:
    """

    comprehensiveness_answer = gpt4o_mini_analysis(evaluate_comprehensiveness_prompt)


    formatted_prompt = f"""
    Please convert the following comprehensiveness score analysis content into the specified JSON structure. Ensure the output is in valid JSON format and preserve the original content as much as possible, only modifying the structure to match the specified JSON format.

The desired JSON structure:
{{
    "Comprehensiveness": {{
        "Score": "An integer between 0 and 5"
        "Reason": "The detailed reason for the score"
    }}
}}

    Important notes:
    - The "Score" field should be an integer between 0 and 5.
    - The "Reason" field should provide the justification for the given Comprehensiveness score.

    Comprehensiveness score analysis content to be converted:
    {comprehensiveness_answer}

    Please provide the JSON output based on this analysis, following the structure and guidelines specified above.
    """

    formatted_comprehensiveness_answer = gpt4o_mini_analysis(formatted_prompt)

    json_comprehensiveness_answer = extract_complete_json(formatted_comprehensiveness_answer)

    return json_comprehensiveness_answer



