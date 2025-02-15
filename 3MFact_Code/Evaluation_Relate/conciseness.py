
from .gpt import *
import logging






def evaluate_conciseness(document_content, all_evidences):

    evaluate_conciseness_prompt = f"""
You will be given the news claim and potentially news image, headline and news post of it. You will then be given one fact-checking reasoning document for this news claim.

Your task is to rate the fact-checking document on one metric. The metric is 'conciseness'.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

**Metric Definition:** Conciseness refers to the brevity and clarity of the fact-checking document in presenting its analysis. It assesses whether the document avoids unnecessary complexity and redundancy, striving for simplicity while still conveying all necessary information. It considers whether the document is clear and to the point.

**Evaluation Criteria:**
You should consider the following criteria when evaluating the conciseness of a fact-checking document:
1. **Avoids Unnecessary Complexity:**
   - Does the document present its analysis in a straightforward manner without overly complicated language or structure?
   - Are technical terms and jargon minimized or clearly explained?

2. **Eliminates Redundancy:**
   - Does the document avoid repeating information unnecessarily?
   - Are the arguments presented efficiently without verbose descriptions?

3. **Strives for Simplicity:**
   - Is the document easy to understand, with clear and direct explanations?
   - Does it focus on essential points without extraneous details?

4. **Clarity of Presentation:**
   - Are the main points and conclusions clearly articulated?
   - Is the document well-organized, making it easy to follow?

**Evaluation Steps:**

1. **Identify Key Points:**
   - Analyze the news claim and identify the main points and arguments.

2. **Assess Complexity:**
   - Evaluate if the fact-checking document presents its analysis without unnecessary complexity.

3. **Check for Redundancy:**
   - Verify if the document avoids repeating information unnecessarily and presents arguments efficiently.

4. **Analyze Simplicity:**
   - Assess if the document is easy to understand, with clear and direct explanations focusing on essential points.

5. **Ensure Clarity:**
   - Check if the main points and conclusions are clearly articulated and well-organized.

6. **Assign a Score:**
   - Based on the above analysis, assign a score from 0 to 5:
       - **1:** Not concise. The document is overly complex, redundant, and difficult to understand.
       - **2:** Low conciseness. The document includes some unnecessary complexity and redundancy, with parts that are hard to follow.
       - **3:** Moderate conciseness. The document is generally clear but may include some unnecessary complexity or redundancy.
       - **4:** High conciseness. The document is straightforward, avoids redundancy, and is easy to understand with few issues.
       - **5:** Excellent conciseness. The document is very clear, simple, and free of unnecessary complexity and redundancy, and to the point.

**Additional Notes:**
* The conciseness score should focus on the simplicity and clarity of the information presented rather than the depth or thoroughness of the analysis.
* The LLM should not penalize the document for providing detailed explanations if they are necessary and presented clearly.

The claim and related information, along with the content of the fact-checking document, are as follows: 
{json.dumps(document_content, indent=4)}

Here are the relevant evidences for the "fact-checking document":
Here are the relevant evidences for the "fact-checking document":
{json.dumps(all_evidences, indent=4)}

**Output Format:**
- Conciseness (score ONLY):
- Judgement reason:
    """

    conciseness_answer = gpt4o_mini_analysis(evaluate_conciseness_prompt)


    formatted_prompt = f"""
    Please convert the following conciseness score analysis content into the specified JSON structure. Ensure the output is in valid JSON format and preserve the original content as much as possible, only modifying the structure to match the specified JSON format.

    The desired JSON structure:
    {{
        "Conciseness": {{
            "Score": "An integer between 0 and 5"
            "Reason": "The detailed reason for the score"
        }}
    }}

    Important notes:
    - The "Score" field should be an integer between 0 and 5.
    - The "Reason" field should provide the justification for the given Conciseness score.

    Conciseness score analysis content to be converted:
    {conciseness_answer}

    Please provide the JSON output based on this analysis, following the structure and guidelines specified above.
    """

    formatted_conciseness_answer = gpt4o_mini_analysis(formatted_prompt)

    json_conciseness_answer = extract_complete_json(formatted_conciseness_answer)

    return json_conciseness_answer


