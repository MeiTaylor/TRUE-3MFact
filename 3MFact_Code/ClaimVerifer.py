import json
import logging
import os
import re

from LLM_Call import gpt4o_mini_analysis, extract_complete_json


















def process_claim_verifier(Claim, Video_information, QA_CONTEXTS, output_file_path):

    logging.warning("\n" * 5)

    logging.warning("----------------------------------")
    logging.warning("--------- Claim Verifier ---------")
    logging.warning("----------------------------------")



    prompt_for_claim_verifier = f"""
{{
  "Claim": "{Claim}",
  "Video_information": {Video_information},
  "QA_CONTEXTS": {QA_CONTEXTS},
  "Task": "Evaluate whether the existing information is sufficient to determine the accuracy of the Claim within the context provided by the Video_information. Please follow these steps:
    1. Analyze whether the information in QA_CONTEXTS sufficiently supports the accuracy assessment of the Claim.
    2. Verify if the verbs, nouns, and other lexical elements used in the Claim align with the provided contextual information.
    3. Determine if the existing data is adequate to make a reliable judgment on the accuracy of the Claim.
    
    Note: The task is not to judge the truthfulness of the Claim but to assess whether the existing information is sufficient to make such a determination.",
  "Output_Format": {{
    "CVResult": {{
      "Judgment": "Yes or No",
      "Confidence": "0%~100%",
      "Reason": "Detailed explanation for the judgment, including specific evidence from the provided information that explains why the information is sufficient or insufficient to determine the accuracy of the Claim"
    }}
  }}
}}
"""




    claim_verifier_answer = gpt4o_mini_analysis(prompt_for_claim_verifier)

    logging.info("################## Claim Verifier Input ##################")
    logging.info(prompt_for_claim_verifier)

    logging.info("################## Claim Verifier Raw Output ##################")
    logging.info(claim_verifier_answer)


    format_prompt_for_claim_verifier = f"""
Please convert the following text content into the specified JSON structure. Ensure the output is in JSON format and maintain the original content as much as possible, changing only the structure to the specified JSON format.

The desired JSON structure:
{{
  "CVResult": {{
    "Judgment": "Yes or No",
    "Confidence": "0%~100%",
    "Reason": "Concise explanation for the judgment"
  }}
}}

The content to be converted:
{claim_verifier_answer}
"""


    json_claim_verifier_answer = gpt4o_mini_analysis(format_prompt_for_claim_verifier)


    complete_json_claim_verifier_answer = extract_complete_json(json_claim_verifier_answer)



    if os.path.exists(output_file_path):
        with open(output_file_path, 'r+') as file:
            try:
                existing_data = json.load(file)
            except json.JSONDecodeError:
                existing_data = {}
    else:
        existing_data = {}

    if not isinstance(existing_data, dict):
        existing_data = {}

    if "CVResult" in existing_data:
        del existing_data["CVResult"]

    existing_data.update(complete_json_claim_verifier_answer)

    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)

    judgment_str = complete_json_claim_verifier_answer["CVResult"]["Judgment"]
    confidence_str = complete_json_claim_verifier_answer["CVResult"]["Confidence"]

    judgment_bool = True if judgment_str.lower() == "yes" else False
    confidence_float = float(confidence_str.strip('%')) / 100.0

    return judgment_bool, confidence_float











































































def analyze_string_yes_no(answer):
    answer = re.sub(r'[^\w\s]', '', answer).lower()
    words = answer.split()
    
    if len(words) == 1 and (words[0] == "yes" or words[0] == "no"):
        return words[0]
    else:
        return "neither"





def get_validator_result(Claim, Video_information, QA_CONTEXTS):


    logging.warning("\n" * 5)

    logging.warning("--------------------------------")
    logging.warning("--------- QA Validator ---------")
    logging.warning("--------------------------------")

    prompt_for_validator = f"""
{{
    "Claim": "{Claim}",
    "Video_information": {json.dumps(Video_information)},
    "QA_CONTEXTS": {json.dumps(QA_CONTEXTS)},
    "Task": "Based on the QA_CONTEXTS, determine if there is enough information to establish whether the Claim is true or false. Provide your answer in the 'QApairIsUseful' section. Answer 'yes' if the Question_Answer pair is valuable for verifying the Claim's accuracy, or 'no' if it is not valuable. Additionally, provide a detailed and specific reason for your answer."
    "QApairIsUseful": {{
        "Useful": "yes / no",
        "Reason": ""
    }}
}}
"""


    max_attempts = 5
    attempts = 0
    true_json_answer = None

    while attempts < max_attempts:
        attempts += 1

        logging.info(f"################## QA Validator Input (Attempt {attempts}) ##################")
        logging.info(prompt_for_validator)

        answer = gpt4o_mini_analysis(prompt_for_validator)

        logging.info(f"################## QA Validator Output (Attempt {attempts}) ##################")
        logging.info(answer)

        prompt_for_format = f"""
Please convert the following text content into the specified JSON structure. 

The desired JSON structure:
{{
    "QApairIsUseful": {{
        "Useful": "yes / no",
        "Reason": ""
    }}
}}

The content to be converted:
{answer}
"""

        json_answer = gpt4o_mini_analysis(prompt_for_format)

        true_json_answer = extract_complete_json(json_answer)

        if true_json_answer and true_json_answer.get("QApairIsUseful", {}).get("Useful") == "yes":
            return True
        elif true_json_answer and true_json_answer.get("QApairIsUseful", {}).get("Useful") == "no":
            return False
        else:
            continue

    return False


