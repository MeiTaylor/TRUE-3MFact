

import json
import logging
import os
import regex

from LLM_Call import *













# ------------------------------- #
# Prompts for Reasoner
# ------------------------------- #


def process_claim_final(Claim, Video_information, QA_CONTEXTS, output_file_path):



    logging.warning("\n" * 5)

    logging.warning("----------------------------------")
    logging.warning("--------- Final Reasoner ---------")
    logging.warning("----------------------------------")



    prompt_for_reasoner = f"""
{{
  "Claim": "{Claim}",
  "Video_information": {Video_information},
  "QA_CONTEXTS": {QA_CONTEXTS},
  "Task": "" Please analyze the authenticity of the Claim by following these steps:
    1. Carefully examine whether the verbs, nouns, and other lexical elements used in the Claim are consistent with the provided Video information and QA_CONTEXTS.
    2. Based on the provided Video information, determine whether the Claim is True or False.
      - True indicates that within the context of the Video information, the Claim is a genuine piece of news.
      - False indicates that within the context of the Video information, the Claim is false and contains misinformation.
    3. If determined to be 'False', specifically identify the type of misinformation: False video description, Video Clip Edit, Computer-generated Imagery, False speech, Staged Video, Text-Video Contradictory, Text unsupported by the video.
    4. Provide a detailed reasoning process, ensuring that the final answer covers all the following aspects: Answer, Reasons, Therefore, the Claim authenticity is, If it is false, the specific type of False Information is. "",
  "Output_Answer_Format": {{
    "Answer": "True or False",
    "Reasons": "",
    "Therefore, the Claim authenticity is": "True or False",
    "If it is false, the specific type of False Information is": ""
  }},
  "Please Note": "Ensure to specifically address Answer, Reasons, Therefore, the Claim authenticity is, and If it is false, the specific type of False Information is based on the provided Video information. In the Reasons section, particularly note whether the vocabulary used in the Claim aligns with the contextual information provided.",
  "Evidence Citation Guidelines": {{
    "1. Reference Format": "For each point or piece of information used, add a citation number following it.",
    "2. Citation Format": "Use the format [Question_type Question_number evidence] for citations. For example:",
      "- Initial Question Generation": "[Initial_Question_Generation evidence]",
      "- Follow Up Questions": "[Follow_Up_Question_Z evidence]",
      "where Z is the follow-up question number.",
    "3. Comprehensive Referencing": "Ensure that every piece of information extracted from the provided evidence is properly cited.",
    "4. Answer Quality": "Strive to provide thorough yet concise answers, directly addressing the question while including relevant evidence.",
    "5. Output Inclusion": "When citing evidence in the final output, include the full citation tags in the 'Reasons' section of the Output_Answer_Format (e.g., [Follow_Up_Question_1 evidence])."
  }}
}}
"""



    attempt = 0
    max_attempts = 3
    true_json_answer = {}

    while attempt < max_attempts:
        attempt += 1
        answer = gpt4o_mini_analysis(prompt_for_reasoner)

        logging.info(f"################## Final Reasoner Input (Attempt {attempt}) ##################")
        logging.info(prompt_for_reasoner)

        answer = gpt4o_mini_analysis(prompt_for_reasoner)

        logging.info(f"################## Final Reasoner Raw Output (Attempt {attempt}) ##################")
        logging.info(answer)


        
        prompt_for_format = f"""
        Please convert the following text content into the specified JSON structure. Ensure the output is in JSON format and maintain the original content as much as possible, changing only the structure to the specified JSON format.

        First, determine whether the claim is True or False.

        For True claims, use the following JSON structure:
        {{
        "Final_Judgement": {{
        "Answer": "True",
        "Reasons": "{{{{reasons}}}}",
        "Therefore, the Claim authenticity is": "True",
        "The information type is": "Real"
        }}
        }}

        For False claims, use the following JSON structure:
        {{
        "Final_Judgement": {{
        "Answer": "False",
        "Reasons": "{{{{reasons}}}}",
        "Therefore, the Claim authenticity is": "False",
        "The information type is": "False",
        "The specific type of False Information is": "{{{{false_info_type}}}}"
        }}
        }}

        If the claim is False, please choose the specific type of False Information from the following options:
        False video description, Video Clip Edit, Computer-generated Imagery, False speech, Staged Video, Text-Video Contradictory, Text unsupported by the video, Other.

        Content to be converted:
        {answer}

        Please Note: The final task is to first determine the authenticity of the claim. If it is True, only the "Reasons" field needs to be filled. If it is False, the "Reasons" field should be filled, and the "The specific type of False Information" field should be selected.
        """

        json_answer = gpt4o_mini_analysis(prompt_for_format)
        true_json_answer = extract_complete_json(json_answer)

        if validate_json_structure(true_json_answer):
            break



    if os.path.exists(output_file_path):
        with open(output_file_path, 'r+', encoding="utf-8") as file:
            try:
                existing_data = json.load(file)
            except json.JSONDecodeError:
                existing_data = {}
    else:
        existing_data = {}

    if not isinstance(existing_data, dict):
        existing_data = {}

    existing_data.update(true_json_answer)

    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)

    logging.info("The claim processing result has been generated and successfully saved.")


    return true_json_answer
















def validate_json_structure(json_data):
    if not isinstance(json_data, dict):
        return False

    final_judgement = json_data.get("Final_Judgement")
    if not final_judgement:
        return False

    answer = final_judgement.get("Answer")
    reasons = final_judgement.get("Reasons")
    claim_authenticity = final_judgement.get("Therefore, the Claim authenticity is")
    info_type = final_judgement.get("The information type is")

    if answer == "True":
        return reasons and claim_authenticity == "True" and info_type == "Real" and "The specific type of False Information is" not in final_judgement

    if answer == "False":
        false_info_type = final_judgement.get("The specific type of False Information is")
        return reasons and claim_authenticity == "False" and info_type == "False" and false_info_type

    return False
