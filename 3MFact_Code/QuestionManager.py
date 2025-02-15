import json
import logging
import os




from LLM_Call import *















# ------------------------------------------ #
# Prompts for the initial question generation
# ------------------------------------------ #


def generate_initial_question(Claim, Video_information, output_file_path):



    logging.warning("\n" * 5)

    logging.warning("------------------------------------------------")
    logging.warning("--------- Initial Question Generator ---------")
    logging.warning("------------------------------------------------")

    max_attempts = 3
    attempts = 0

    while attempts < max_attempts:


        prompt_for_initial_questions = f"""
{{
    "Claim": "{Claim}",
    "Video_information": {json.dumps(Video_information)},
    "Task": "Based on the provided Video_information and Claim, generate a series of professional, detailed questions to comprehensively assess the authenticity of the video content and verify the accuracy of the Claim.",
    "Requirements": {{
        "1. Primary Question": {{
            "Description": "Generate 1 crucial question addressing the most significant or potentially dubious aspect of the Claim.",
            "Rule": "This question should target the core issue, most effectively validating the Claim and video content authenticity."
        }},
        "2. Secondary Questions": {{
            "Description": "Generate 2 to 4 secondary questions, each targeting a different potential point of contention in the Claim.",
            "Rules": [
                "Each secondary question should focus on Claim content not addressed by the primary question.",
                "Ensure secondary questions cover all potentially problematic aspects of the Claim.",
                "Secondary questions should be distinct, each addressing a different point of contention.",
                "The number of secondary questions should flexibly range from 2 to 4, based on the Claim's complexity and number of contentious points."
            ]
        }},
        "General Rules": [
            "Each question should be a complete sentence, not exceeding 20 words.",
            "Questions should be clear, specific, and directly aimed at verifying authenticity.",
            "Questions should consider the relevant context provided in the Video_information."
        ]
    }},
    "Analysis Steps": [
        "1. Carefully analyze the Claim, identifying all potential points of contention.",
        "2. Focus the primary question on the most critical or likely contentious point.",
        "3. Allocate remaining contentious points to secondary questions, ensuring comprehensive coverage.",
        "4. Determine the appropriate number of secondary questions (2-4) based on the quantity and significance of contentious points.",
        "5. Reference the Video_information to ensure questions are relevant to the video content.",
        "6. Review and refine each question according to the general rules."
    ],
    "Output Format": {{
        "Initial_Question_Generation": {{
            "Primary_Question": "",
            "Secondary_Questions": {{
                "Secondary_Question_1": "",
                "Secondary_Question_2": "",
                "Secondary_Question_3": "",
                "Secondary_Question_4": ""
            }}
        }}
    }},
    "Note": "When outputting, if fewer than 4 secondary questions are generated, please remove the excess 'Secondary_Question' fields."
}}
"""






        logging.info(f"################## Initial Questions Generator Input (Attempt {attempts + 1}) ##################")
        logging.info(prompt_for_initial_questions)

        initial_questions_answer = gpt4o_mini_analysis(prompt_for_initial_questions)

        logging.info(f"################## Initial Question Generator Raw Output (Attempt {attempts + 1}) ##################")
        logging.info(initial_questions_answer)


        prompt_for_initial_question_json = f"""
{{
    "Task": "Convert the provided question generation results into a structured JSON format.",
    "Input": {{
        "Content": "{initial_questions_answer}"
    }},
    "Output_Requirements": {{
        "Format": "Valid JSON",
        "Structure": {{
            "Initial_Question_Generation": {{
                "Primary_Question": "string",
                "Secondary_Questions": {{
                    "Secondary_Question_1": "string",
                    "Secondary_Question_2": "string",
                    "Secondary_Question_3": "string (if present)",
                    "Secondary_Question_4": "string (if present)"
                }}
            }}
        }}
    }},
    "Rules": [
        "1. Extract the primary question and all secondary questions from the input content.",
        "2. Ensure each question is a complete sentence and does not exceed 20 words.",
        "3. Include all secondary questions present in the input, up to a maximum of 4.",
        "4. If fewer than 4 secondary questions are present, omit the unused fields.",
        "5. Maintain the original wording of the questions as much as possible.",
        "6. Ensure the output is in valid JSON format with proper escaping of special characters."
    ],
    "Note": "The output should match the actual number of questions in the input, with a minimum of one primary question and two secondary questions, and a maximum of four secondary questions."
}}

Please process the input content and provide the output in the specified JSON structure without any additional explanations or examples.
"""

        json_initial_question_answer = gpt4o_mini_analysis(prompt_for_initial_question_json)

        complete_json_initial_question_answer = extract_complete_json(json_initial_question_answer)


        
        primary_question_answer = complete_json_initial_question_answer.get('Initial_Question_Generation', {}).get('Primary_Question', '')

        secondary_questions = complete_json_initial_question_answer.get('Initial_Question_Generation', {}).get('Secondary_Questions', {})


        result = check_usefulness(Claim, Video_information, primary_question_answer)


        if result:
            logging.info("Generated question is useful.")
            break
        else:
            logging.info("Generated question is not useful. Regenerating...")
            attempts += 1

    if attempts == max_attempts:
        logging.info("Maximum attempts reached. Returning the last generated question.")
    

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
    
    existing_data.pop("Initial_Question_Generation", None)

    existing_data.update(complete_json_initial_question_answer)

    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)

    logging.info("Initial question has been generated and saved successfully.")

    return "Initial_Question_Generation", primary_question_answer, secondary_questions

















# ---------------------------------------------- #
# Prompts for the follow-up question generation
# ---------------------------------------------- #



def generate_follow_up_question(Claim, Video_information, QA_CONTEXTS, secondary_questions, output_file_path):


    logging.warning("\n" * 5)

    logging.warning("------------------------------------------------")
    logging.warning("--------- Follow Up Question Generator ---------")
    logging.warning("------------------------------------------------")

    max_attempts = 3
    attempts = 0

    while attempts < max_attempts:



        prompt_for_the_follow_up_question = f"""
{{
  "Claim": "{Claim}",
  "Video_information": {json.dumps(Video_information)},
  "QA_CONTEXTS": {json.dumps(QA_CONTEXTS)},
  "Secondary_Questions": {json.dumps(secondary_questions)},
  "Task": "Generate a professional and detailed follow-up question to verify the Claim, utilizing the provided information sources.",
  "Information Sources": [
    "1. Claim: The statement being investigated",
    "2. Video_information: Details about the video related to the claim",
    "3. QA_CONTEXTS: Previous question-answer pairs serving as reference"
  ],
  "Question Generation Methods": [
    {{"Method 1: Select from Secondary_Questions": [
      "- Review the provided Secondary_Questions list",
      "- Choose the most relevant question that addresses current information gaps",
      "- Ensure the selected question is not a duplicate of any in QA_CONTEXTS",
      "- The chosen question should provide a fresh perspective on the investigation"
    ]}},
    {{"Method 2: Generate a New Question": [
      "- Analyze Claim, Video_information, and QA_CONTEXTS thoroughly",
      "- Identify aspects of the Claim that haven't been addressed or need further clarification",
      "- Formulate a new, targeted question that explores these unexplored angles",
      "- Ensure the new question is unique and not present in QA_CONTEXTS"
    ]}}
  ],
  "Guidelines for Both Methods": [
    "- The question must help assess the authenticity of the Claim and Video_information",
    "- Avoid any duplication with questions in QA_CONTEXTS",
    "- Provide a fresh perspective that advances the investigation",
    "- Keep the question specific, relevant, and concise (max 20 words)",
    "- Focus on aspects that are crucial for identifying potential misinformation"
  ],
  "Output Requirements": [
    "A single, well-formulated question that meets all the above criteria. The question should provide a fresh perspective on the investigation, avoiding any duplication of previously asked questions. Ensure the question is specific, relevant, and concise, not exceeding 20 words."
  ]
}}
"""



        logging.info(f"################## Follow Up Question Input (Attempt {attempts + 1}) ##################")
        logging.info(prompt_for_the_follow_up_question)

        follow_up_question_answer = gpt4o_mini_analysis(prompt_for_the_follow_up_question)

        logging.info(f"################## Follow Up Question Output (Attempt {attempts + 1}) ##################")
        logging.info(follow_up_question_answer)

        result = check_usefulness(Claim, Video_information, follow_up_question_answer)

        if result:
            logging.info("Generated follow-up question is useful.")
            break
        else:
            logging.info("Generated follow-up question is not useful. Regenerating...")
            attempts += 1

    if attempts == max_attempts:
        logging.info("Maximum attempts reached. Returning the last generated follow-up question.")

    prompt_for_follow_up_question_formatting = f"""
    Please convert the following text content into the specified JSON structure. Ensure the output is in JSON format and maintain the original content as much as possible, changing only the structure to the specified JSON format. The final output should be a single question, in one sentence, not exceeding 30 words.

    The desired JSON structure:
    {{
      "Follow_Up_Question_Generation": {{
        "Question": ""
      }}
    }}

    The content to be converted:
    {follow_up_question_answer}
    """


    json_follow_up_question_answer = gpt4o_mini_analysis(prompt_for_follow_up_question_formatting)


    complete_json_follow_up_question_answer = extract_complete_json(json_follow_up_question_answer)



    if os.path.exists(output_file_path):
        with open(output_file_path, 'r+', encoding='utf-8') as file:
            try:
                existing_data = json.load(file)
            except json.JSONDecodeError:
                existing_data = {}
    else:
        existing_data = {}

    if not isinstance(existing_data, dict):
        existing_data = {}

    counter = 1
    new_key = f"Follow_Up_Question_{counter}"
    while new_key in existing_data:


        # if all(k in existing_data[new_key] for k in ['Question', 'Answer', 'Confidence']):
        #     counter += 1
        #     new_key = f"Follow_Up_Question_{counter}"

        if "QA" in existing_data[new_key] and isinstance(existing_data[new_key]["QA"], dict):
            qa_content = existing_data[new_key]["QA"]
            if all(k in qa_content for k in ['Question', 'Answer', 'Confidence']):
                print(f"键 {new_key} 中的 QA 包含所有必要字段")
                counter += 1
                new_key = f"Follow_Up_Question_{counter}"


        else:
            break

    existing_data.pop(f"Follow_Up_Question_{counter}", None)

    complete_json_follow_up_question_answer = {
        new_key: complete_json_follow_up_question_answer["Follow_Up_Question_Generation"]
    }

    existing_data.update(complete_json_follow_up_question_answer)

    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)

    logging.info("Follow-up question has been generated and saved successfully.")

    return new_key, complete_json_follow_up_question_answer[new_key]["Question"]







































def check_usefulness(Claim, Video_information, question):


    logging.warning("------------------------------------------------------")
    logging.warning("--------- Check Generate Question Usefulness ---------")
    logging.warning("------------------------------------------------------")

    attempts = 0
    max_attempts = 3

    while attempts < max_attempts:
        logging.info(f"################## Attempt {attempts + 1} ##################")
        
        prompt_for_usefulness_confirmation = f"""
{{
    "Claim": "{Claim}",
    "Video_information": {json.dumps(Video_information)},
    "Generated_Question": "{question}",
    "Task": "Evaluate if the generated question is useful for determining the authenticity of the Claim and Video_information to assess if they constitute misinformation. If the question addresses any aspect that helps in judging the truthfulness of the Claim and Video_information, it is considered useful. Provide your answer as 'yes' if the question is useful, and 'no' if it is not. Additionally, provide a reason explaining your assessment.",
    "Usefulness_Assessment": {{
        "Useful": "yes / no",
        "Reason": ""
    }}
}}
"""

        logging.info("################## Check Generate Question Usefulness Input ##################")
        logging.info(prompt_for_usefulness_confirmation)

        question_usefulness_confirmation = gpt4o_mini_analysis(prompt_for_usefulness_confirmation)

        logging.info("################## Check Generate Question Usefulness Output ##################")
        logging.info(question_usefulness_confirmation)

        prompt_for_format = f"""
Please convert the following text content into the specified JSON structure. 

The desired JSON structure:
{{
    "Usefulness_Assessment": {{
        "Useful": "yes / no",
        "Reason": ""
    }}
}}

The content to be converted:
{question_usefulness_confirmation}
"""


        answer_format = gpt4o_mini_analysis(prompt_for_format)
        json_useful = extract_complete_json(answer_format)

        if json_useful and 'Usefulness_Assessment' in json_useful:
            useful_value = json_useful['Usefulness_Assessment'].get('Useful', '').strip().lower()
            reason_value = json_useful['Usefulness_Assessment'].get('Reason', '').strip()

            if useful_value in ['yes', 'no']:
                return useful_value == 'yes'
        
        attempts += 1
    
    return False









