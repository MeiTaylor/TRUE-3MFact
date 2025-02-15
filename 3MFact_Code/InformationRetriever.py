# -*- coding: utf-8 -*-

# Standard library imports
import difflib
import json
import logging
import os
import random
import re
import threading
import time
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Lock, Semaphore


import spacy
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
import time
import concurrent.futures



import regex
import requests
from bs4 import BeautifulSoup
from dateutil.parser import parse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


import trafilatura
import tiktoken



# Local module imports
from LLM_Call import *
from Config import INFORMATION_RETRIEVER_CONFIG, MODEL_CONFIG





def information_retriever_complete(claim, Video_information, QA_CONTEXTS, question, output_file_path, video_id):
    
    
    logging.warning("\n" * 5)
    logging.warning("-----------------------------------------")
    logging.warning("--------- Information Retriever ---------")
    logging.warning("-----------------------------------------")
    

    def process_query(query_key, query, searching_goal, claim, Video_information, QA_CONTEXTS, question, output_file_path):
        logging.info(f"Processing query key: {query_key}, query: {query}, searching_goal: {searching_goal}")

        prefix = os.path.dirname(output_file_path)

        single_query_path = os.path.join(prefix, f"{query_key}.json")

        process_query_and_quality_score_value(query, searching_goal, claim, Video_information, QA_CONTEXTS, question, single_query_path)
        
        updated_single_query_path = single_query_path.replace(".json", "_updated.json")

        process_evidence_and_Newness_Relevance(query_key, query, claim, Video_information, QA_CONTEXTS, question, updated_single_query_path)





    attempt_count = 0  

    while attempt_count < INFORMATION_RETRIEVER_CONFIG.get("max_iterations"):

        try:
            with open(output_file_path, 'r', encoding='utf-8') as file:
                try:
                    full_data = json.load(file)
                except json.JSONDecodeError:
                    full_data = {}  
        except FileNotFoundError:
            full_data = {}  

        full_data["Question"] = question

        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(full_data, f, indent=4, ensure_ascii=False)


        attempt_count += 1
        need_online_search = check_online_search_needed(claim, Video_information, QA_CONTEXTS, question, output_file_path)

        if not need_online_search:
            logging.info("No online search needed.")
            process_question_VideoLLM(claim, Video_information, question, output_file_path, video_id)

            return  
        
        else:


            logging.info("Online search needed.")
            OnlineSearchTerms = generate_OnlineSearchTerms(claim, Video_information, QA_CONTEXTS, question, output_file_path)
            
            queries = OnlineSearchTerms['OnlineSearchTerms']['Queries']
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = []
                for query_key, query_details in queries.items():
                    query = query_details['query']
                    searching_goal = query_details['searching_goal']
                    futures.append(executor.submit(
                        process_query, 
                        query_key, 
                        query, 
                        searching_goal, 
                        claim, 
                        Video_information, 
                        QA_CONTEXTS, 
                        question, 
                        output_file_path
                    ))
            
                for future in futures:
                    future.result()
            logging.info("两个查询处理均执行完毕")




            now_folder_path = os.path.dirname(output_file_path)

            process_json_files(now_folder_path, output_file_path)


            now_evidences_useful = select_useful_evidence(claim, Video_information, QA_CONTEXTS, question, output_file_path)

            if now_evidences_useful:
                break  
            else:
                logging.info("Some queries failed, retrying...")


                for key in queries.keys():
                    prefix = os.path.dirname(output_file_path)
                    single_query_path = os.path.join(prefix, f"{key}.json")
                    updated_single_query_path = single_query_path.replace(".json", "_updated.json")

                    if os.path.exists(single_query_path):
                        os.remove(single_query_path)
                        logging.info(f"Deleted file: {single_query_path}")
                        
                    if os.path.exists(updated_single_query_path):
                        os.remove(updated_single_query_path)
                        logging.info(f"Deleted file: {updated_single_query_path}")

                with open(output_file_path, 'w', encoding='utf-8') as f:
                    f.truncate(0)
                logging.info(f"Cleared file contents: {output_file_path}")




    process_claim_and_generate_answer(claim, Video_information, QA_CONTEXTS, question, output_file_path)










































process_google_search_lock = Lock()
last_call_time = 0

from Config import API_CONFIG


def google_search(question):

    try:

        api_key = API_CONFIG.get("google_api_key")


        base_url = "https://cn2us02.opapi.win/api/v1/openapi/search/google-search/v1"
        excluded_sites = (
            "www.snopes.com, www.factcheck.org, www.politifact.com, "
            "www.truthorfiction.com, fullfact.org, www.hoax-slayer.com, leadstories.com, "
            "www.opensecrets.org, www.washingtonpost.com/news/fact-checker, "
            "www.reuters.com/fact-check, apnews.com/APFactCheck, www.bbc.com/news/reality_check, "
            "factcheckni.org, facta.news, checkyourfact.com, africacheck.org, verafiles.org, "
            "maldita.es, correctiv.org, teyit.org"
        )
        
        url = f"{base_url}?key={api_key}&q={question}&siteSearch={excluded_sites}&siteSearchFilter=e"
        
        headers = {
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
            'Content-Type': 'application/x-www-form-urlencoded',
            'Authorization': 'Bearer ' + api_key,
        }
        
        response = requests.request("GET", url, headers=headers)
        return response.text
    
    except requests.exceptions.RequestException as e:
        print(f"请求发生错误: {str(e)}")
        logging.error(f"请求发生错误: {str(e)}")
        return None
    except ValueError as e:
        print(f"JSON解析错误: {str(e)}")
        logging.error(f"JSON解析错误: {str(e)}")
        return None
    except Exception as e:
        print(f"发生未知错误: {str(e)}")
        logging.error(f"发生未知错误: {str(e)}")
        return None



def extract_complete_json(response_text):
    json_pattern = r'(\{(?:[^{}]|(?1))*\})'
    matches = regex.findall(json_pattern, response_text)
    if matches:
        try:
            for match in matches:
                json_data = json.loads(match)
                return json_data
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
    return None





def find_best_match(link, evaluation):
    best_match = None
    highest_ratio = 0
    for eval_url in evaluation:

        ratio = difflib.SequenceMatcher(None, link, eval_url).ratio()
        if ratio > highest_ratio:
            highest_ratio = ratio
            best_match = eval_url
    return best_match, highest_ratio



def process_item(i, item):
    link = item.get('link')
    snippet = item.get('snippet')
    content, content_tokens = get_content_and_word_count(link, snippet)
    item['website_content'] = {
        'content': content,
        'content_tokens': content_tokens
    }
    return {f'evidence{i}': item}


def process_google_search(query, output_file_path):
    data = google_search(query)
    logging.info("Google search over")

    data = json.loads(data)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_item, i, item) for i, item in enumerate(data.get('items', []))]
        new_items = [future.result() for future in futures]

    data['items'] = new_items

    with open(output_file_path, 'w', encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)







def analyze_string(answer):
    answer = re.sub(r'[^\w\s]', '', answer).lower()
    words = answer.split()
    
    yes_count = words.count("yes")
    no_count = words.count("no")
    
    first_word = words[0] if words else ""
    
    first_word_is = "yes" if first_word == "yes" else "no" if first_word == "no" else "neither"
    
    final_result = "neither"  

    if yes_count > no_count and first_word_is == "yes":
        final_result = "yes"
    elif yes_count < no_count and first_word_is == "no":
        final_result = "no"
    
    return final_result











def check_online_search_needed(claim, Video_information, QA_CONTEXTS, question, output_file_path):
    logging.info("Checking if online search is needed")

    
    prompt_for_CheckOnlineSearch = f"""
{{
  "Claim": "{claim}",
  "New_Question": {{
    "Question": "{question}"
  }},
  "Task": "Determine if the question requires online search or can be answered by video analysis alone",
  "Input": {{
    "Claim": "{claim}",
    "Question": "{question}"
  }},
  "Decision_Criteria": [
    "1. Online search is generally recommended for most questions to provide comprehensive and up-to-date information.",
    "2. If the question specifically asks about details or content exclusively present in the video, online search may not be needed.",
    "3. For any question that could benefit from additional context, background, or current information, recommend online search.",
    "4. If in doubt, default to recommending online search to ensure a more thorough investigation."
  ],
  "Examples": [
    {{
      "Question": "In the video, what specific statements does the speaker make about climate change?",
      "CheckOnlineSearch": {{
        "need_online_search": "No",
        "reasoning": "The question explicitly asks about content in the video, which can be answered through video analysis alone."
      }}
    }},
    {{
      "Question": "How does the claim compare to current scientific consensus?",
      "CheckOnlineSearch": {{
        "need_online_search": "Yes",
        "reasoning": "This question requires up-to-date scientific information not likely present in the video, necessitating online search."
      }}
    }}
  ],
  "Output Format": {{
    "CheckOnlineSearch": {{
      "need_online_search": "[Yes/No]",
      "reasoning": "Briefly explain the rationale behind the decision to search or not search online, including how you arrived at the probability"
    }}
  }}
}}
"""

    
    
    logging.info("################## Checking if online search is needed Input ##################")
    logging.info(prompt_for_CheckOnlineSearch)

    answer = gpt4o_mini_analysis(prompt_for_CheckOnlineSearch)

    logging.info("################## Checking if online search is needed Raw Output ##################")
    logging.info(answer)



    



    prompt_for_query_format = f"""
Please convert the following text into the specified JSON structure without altering the original meaning. Ensure the output is in valid JSON format.

Required JSON structure:
{{
  "CheckOnlineSearch": {{
    "need_online_search": "[Yes/No]",
    "reasoning": "Explanation for whether online search is needed"
  }}
}}

Notes:
1. The "need_online_search" field must strictly be "Yes" or "No", with the first letter capitalized.
2. The "reasoning" field should concisely explain the decision reason, not exceeding 100 words.
3. Ensure the JSON format is correct and parsable.
4. Do not add any extra fields or comments.

Text to be converted:
{answer}

Please output only the JSON that adheres to the above structure, without including any other text or explanation.
"""



    CheckOnlineSearch_json_answer = gpt4o_mini_analysis(prompt_for_query_format)
    CheckOnlineSearch_complete_json_answer = extract_complete_json(CheckOnlineSearch_json_answer)

    need_online_search_value = CheckOnlineSearch_complete_json_answer.get("CheckOnlineSearch", {}).get("need_online_search", "").lower()
    final_result = need_online_search_value == "yes"



    if os.path.exists(output_file_path):
        with open(output_file_path, 'r') as file:
            data = json.load(file)
    else:
        data = {}

    data["CheckOnlineSearch"] = CheckOnlineSearch_complete_json_answer


    with open(output_file_path, 'w') as file:
        json.dump(data, file)
    
    return final_result




def generate_OnlineSearchTerms(claim, Video_information, QA_CONTEXTS, question, output_file_path):
    logging.info("Generating and formatting queries")
    prompt_for_queries_generation = f"""
{{
  "Claim": "{claim}",
  "Video_information": {json.dumps(Video_information)},
  "QA_CONTEXTS": {json.dumps(QA_CONTEXTS)},
  "New_Question": {{
    "Question": "{question}"
  }},
  "Task": "In order to better answer the 'Question': '{question}', please determine what information is required and design two new queries to search for this information on Google. These two queries should be specifically aimed at retrieving relevant information from the web to better answer the 'Question': '{question}'. Please note that the generated queries should not exceed two, and they should focus on different aspects and not be repetitive. The above 'Claim', 'Video_information', and 'QA_CONTEXTS' are just background information and the queries should focus on answering 'the new question'. Ensure that the queries are in the format suitable for entering into a Google search.",
  "Analysis Process":
    "Generate the queries by following these steps, and output the result of each step:
    Step 1: Generate the 'searching goals' to answer this question.
    Step 2: Please generate 1 professional searching query for each goal used to perform a search using the Google Custom Search API. Based on the above 'searching goals'.",
  "OnlineSearchTerms": {{
    "Queries": {{
      "Query1": {{
        "query": "Concise search query (max 10 words) for one aspect, suitable for Google Search",
        "searching_goal": "Provide detailed and specific explanations to clearly define the objectives that this query aims to achieve"
      }},
      "Query2": {{
        "query": "Different concise search query (max 10 words) for another aspect, suitable for Google Search",
        "searching_goal": "Provide detailed and specific explanations to clearly define the objectives that this query aims to achieve"
      }}
    }}
  }}
}}

Note: Both the two-step analysis process and the final JSON answer need to be outputted. The two-step analysis includes:
Step 1: Generate the 'searching goals' to answer this question.
Step 2: Please generate 1 professional searching query for each goal used to perform a search using the Google Custom Search API. Based on the above 'searching goals'.
"""
    


    logging.info("################## Generating OnlineSearchTerms Input ##################")
    logging.info(prompt_for_queries_generation)

    query_answer = gpt4o_mini_analysis(prompt_for_queries_generation)

    logging.info("################## Generating OnlineSearchTerms Raw Output ##################")
    logging.info(query_answer)




    prompt_for_query_format = f"""
Please convert the following text content into the specified JSON structure without altering the original query content. Ensure the output is in valid JSON format and maintains the exact wording of the original queries.

Input Content:
{query_answer}

Desired JSON Structure:
{{
  "OnlineSearchTerms": {{
    "Queries": {{
      "Query1": {{
        "query": "Exact text of the first query (max 10 words)",
        "searching_goal": "Detailed explanation of the first query's objective"
      }},
      "Query2": {{
        "query": "Exact text of the second query (max 10 words)",
        "searching_goal": "Detailed explanation of the second query's objective"
      }}
    }}
  }}
}}

Instructions:
1. Extract the exact text of both queries from the input content.
2. Identify the corresponding searching goals for each query.
3. Format the extracted information into the specified JSON structure, including the "OnlineSearchTerms" key.
4. Ensure that the "query" fields contain only the concise search queries (maximum 10 words each).
5. Include the full explanations of the searching goals in the "searching_goal" fields.
6. Maintain the original wording and order of the queries as they appear in the input content.
7. If only one query is provided in the input, leave the second query fields empty but include them in the JSON structure.

The output should be a valid JSON object that can be parsed without errors. Do not include any additional text or explanations outside of the JSON structure.
"""


    query_json_answer = gpt4o_mini_analysis(prompt_for_query_format)
    query_complete_json_answer = extract_complete_json(query_json_answer)

    logging.info("Query Complete JSON Answer")
    logging.info(query_complete_json_answer)

    with open(output_file_path, 'r', encoding='utf-8') as file:
        full_data = json.load(file)
        full_data.update(query_complete_json_answer)

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(full_data, f, indent=4, ensure_ascii=False)

    return query_complete_json_answer




def check_online_search_and_generate_queries(claim, Video_information, QA_CONTEXTS, question, output_file_path):



    logging.warning("---------------------------------------------------------")
    logging.warning("--------- Does the question need online search? ---------")
    logging.warning("---------- Generating search terms in English -----------")
    logging.warning("---------------------------------------------------------")









    prompt_for_information_retrieving_verifier_and_queries_generation = f"""
{{
  "Claim": "{claim}",
  "Video_information": {json.dumps(Video_information)},
  "QA_CONTEXTS": {json.dumps(QA_CONTEXTS)},
  "New_Question": {{
    "Question": "{question}"
  }},
  "Task": "Determine if online information searching is needed to answer the new question. Consider the following criteria:

1. Online search is generally recommended for most questions to provide comprehensive and up-to-date information.
2. If the question can be fully and confidently answered using only the given 'Claim', 'Video_information', and 'QA_CONTEXTS', online search may not be necessary.
3. If the question specifically asks about details or content exclusively present in the video, online search may not be needed.
4. For any question that could benefit from additional context, background, or current information, online search is recommended.

Based on these criteria, determine if online search is needed (Yes or No).

If online search could potentially provide any useful information or context, even if not strictly necessary, choose 'Yes'. Only choose 'No' if the question is entirely about specific video content that cannot be found online.

If the result is 'Yes', generate two new queries for Google Custom Search API. If 'No', do not provide any queries.

Output Format:
**Analysis Process:**
If generating queries:
Generate the queries by following these steps, and output the result of each step. This will ensure that the output is well-considered:
Step 1: Generate the 'searching goals' to answer this question.
Step 2: Please generate 1 professional searching query for each goal used to perform a search using the Google Custom Search API. Based on the above 'searching goals'.
Generate output results for each step. This will ensure that the output is fully considered and well-structured.


"Prediction": {{
    "need_online_search": "",
    "reasoning": "Briefly explain the rationale behind the decision to search or not search online",
    "Queries": {{
      "Query1": {{
        "query": "Concise search query (max 10 words) for one aspect, suitable for Google Search",
        "searching_goal": "Provide detailed and specific explanations to clearly define the objectives that this query aims to achieve"
      }},
      "Query2": {{
        "query": "Different concise search query (max 10 words) for another aspect, suitable for Google Search",
        "searching_goal": "Provide detailed and specific explanations to clearly define the objectives that this query aims to achieve"
      }}
    }}
  }}
}}
Note: Both the two-step analysis process and the final JSON answer need to be outputted. The two-step analysis includes:
Step 1: Generate the 'searching goals' to answer this question.
Step 2: Please generate 1 professional searching query for each goal used to perform a search using the Google Custom Search API. Based on the above 'searching goals'.
"""






    logging.info("################## Online Search and Query Generation Input ##################")
    logging.info(prompt_for_information_retrieving_verifier_and_queries_generation)

    query_answer = gpt4o_mini_analysis(prompt_for_information_retrieving_verifier_and_queries_generation)

    logging.info("################## Online Search and Query Generation Raw Output ##################")
    logging.info(query_answer)



    prompt_for_query_format = f"""
Please convert the following text content into the specified JSON structure, preserving the original content as much as possible while ensuring the output conforms to the required format. The output must be valid JSON.

The desired JSON structure for when online search is needed:
{{
  "Prediction": {{
    "need_online_search": "Yes",
    "reasoning": "",
    "Queries": {{
      "Query1": {{
        "query": "Concise search query (max 10 words) for one aspect, suitable for Google Search",
        "searching_goal": "Provide detailed and specific explanations to clearly define the objectives that this query aims to achieve"
      }},
      "Query2": {{
        "query": "Different concise search query (max 10 words) for another aspect, suitable for Google Search",
        "searching_goal": "Provide detailed and specific explanations to clearly define the objectives that this query aims to achieve"
      }}
    }}
  }}
}}

The desired JSON structure for when no online search is needed:
{{
  "Prediction": {{
    "need_online_search": "No",
    "reasoning": ""
  }}
}}

Please ensure that:
1. The "need_online_search" value is either "Yes" or "No".
2. The "reasoning" field contains a brief explanation for the decision.
3. If "need_online_search" is "Yes", include both Query 1 and Query 2 with their respective "query" and "searching_goal".
4. The generated queries should be phrases suitable for Google Search.
5. The searching_goal should specify the objective the query aims to achieve.

The content to be converted:
{query_answer}
"""


    query_json_answer = gpt4o_mini_analysis(prompt_for_query_format)




    query_complete_json_answer = extract_complete_json(query_json_answer)


    with open(output_file_path, 'r', encoding='utf-8') as file:
        full_data = json.load(file)
        full_data.update(query_complete_json_answer)

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(full_data, f, indent=4, ensure_ascii=False)

    return query_complete_json_answer






def process_query_and_quality_score_value(query, key_terms, claim, Video_information, QA_CONTEXTS, question, output_file_path):
    logging.warning("-----------------------------------------------------")
    logging.warning("---------------- Google Search Terms ----------------")
    logging.warning("------------- Evaluating Website Quality ------------")
    logging.warning("-----------------------------------------------------")



    process_google_search(query, output_file_path)

    with open(output_file_path, 'r', encoding="utf-8") as file:
        data = json.load(file)

    display_links = []
    for evidence in data.get('items', []):
        for key, item in evidence.items():
            if isinstance(item, dict):
                display_link = item.get('displayLink')
                if display_link:
                    display_links.append(display_link)




    prompt = f"""
{{
"Claim": "{claim}",
"Video_information": {Video_information},
"QA_CONTEXTS": {QA_CONTEXTS},
"Relate_Website": {display_links},
"Task": "Based on the provided Claim, Video_information, and QA_CONTEXTS, evaluate the listed websites to determine which ones have high credibility in terms of truthfulness and accuracy, and can aid in detecting fake news. Please provide a quality score (website_quality) out of 10 for each website and explain the reasoning for the score. The evaluation criteria include the website's overall reliability, historical accuracy, and capability to detect and expose fake news.

Please combine the evaluations for these aspects to give an overall quality score (website_quality) for each website, and provide detailed explanations for each score."
}}
In your response, when referring to related websites, be sure to provide the original name of the specific and detailed website in Relate_Website, and do not modify this name.
It is required to rate and explain the reasons for all websites. Each website should be rated an overall quality score (website_quality) out of 10, with detailed explanations for each score.
"""





    max_iterations = 3
    iterations = 0

    while iterations < max_iterations:
        logging.info(f"################## Google Search Terms Input (Attempt {iterations}) ##################")
        logging.info(prompt)

        answer = gpt4o_mini_analysis(prompt)


        logging.info(f"################## Google Search Terms Output (Attempt {iterations}) ##################")
        logging.info(answer)

        prompt_for_format = f"""
    Please convert the following text content into JSON format. For each website, use the following format:
    {{
        "website": {{
        "website_qualityScore": "quality_score_value",
        "justification": "justification_text"
        }}
    }}

    Note: 
    - "quality_score_value" should be an integer between 0 and 10.
    - "justification" should be a string.

    Website represents the current website URL for rating and evaluation, rather than the word "website", preferably with a complete link via HTTP or HTTPS. For each website, quality_score_value, relevances_score_value, and newness_score_value are integers that represent the website's ratings in terms of quality, relevance, and novelty, ranging from 1 to 10. Justification_text is a string that provides reasons and explanations for the rating.
    The following is the text content that needs to be converted:
    {answer}
    """

        answer_format = gpt4o_mini_analysis(prompt_for_format)

        evaluation = extract_complete_json(answer_format)
        if not evaluation:
            logging.error("未能提取有效的JSON格式评价信息，重新获取GPT-3.5的分析结果。")
            iterations += 1
            continue 
        match_count = 0
        total_items = 0

        for evidence in data.get('items', []):
            total_items += 1  
            for key, item in evidence.items():
                if isinstance(item, dict):
                    display_link = item.get('displayLink')
                    if display_link:
                        best_match, ratio = find_best_match(display_link, evaluation)
                        if ratio > 0.6:  
                            item['website_quality_evaluation'] = evaluation[best_match]
                            match_count += 1

        if match_count == total_items:
            break  

        iterations += 1

    if iterations == max_iterations:
        logging.error("在最大尝试次数内未能成功匹配所有评价信息。")


    updated_single_query_path = output_file_path.replace(".json", "_updated.json")
    with open(updated_single_query_path, 'w', encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)











    

def process_newness(file_path):

    def extract_date_from_snippet(snippet):
        match = re.search(r'(\b\w+\b \d{1,2}, \d{4})', snippet)
        if match:
            try:
                date = parse(match.group(1), fuzzy=True).date()
                return date
            except ValueError:
                return None
        return None

    def extract_date_from_metatags(metatags):
        date_keys = ['article:published_time', 'sort_date']
        for tag in metatags:
            for key in date_keys:
                if key in tag:
                    date_str = tag[key]
                    try:
                        date = parse(date_str, fuzzy=True).date()
                        return date
                    except ValueError:
                        pass
        return None

    def extract_dates_from_items(items):
        evidence_dates = {}
        for item in items:
            for evidence_key, evidence in item.items():
                snippet = evidence.get('snippet', '')
                date_from_snippet = extract_date_from_snippet(snippet)
                if date_from_snippet:
                    evidence_dates[evidence_key] = date_from_snippet
                    continue

                pagemap = evidence.get('pagemap', {})
                metatags = pagemap.get('metatags', [])
                date_from_metatags = extract_date_from_metatags(metatags)
                if date_from_metatags:
                    evidence_dates[evidence_key] = date_from_metatags
                    continue

                evidence_dates[evidence_key] = None

        return evidence_dates

    def score_by_time_gradient(dates):
        now = datetime.now().date()
        gradients = [7, 15, 30, 90, 180, 365, 730]  
        scores = {}
        for evidence_key, date in dates.items():
            if date is None:
                scores[evidence_key] = 0
            else:
                diff_days = (now - date).days
                score = 1  
                for i, gradient in enumerate(gradients):
                    if diff_days <= gradient:
                        score = 10 - i
                        break
                scores[evidence_key] = score
        return scores

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        items = data.get('items', [])
        evidence_dates = extract_dates_from_items(items)
        scores = score_by_time_gradient(evidence_dates)

        for item in items:
            for evidence_key, evidence in item.items():
                date = evidence_dates.get(evidence_key, None)
                evidence['Newness'] = {
                    'NewnessScore': scores.get(evidence_key, 0),
                    'Date': date.isoformat() if date else 'No date found'
                }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"Processed {file_path} successfully.")
        return True

    except (json.JSONDecodeError, KeyError, IOError) as e:
        print(f"Failed to process {file_path}: {e}")
        return False




def process_evidence_and_Newness_Relevance(key, query, claim, Video_information, QA_CONTEXTS, question, updated_single_query_path):


    max_attempts = 3
    attempt = 0

    while attempt < max_attempts:
        attempt += 1

        process_newness(updated_single_query_path)
        
        evidence_json = process_evidence(updated_single_query_path)
    

        logging.warning("----------------------------------------------")
        logging.warning("--------- Evaluating Relevance Score ---------")
        logging.warning("----------------------------------------------")



        prompt_for_evidence_scores = f"""
{{
"Claim": "{claim}",
"Video_information": {json.dumps(Video_information)},
"QA_CONTEXTS": {json.dumps(QA_CONTEXTS)},
"New_Question": {{
    "Question": "{question}"
}},
"Current Evidence Searching queries on google": {{
    "{key}": "{query}"
}},
"Evidences": {json.dumps(evidence_json)},
"Task": "Based on the information of each evidence in 'Evidences', especially the content of website_content, evaluate the relevance of this evidence to the current question (RelevanceScore, 0~10 points, provide a score). Consider how closely the evidence addresses the specifics of the question '{question}', with a strong emphasis on how the evidence helps in determining whether the Claim and Video_information constitute false news. Evidence that significantly aids in judging the veracity of the Claim and Video_information should receive higher scores, while less relevant evidence should receive lower scores. The more the evidence helps in determining the truthfulness of the Claim and Video_information, the higher the RelevanceScore should be.

For each evidence (evidence0, evidence1, evidence2, evidence3, evidence4, evidence5, evidence6, evidence7, evidence8, evidence9), provide the following:
1. 'RelevanceScore': score, justification
Each evidence should include these details, specified as 'evidenceN' where N is the evidence number."
}}
"""



        complete_json_evidence_answer = {}
        expected_evidences = {f"evidence{i}" for i in range(10)}


        logging.info(f"################## Process RelevanceScore Input (Attempt {attempt}) ##################")
        logging.info(prompt_for_evidence_scores)

        evidence_scores = gpt4o_mini_analysis(prompt_for_evidence_scores)

        logging.info(f"################## Process RelevanceScore Output (Attempt {attempt}) ##################")
        logging.info(evidence_scores)



        format_prompt = f"""
Please convert the following text content into the specified JSON structure. Ensure the output is in JSON format and maintain the original content as much as possible, changing only the structure to the specified JSON format.

The desired JSON structure:
"evidenceN": {{
  "RelevanceScore": "score",
  "Relevance Justification": "justification"
}}

Note: 
- "score" should be an integer between 0 and 10.
- "justification" should be a string.

The content to be converted:
{evidence_scores}
"""

        json_evidence_answer = gpt4o_mini_analysis(format_prompt)


        new_evidence = extract_complete_json(json_evidence_answer)

        complete_json_evidence_answer.update(new_evidence)
        
        if expected_evidences.issubset(complete_json_evidence_answer.keys()):
            break

    with open(updated_single_query_path, 'r', encoding="utf-8") as file:
        data = json.load(file)

    for evidence_key, scores in complete_json_evidence_answer.items():
        for item in data['items']:
            if evidence_key in item:
                item[evidence_key]['Relevance'] = scores

    with open(updated_single_query_path, 'w', encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    logging.info("success updated_new_evidence")







def select_useful_evidence(claim, Video_information, QA_CONTEXTS, question, output_file_path):



    logging.warning("--------------------------------------------")
    logging.warning("------------ Evidence Selection ------------")
    logging.warning("----- Evaluating Usefulness of Evidence -----")
    logging.warning("--------------------------------------------")

    with open(output_file_path, 'r', encoding='utf-8') as file:
        new_data = json.load(file)

    all_transformed_evidence = {}

    all_evidences_with_scores = []

    for query_key in new_data:
        if query_key.startswith("Query"):  
            evidence_list = new_data[query_key]
            
            for i, evidence_dict in enumerate(evidence_list):
                quality_score = evidence_dict.get('website_quality_evaluation', {}).get('website_qualityScore', 0)
                newness_score = evidence_dict.get('Newness', {}).get('NewnessScore', 0)
                relevance_score = evidence_dict.get('Relevance', {}).get('RelevanceScore', 0)
                total_score = quality_score + newness_score + relevance_score * 2
                
                evidence_dict['total_score'] = total_score
                
                all_evidences_with_scores.append((query_key, i, evidence_dict, total_score))

    sorted_all_evidences = sorted(all_evidences_with_scores, key=lambda x: x[3], reverse=True)

    top_three_evidences = sorted_all_evidences[:3]

    for query_key, i, evidence_dict, total_score in top_three_evidences:
        new_key = f"{query_key}_evidence_{i + 1}"
        
        extracted_info = {
            "title": evidence_dict.get("title", ""),
            "link": evidence_dict.get("link", ""),
            "snippet": evidence_dict.get("snippet", ""),
            "content": evidence_dict.get("content", {}).get("content", "")
        }
        
        all_transformed_evidence[new_key] = extracted_info

    

    new_data["RelevantEvidence"] = all_transformed_evidence


    true_json_answer = None
    max_attempts = 3
    attempt = 0

    while attempt < max_attempts:
        attempt += 1

        validation_evidence_prompt = f"""
{{
"Claim": "{claim}",
"Video_information": {json.dumps(Video_information, indent=2)},
"QA_CONTEXTS": {json.dumps(QA_CONTEXTS, indent=2)},
"New_Question": {{
    "Question": "{question}"
}},
"Evidences": {json.dumps(all_transformed_evidence, indent=2)},
"Task": "Evaluate the provided evidence to determine its relevance and usefulness in addressing the New_Question and ultimately assessing the truthfulness of the Claim within the context of the Video_information. Consider the following:

1. Relevance: Does the evidence directly relate to the New_Question or the Claim?
2. Support or Refutation: Does the evidence support or refute the New_Question or Claim? Both supporting and refuting evidence can be useful.
3. Context: Does the evidence provide important context or background information?
4. Factual Content: Does the evidence contain factual information that can be used to evaluate the New_Question or Claim?
5. Source Credibility: If the source of the evidence is mentioned, is it from a reputable or relevant source?

Based on these criteria, determine if the evidence is useful. Even if the evidence contradicts the New_Question or Claim, it can still be considered useful if it's relevant to the overall assessment.

Output 'yes' if the evidence is useful (relevant and informative) or 'no' if it's not. Provide a detailed reason explaining your assessment, referencing specific aspects of the evidence that led to your conclusion.",
"EvidenceIsUseful": {{
    "Useful": "yes / no",
    "Reason": "Provide a detailed explanation here, referencing specific content from the evidence and how it relates to the New_Question and Claim."
}}
}}
"""



        logging.info(f"################## Validation Evidence Prompt (Attempt {attempt}) ##################")
        logging.info(validation_evidence_prompt)

        validation_evidence_answer = gpt4o_mini_analysis(validation_evidence_prompt)

        logging.info(f"################## Validation Evidence Answer (Attempt {attempt}) ##################")
        logging.info(validation_evidence_answer)

        prompt_for_format = f"""
Please convert the following text content into the specified JSON structure. Ensure the output is in JSON format and maintain the original content as much as possible, changing only the structure to the specified JSON format.

The desired JSON structure:
{{
    "EvidenceIsUseful": {{
        "Useful": "yes / no",
        "Reason": ""
    }}
}}

The content to be converted:
{validation_evidence_answer}
"""


        
        json_answer = gpt4o_mini_analysis(prompt_for_format)

        true_json_answer = extract_complete_json(json_answer)


        if true_json_answer and "EvidenceIsUseful" in true_json_answer and true_json_answer["EvidenceIsUseful"]["Useful"] in ["yes", "no"]:
            break
    

    if true_json_answer and true_json_answer.get("EvidenceIsUseful", {}).get("Useful") == "yes":
        with open(output_file_path, 'w', encoding='utf-8') as file:
            json.dump(new_data, file, indent=4, ensure_ascii=False)
        return True
    else:
        return False


    

















def process_claim_and_generate_answer(claim, Video_information, QA_CONTEXTS, question, output_file_path):

    
    logging.warning("----------------------------------------------------------")
    logging.warning("--------- Question Anser Model With GoogleSearch ---------")
    logging.warning("----------------------------------------------------------")



    with open(output_file_path, 'r', encoding='utf-8') as file:
        new_data = json.load(file)


    all_transformed_evidence = new_data.get("RelevantEvidence", {})


    queries_content = new_data.get("Queries", {})

    prompt_for_question_answer_based_on_evidence = f"""
    {{
      "Claim": "{claim}",
      "Video_information": {json.dumps(Video_information, ensure_ascii=False, indent=4)},
      "QA_CONTEXTS": {json.dumps(QA_CONTEXTS, ensure_ascii=False, indent=4)},
      "New_Question": {{
        "Question": "{question}"
      }},
      "Queries": {json.dumps(queries_content, ensure_ascii=False, indent=4)},
      "Good evidence information": {json.dumps(all_transformed_evidence, ensure_ascii=False, indent=4)},
      "Task": "Based on the evidence extracted, generate an explanatory answer to the question: '{question}' that references the evidence. Note to add the referenced evidence number after the argument for each reason, e.g., [Query 1_evidence1····]. And evaluate the confidence (XX%) of your answer based on the analysis of the above evaluation of the evidence and the logic of the reasoning process."
    }}
    """





    attempt = 0
    max_attempts = 5
    final_json_answer = None

    while attempt < max_attempts:
        attempt += 1
        logging.info(f"################## Question Answer Input (Attempt {attempt}) ##################")
        logging.info(prompt_for_question_answer_based_on_evidence)

        answer = gpt4o_mini_analysis(prompt_for_question_answer_based_on_evidence)

        

        logging.info(f"################## Question Answer Output (Attempt {attempt}) ##################")
        logging.info(answer)

        format_final_prompt = f"""
        Please convert the following text content into the specified JSON structure. Ensure the output is in JSON format and maintain the original content as much as possible, changing only the structure to the specified JSON format.

        Desired JSON structure:
        {{
          "QA": {{
            "Question": "{question}",
            "Answer": "",
            "Confidence": ""
          }}
        }}

        Please note: Only modify the structure of the given following content, keeping the content as intact as possible and preserving the original language and descriptions.

        Text content to be converted:
        "{answer}"
        The final output should be in JSON format, which includes the extracted content of the 'Answer' and the 'Confidence' of the non empty percentage.
        """


        final_answer = gpt4o_mini_analysis(format_final_prompt)

        final_json_answer = extract_complete_json(final_answer)


        if final_json_answer and final_json_answer.get("QA", {}).get("Answer") and final_json_answer.get("QA", {}).get("Confidence"):
            break 

    with open(output_file_path, 'r', encoding='utf-8') as file:
        full_data = json.load(file)
    full_data.update(final_json_answer)

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(full_data, f, indent=4, ensure_ascii=False)














def process_claim_and_generate_answer_without_gs(claim, Video_information, question, output_file_path):
    
    logging.warning("-------------------------------------------------------------")
    logging.warning("--------- Question Anser Model Without GoogleSearch ---------")
    logging.warning("-------------------------------------------------------------")



    prompt_for_question_answer = f"""
{{
    "Claim": "{claim}",
    "Video_information": {json.dumps(Video_information, ensure_ascii=False, indent=4)},
    "New_Question": {{
    "Question": "{question}"
    }},
    "Task": "Based on the Claim and the Video_information, generate an explanatory answer to the question: '{question}' that references the evidence.  And evaluate the confidence (XX%) of your answer based on the analysis of the logic of the reasoning process."
}}
"""



    attempt = 0
    max_attempts = 5
    final_json_answer = None

    while attempt < max_attempts:
        attempt += 1
        
        logging.info("prompt_for_question_answer")
        logging.info(prompt_for_question_answer)

        answer = gpt4o_mini_analysis(prompt_for_question_answer)        

        logging.info("prompt_for_question_answer ANSWER")
        logging.info(answer)

        format_final_prompt = f"""
Please convert the following text content into the specified JSON structure. Ensure the output is in JSON format and maintain the original content as much as possible, changing only the structure to the specified JSON format.

Desired JSON structure:
{{
    "QA": {{
    "Question": "{question}",
    "Answer": "",
    "Confidence": ""
    }}
}}

Please note: Only modify the structure of the given following content, keeping the content as intact as possible and preserving the original language and descriptions.

Text content to be converted:
"{answer}"
The final output should be in JSON format, which includes the extracted content of the 'Answer' and the 'Confidence' of the non empty percentage.
"""


        final_answer = gpt4o_mini_analysis(format_final_prompt)

        final_json_answer = extract_complete_json(final_answer)

        logging.info("QA Model final_json_answer")
        logging.info(final_json_answer)

        if final_json_answer and final_json_answer.get("QA", {}).get("Answer") and final_json_answer.get("QA", {}).get("Confidence"):
            break

    with open(output_file_path, 'r', encoding='utf-8') as file:
        full_data = json.load(file)
    full_data.update(final_json_answer)

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(full_data, f, indent=4, ensure_ascii=False)










    











def process_question_VideoLLM(claim, Video_information, question, output_file_path, video_id):
    logging.warning("---------------------------------------------------------------")
    logging.warning("--------- Question Answer Model With LMM MiniCPM-V 2.6 --------")
    logging.warning("---------------------------------------------------------------")

    prompt_for_question_answer = f"""
{{
  "Task": "Rewrite the given question as a direct video-specific inquiry",
  "Input": {{
    "Original_Question": "{question}",
    "Claim": "{claim}",
    "Video_information": {json.dumps(Video_information)}
  }},
  "Rewriting_Guidelines": [
    "1. Focus exclusively on video content",
    "2. Begin with 'In the video,' or similar phrases",
    "3. Use simple, clear language",
    "4. Limit to 30 words maximum",
    "5. Ensure question is answerable from video alone",
    "6. Remove references to 'Claim' or external information"
  ],
  "Example_Transformations": [
    {{
      "Before": "Is the quote attributed to Cher in the Claim accurately represented in the video transcript?",
      "After": "In the video, what exact statements or quotes, if any, are made by Cher?"
    }},
    {{
      "Before": "Does the video provide any evidence to support or refute the Claim's assertion about climate change?",
      "After": "What specific information or data does the video present about climate change?"
    }}
  ],
  "Output_Instruction": "Provide only the rewritten question. No explanations or additional text."
}}
"""
    
    new_question = gpt4o_mini_analysis(prompt_for_question_answer)

    logging.info("---------- original question ----------")
    logging.info(question)

    logging.info("---------- new_question ----------")
    logging.info(new_question)

    video_folder = r"/home/public/FakeNews/code/NKP/LLMFND/select_all_videos"
    video_path = os.path.join(video_folder, f"{video_id}.mp4")



    answer = analysis_video_minicpm(video_path, new_question)

    logging.info("--- VideoLLM answer ---")
    logging.info(answer)

    final_json_answer = {
        "QA": {
            "Original_Question": question,
            "Rewritten_Question": new_question,
            "Answer": answer,
        }
    }

    if os.path.exists(output_file_path):
        with open(output_file_path, 'r', encoding='utf-8') as file:
            full_data = json.load(file)
    else:
        full_data = {}

    full_data.update(final_json_answer)

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(full_data, f, indent=4, ensure_ascii=False)


    




















def process_evidence(file_path):


    logging.warning("\n" * 5)
    logging.warning("--------------------------------------")
    logging.warning("--------- Processing evidence ---------")
    logging.warning("--------- Extracting necessary content from Google search results ---------")
    logging.warning("--------------------------------------")



    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    items = data.get("items", [])

    top_items = items[:10]

    evidence_found_and_judgments = []
    for i, item_dict in enumerate(top_items):
        for key, item in item_dict.items():
            evidence = {
                "title": item.get("title"),
                "link": item.get("link"),
                "displayLink": item.get("displayLink"),
                "snippet": item.get("snippet"),
                "htmlSnippet": item.get("htmlSnippet"),
                "website_content": item.get("website_content", {}).get("content")
            }
            evidence_found_and_judgments.append({f'evidence{i}': evidence})

    output_json = json.dumps(evidence_found_and_judgments, indent=4, ensure_ascii=False)

    return output_json












user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
]

def fetch_webpage_content_bs4(link, retries=1):
    headers = {
        'User-Agent': random.choice(user_agents),
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Upgrade-Insecure-Requests': '1'
    }
    
    session = requests.Session()
    session.headers.update(headers)

    for attempt in range(retries):
        try:
            response = session.get(link)
            response.raise_for_status()
            response.encoding = 'utf-8' 
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            content = '\n'.join([para.get_text() for para in paragraphs])

            content = content.encode('utf-8', errors='replace').decode('utf-8')

            return {"success": True, "content": content}
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                time.sleep(3)  
            else:
                return {"success": False, "error": f"Error fetching {link}: {str(e)}"}

def fetch_webpage_content_selenium(link):
    options = webdriver.FirefoxOptions()
    options.add_argument('--headless')  
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920x1080')
    options.set_preference("general.useragent.override", random.choice(user_agents))
    
    driver = None
    try:
        driver_path = '/usr/local/bin/geckodriver' 
        service = FirefoxService(executable_path=driver_path)
        driver = webdriver.Firefox(service=service, options=options)
        
        driver.get(link)
        
        SCROLL_PAUSE_TIME = 2
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(SCROLL_PAUSE_TIME)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        
        WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.TAG_NAME, 'p')))
        
        paragraphs = driver.find_elements(By.TAG_NAME, 'p')
        content = '\n'.join([para.text for para in paragraphs])
        
        return {"success": True, "content": content}
    except Exception as e:
        return {"success": False, "error": f"Error fetching {link}: {str(e)}"}
    finally:
        if driver:
            driver.quit()







class TooManyRequestsError(Exception):
    """Custom exception for too many requests error."""
    pass

def readAPI_fetch_content(url):
    api_key = None

    def fetch(url, headers):
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        if 'application/json' in response.headers.get('Content-Type'):
            return response.json()
        else:
            raise requests.exceptions.ContentTypeError(
                f"Unexpected content type: {response.headers.get('Content-Type')}"
            )

    def remove_unwanted_text(text):
        url_pattern = re.compile(r'\(https?://[^\)]+\)')
        mailto_pattern = re.compile(r'\(mailto:[^\)]+\)')
        brackets_pattern = re.compile(r'\[.*?\]')
        text = url_pattern.sub('', text)
        text = mailto_pattern.sub('', text)
        text = brackets_pattern.sub('', text)
        text = '\n'.join([line for line in text.split('\n') if line.strip()])
        return text

    headers_common = {
        "Accept": "application/json",
    }

    if api_key:
        headers_common["Authorization"] = f"Bearer {api_key}"

    url1 = f"https://r.jina.ai/{url}"
    retries = 3
    delay = 60  # seconds

    result = {
        'success': False,
        'error': f"Failed to fetch {url} after {retries} attempts"
    }

    for attempt in range(retries):
        logging.info(f"readerAPI Process {url}, attempt {attempt + 1}")
        try:
            response_default = fetch(url1, headers_common)
            default_content = response_default.get('data').get('content')
            clean_default_content = remove_unwanted_text(default_content)

            result = {
                'success': True,
                'content': clean_default_content,
            }
            break
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logging.warning(f"429 Too Many Requests for url: {url}. Retrying in {delay} seconds...")
                time.sleep(delay)
                if attempt == retries - 1:
                    logging.error(f"Failed to fetch {url} due to repeated 429 errors after {retries} attempts. Exiting.")
                    sys.exit(1) 
            else:
                result = {
                    'success': False,
                    'error': f"Error fetching {url}: {str(e)}"
                }
                break
        except Exception as e:
            result = {
                'success': False,
                'error': f"Error fetching {url}: {str(e)}"
            }
            break

    return result















def fetch_webpage_content_trafilatura(link, retries=1):
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
    ]

    headers = {
        'User-Agent': random.choice(user_agents),
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Upgrade-Insecure-Requests': '1'
    }

    for attempt in range(retries):
        try:
            downloaded = trafilatura.fetch_url(link, headers=headers)
            
            if downloaded:
                content = trafilatura.extract(downloaded)

                content = content.encode('utf-8', errors='replace').decode('utf-8')

                return {"success": True, "content": content}
            else:
                raise Exception("Failed to download the webpage")
        
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3)  
            else:
                return {"success": False, "error": f"Error fetching {link}: {str(e)}"}

























def fetch_webpage_content(link, retries=2):
    results = [None, None, None, None]

    def run_bs4():
        results[0] = fetch_webpage_content_bs4(link, retries)
        # logging.info(f"BS4 result: {results[0]}")
        # logging.info(f"BS4 result: {results[0].get('success')}")

    def run_selenium():
        results[1] = fetch_webpage_content_selenium(link)
        # logging.info(f"Selenium result: {results[1]}")
        # logging.info(f"Selenium result: {results[1].get('success')}")

    def run_readapi():
        results[2] = readAPI_fetch_content(link)
        # logging.info(f"ReadAPI result: {results[2]}")
        # logging.info(f"ReadAPI result: {results[2].get('success')}")

    def run_trafilatura():
        results[3] = fetch_webpage_content_trafilatura(link)

    threads = [
        # threading.Thread(target=run_bs4),
        # threading.Thread(target=run_selenium),
        threading.Thread(target=run_readapi),
        threading.Thread(target=run_trafilatura),
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


    valid_results = [result for result in results if result and result.get("content")]
    if valid_results:
        valid_results.sort(key=lambda x: len(x.get("content", "")), reverse=True)
        return valid_results[0]

    return results[-1] if results[-1] else {"success": False, "content": ""}













def count_tokens(text, model_name='gpt-3.5-turbo'):
    encoding = tiktoken.encoding_for_model(model_name)
    
    tokens = encoding.encode(text)
    
    return len(tokens)







def modified_final_evidence(evidence):
    json_evidence = evidence

    for query_key, evidences in json_evidence.items():
        if query_key.startswith("Query") and isinstance(evidences, dict):
            for evidence_key, value in evidences.items():
                if 'website_quality_evaluation' in value:
                    del value['website_quality_evaluation']

                link = value['link']
                content_result = fetch_webpage_content(link)
                if content_result["success"]:
                    content = content_result["content"]
                    if count_tokens(content) > 4000:
                        content = ' '.join(content.split()[:4000]) + " ... [Content truncated]"
                    value['complete_content'] = content
                else:
                    value['complete_content'] = content_result["error"]

    return json_evidence







def get_content_and_word_count(link, snippet):

    content_result = fetch_webpage_content(link)
    if content_result["success"]:
        content = content_result["content"]
        content_tokens = count_tokens(content)

        if content_tokens < 50:
            content = f"Error: Content too short. Original content: {content}"
            content_tokens = 0
        elif content_tokens > 500:
            content = extract_surrounding_text(content, snippet)
            logging.info(f"Link: {link} \t Content tokens: {content_tokens}")
            content_tokens = 500
    else:
        content = content_result["error"]
        content_tokens = 0
    
    return content, content_tokens
















def extract_surrounding_text(content, snippet, num_tokens=250):
    start_time = time.time()

    def count_tokens(text, model_name='gpt-3.5-turbo'):
        encoding = tiktoken.encoding_for_model(model_name)
        tokens = encoding.encode(text)
        return len(tokens)

    def encode_text(text, model_name='gpt-3.5-turbo'):
        encoding = tiktoken.encoding_for_model(model_name)
        tokens = encoding.encode(text)
        return tokens, encoding

    def get_surrounding_tokens(content, snippet, num_tokens=INFORMATION_RETRIEVER_CONFIG.get("num_tokens", 250)):
 
        en_core_web_sm_path = MODEL_CONFIG["en_core_web_sm"]["local_model_path"]
        
        nlp = spacy.load(en_core_web_sm_path)

        max_length = 30000
        if len(content) > max_length:
            content = content[:max_length]

        content_doc = nlp(content)

        sentences = [sent.text for sent in content_doc.sents]

        snippet_doc = nlp(snippet)

        def calculate_similarity(snippet_doc, sentences):
            vectorizer = CountVectorizer().fit_transform([snippet_doc.text] + sentences)
            vectors = vectorizer.toarray()
            cosine_matrix = cosine_similarity(vectors)
            similarities = cosine_matrix[0][1:]  
            return similarities

        similarities = calculate_similarity(snippet_doc, sentences)

        best_sentence_index = np.argmax(similarities)

        best_sentence = sentences[best_sentence_index]

        target_start_index = content_doc.text.find(best_sentence)
        target_end_index = target_start_index + len(best_sentence)
        
        target_start_token_index = None
        target_end_token_index = None
        
        for token in content_doc:
            if token.idx == target_start_index:
                target_start_token_index = token.i
            if token.idx + len(token.text) - 1 == target_end_index - 1:
                target_end_token_index = token.i

        if target_start_token_index is None or target_end_token_index is None:
            return ""
        
        all_tokens, encoding = encode_text(content_doc.text)

        start_token_index = max(0, target_start_token_index - num_tokens)
        end_token_index = min(len(all_tokens), target_end_token_index + num_tokens + 1)

        prefix = "" if start_token_index == 0 else "[Content truncated]..."
        suffix = "" if end_token_index == len(all_tokens) else "...[Content truncated]"

        surrounding_tokens = all_tokens[start_token_index:end_token_index]
        surrounding_text = prefix + encoding.decode(surrounding_tokens) + suffix
        return surrounding_text

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(get_surrounding_tokens, content, snippet, num_tokens)
        try:
            surrounding_text = future.result(timeout=60)
        except concurrent.futures.TimeoutError:
            surrounding_text = encode_text(content[:500])[1].decode(encode_text(content[:500])[0])

    return surrounding_text




















def contains_fact_check(link):

    pattern = re.compile(r'fact[\W_]*check', re.IGNORECASE)
    return bool(pattern.search(link))


def process_json_files(folder_path, output_file_path):
    logging.info(f"Processing JSON files in folder: {folder_path}")
    output_data = {}

    files = [f for f in os.listdir(folder_path) if f.startswith("Query") and f.endswith("_updated.json")]

    files.sort(key=lambda f: int(re.search(r'Query(\d+)', f).group(1)))

    for filename in files:
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            evidences = []
            for item in data.get("items", []):
                for key, evidence in item.items():
                    if evidence.get("website_content", {}).get("content_tokens", 0) != 0:
                        try:
                            quality_score = int(evidence["website_quality_evaluation"].get("website_qualityScore", 0))
                            newness_score = int(evidence["Newness"].get("NewnessScore", 0))
                            relevance_score = int(evidence["Relevance"].get("RelevanceScore", 0))
                            # total_score = quality_score + newness_score + relevance_score*2


                            weights = INFORMATION_RETRIEVER_CONFIG["scoring_weights"]
                            website_quality_weight = weights["website_quality"]  # 0.25
                            newness_weight = weights["newness"]                 # 0.25
                            relevance_weight = weights["relevance"]             # 0.50

                            total_score = (quality_score * website_quality_weight + 
                                        newness_score * newness_weight + 
                                        relevance_score * relevance_weight)

                            if not contains_fact_check(evidence["link"]):
                                evidences.append({
                                    "title": evidence["title"],
                                    "link": evidence["link"],
                                    "snippet": evidence["snippet"],
                                    "content": evidence["website_content"],
                                    "website_quality_evaluation": evidence["website_quality_evaluation"],
                                    "Newness": evidence["Newness"],
                                    "Relevance": evidence["Relevance"],
                                    "total_score": total_score  
                                })
                        except KeyError as e:
                            logging.info(f"Error processing evidence: {evidence}")
                            logging.info(f"KeyError: {e}")

            max_num_evidences = INFORMATION_RETRIEVER_CONFIG["max_num_evidences"]
            top_evidences = sorted(evidences, key=lambda x: x["total_score"], reverse=True)[:max_num_evidences]
            for evidence in top_evidences:
                del evidence["total_score"]

            query_key = filename.replace('_updated.json', '')
            output_data[query_key] = top_evidences

    if os.path.exists(output_file_path):
        with open(output_file_path, 'r', encoding='utf-8') as output_file:
            existing_data = json.load(output_file)
    else:
        existing_data = {}

    existing_data.update(output_data)

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(existing_data, output_file, ensure_ascii=False, indent=4)

    print(f"结果已追加写入 {output_file_path}")


