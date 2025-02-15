import logging
import json
import os
import time
from typing import Dict, Any


from .Comprehensiveness import evaluate_comprehensiveness
from .conciseness import evaluate_conciseness
from .currency import evaluate_currency
from .Fact_Hallucination import evaluate_fact_hallucination
from .faithfulness import evaluate_faithfulness
from .logical_consistency import evaluate_logical_consistency
from .strength_of_evidence import evaluate_strength_of_evidence
from .Reasons_Evidence_Omission import evaluate_Omission_of_Reasons_and_Evidence
from .bleu_rouge import compute_rouge_bleu




def evaluate_content_credibility(json_path: str) -> Dict[str, Any]:

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        document_content = {
            "Claim": data['Claim'],
            "Video_information": {
                "video_date": data['Video_information']['video_date'],
                "platform": data['Video_information']['platform'],
                "video_headline": data['Video_information']['video_headline'],
                "video_transcript": data['Video_information']['video_transcript']
            },
            "Final_Judgement": {
                "Answer": data['Final_Judgement']['Answer'],
                "Reasons": data['Final_Judgement']['Reasons']
            }
        }
        
        all_evidences_content = {
            "Evidences": data['Evidences']
        }
        
        logging.info(f"Document Content: \n{document_content}")
        logging.info(f"All Evidences Content: \n{all_evidences_content}")
        
        content_credibility = {
            "comprehensiveness": evaluate_comprehensiveness(document_content, all_evidences_content),
            "conciseness": evaluate_conciseness(document_content, all_evidences_content),
            "currency": evaluate_currency(document_content, all_evidences_content),
            "fact_hallucination": evaluate_fact_hallucination(document_content, all_evidences_content),
            "faithfulness": evaluate_faithfulness(document_content, all_evidences_content),
            "logical_consistency": evaluate_logical_consistency(document_content, all_evidences_content),
            "strength_of_evidence": evaluate_strength_of_evidence(document_content, all_evidences_content)
        }
        
        data['content_credibility'] = content_credibility
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        print("Content credibility added successfully.")
        return {"document_content": document_content, "all_evidences_content": all_evidences_content}
        
    except Exception as e:
        logging.error(f"Error in evaluate_content_credibility: {str(e)}")
        raise










def evaluate_ground_truth(target_gt_path: str, json_path: str) -> None:

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        video_id = os.path.basename(json_path).split('_')[0]
        gt_file_path = os.path.join(target_gt_path, f"{video_id}.json")
        
        document_content = {
            "Claim": data['Claim'],
            "Video_information": data['Video_information'],
            "Final_Judgement": data['Final_Judgement']
        }
        all_evidences_content = {"Evidences": data['Evidences']}
        
        def flatten_dict_values(d: Dict) -> str:
            """Flatten nested dictionary values into a single string"""
            values = []
            for value in d.values():
                if isinstance(value, dict):
                    values.append(flatten_dict_values(value))
                else:
                    values.append(str(value))
            return " ".join(values)
        
        if os.path.exists(gt_file_path):
            with open(gt_file_path, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
                ground_truth_content = {
                    "original_rationales": json_data.get('original_rationales', {}),
                    "summary_rationales": json_data.get('summary_rationales', {}),
                    "evidences": json_data.get('evidences', {}),
                    "relationship_with_evidence": json_data.get('relationship_with_evidence', [])
                }
                logging.info(f"GroundTruth Content: \n{ground_truth_content}")
        
            reasons_str = document_content["Final_Judgement"]["Reasons"]
            logging.info("-" * 50)
            
            original_rationales_str = flatten_dict_values(ground_truth_content["original_rationales"])
            summary_rationales_str = flatten_dict_values(ground_truth_content["summary_rationales"])
            
            logging.info(f"Reasons: \n{reasons_str}")
            logging.info(f"Original Rationales: \n{original_rationales_str}")
            logging.info(f"Summary Rationales: \n{summary_rationales_str}")
            
            original_scores = compute_rouge_bleu(original_rationales_str, reasons_str)
            summary_scores = compute_rouge_bleu(summary_rationales_str, reasons_str)
            
            data['comparison_with_ground_truth'] = {
                "reasons_vs_original_rationales": original_scores,
                "reasons_vs_summary_rationales": summary_scores
            }
            
            comparison_with_gt_score = evaluate_Omission_of_Reasons_and_Evidence(
                document_content, all_evidences_content, ground_truth_content["summary_rationales"]
            )
            data['llm_comparison_with_gt_score'] = comparison_with_gt_score
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print("GroundTruth content added successfully.")
            
        else:
            print(f"Ground truth file not found: {gt_file_path}")
            
    except Exception as e:
        logging.error(f"Error in evaluate_ground_truth: {str(e)}")
        raise







def evaluate_folder(json_folder_path, target_gt_path):
    for filename in os.listdir(json_folder_path):
        if filename.endswith('.json'):
            json_path = os.path.join(json_folder_path, filename)
            evaluate_content_credibility(json_path)
            evaluate_ground_truth(target_gt_path, json_path)





import os
import json

def extract_metrics(data):
    metrics = {}
    
    content_credibility = {}
    content_credibility['comprehensiveness'] = data['content_credibility']['comprehensiveness']['Comprehensiveness']['Score']
    content_credibility['conciseness'] = data['content_credibility']['conciseness']['Conciseness']['Score']
    content_credibility['currency'] = data['content_credibility']['currency']['Currency']['Score']
    content_credibility['fact_hallucination'] = data['content_credibility']['fact_hallucination']['Fact_Hallucination']['Score']
    content_credibility['faithfulness'] = data['content_credibility']['faithfulness']['Faithfulness']['Score']
    content_credibility['logical_consistency'] = data['content_credibility']['logical_consistency']['Logical_Consistency']['Score']
    content_credibility['strength_of_evidence'] = data['content_credibility']['strength_of_evidence']['Strength_of_Evidence']['Score']
    metrics['content_credibility'] = content_credibility
    
    comparison_ground_truth = {}
    comparison_ground_truth['ROUGE-1'] = data['comparison_with_ground_truth']['reasons_vs_original_rationales']['ROUGE-1']
    comparison_ground_truth['ROUGE-2'] = data['comparison_with_ground_truth']['reasons_vs_original_rationales']['ROUGE-2']
    comparison_ground_truth['ROUGE-L'] = data['comparison_with_ground_truth']['reasons_vs_original_rationales']['ROUGE-L']
    comparison_ground_truth['BLEU-1'] = data['comparison_with_ground_truth']['reasons_vs_original_rationales']['BLEU-1']
    comparison_ground_truth['BLEU-2'] = data['comparison_with_ground_truth']['reasons_vs_original_rationales']['BLEU-2']
    comparison_ground_truth['BLEU-3'] = data['comparison_with_ground_truth']['reasons_vs_original_rationales']['BLEU-3']
    comparison_ground_truth['BLEU-4'] = data['comparison_with_ground_truth']['reasons_vs_original_rationales']['BLEU-4']
    metrics['comparison_with_ground_truth'] = comparison_ground_truth
    
    llm_comparison = {}
    llm_comparison['Omission_of_Reasons_and_Evidence'] = data['llm_comparison_with_gt_score']['Omission_of_Reasons_and_Evidence']['Score']
    metrics['llm_comparison_with_gt_score'] = llm_comparison
    
    return metrics

def calculate_average_eval_scores(json_folder_path):
    all_metrics = {
        "content_credibility": {
            "comprehensiveness": [],
            "conciseness": [],
            "currency": [],
            "fact_hallucination": [],
            "faithfulness": [],
            "logical_consistency": [],
            "strength_of_evidence": []
        },
        "comparison_with_ground_truth": {
            "ROUGE-1": [],
            "ROUGE-2": [],
            "ROUGE-L": [],
            "BLEU-1": [],
            "BLEU-2": [],
            "BLEU-3": [],
            "BLEU-4": []
        },
        "llm_comparison_with_gt_score": {
            "Omission_of_Reasons_and_Evidence": []
        }
    }
   
    for filename in os.listdir(json_folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(json_folder_path, filename)
           
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                extracted_metrics = extract_metrics(data)
                
                # 收集评价指标
                for category, metrics in extracted_metrics.items():
                    for metric_name, value in metrics.items():
                        all_metrics[category][metric_name].append(value)
   
    average_scores = {
        category: {
            metric: round(sum(values) / len(values), 4) if values else 0
            for metric, values in metrics.items()
        }
        for category, metrics in all_metrics.items()
    }
   
    output_path = os.path.join(json_folder_path, 'average_eval_scores.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(average_scores, f, indent=4, ensure_ascii=False)
   
    print(f"平均评价指标已保存到 {output_path}")
    return average_scores





