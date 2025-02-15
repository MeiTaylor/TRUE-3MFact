
import os
import json
import shutil

def extract_evidence(subfolder_path):
    evidence_summary = {}

    ir_result_path = os.path.join(subfolder_path, 'IR_result.json')

    if os.path.exists(ir_result_path):
        with open(ir_result_path, 'r', encoding='utf-8') as f:
            ir_result = json.load(f)
            
            relevant_evidence = ir_result.get('RelevantEvidence', {})

            subfolder_name = os.path.basename(subfolder_path)
            evidence_summary[subfolder_name] = relevant_evidence
    
    return evidence_summary

def save_to_cv_result(json_data, output_file):
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    else:
        existing_data = {}

    existing_data.update(json_data)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)

def process_subfolders(folder_path):
    evidence_summary = {}
    
    for subfolder_name in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder_name)
        
        if os.path.isdir(subfolder_path):
            subfolder_evidence = extract_evidence(subfolder_path)
            evidence_summary.update(subfolder_evidence)
    
    return {"Evidences": evidence_summary}

def process_all_folders(base_path):
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        
        if os.path.isdir(folder_path):
            output_file = os.path.join(folder_path, f'{folder_name}_CV_result.json')

            evidence_data = process_subfolders(folder_path)

            save_to_cv_result(evidence_data, output_file)

            print(f"Evidence数据已成功保存到 {output_file}")










def extract_cv_result_files(base_path, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        
        if os.path.isdir(folder_path):
            cv_result_file = os.path.join(folder_path, f'{folder_name}_CV_result.json')

            if os.path.exists(cv_result_file):
                destination_file_path = os.path.join(destination_folder, f'{folder_name}_CV_result.json')

                shutil.copy(cv_result_file, destination_file_path)

                print(f"{cv_result_file} 已复制到 {destination_file_path}")

