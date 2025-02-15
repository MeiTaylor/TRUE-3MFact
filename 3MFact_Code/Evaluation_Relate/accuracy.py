import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from prettytable import PrettyTable



def map_rating_to_binary(rating):
    """Maps the rating to binary True/False based on the given categories"""
    true_categories = ["True", "Mostly True", "Correct Attribution"]
    false_categories = ["False", "Miscaptioned", "Mixture", "Fake", "Mostly False"]
    
    if rating in true_categories:
        return "True"
    elif rating in false_categories:
        return "False"
    else:
        return None




def calculate_acc_metrics(test_folder, cv_result_folder):
    
    y_true = []
    y_pred = []
    
    for filename in os.listdir(cv_result_folder):
        if filename.endswith('_CV_result.json'):
            video_id = filename.replace('_CV_result.json', '')
            original_file = os.path.join(test_folder, f"{video_id}.json")
            cv_file = os.path.join(cv_result_folder, filename)
            
            if os.path.exists(original_file) and os.path.exists(cv_file):
                try:
                    with open(original_file, 'r', encoding='utf-8') as f:
                        original_data = json.load(f)
                        rating = original_data.get('rating')
                        true_label = map_rating_to_binary(rating)
                    
                    with open(cv_file, 'r', encoding='utf-8') as f:
                        cv_data = json.load(f)
                        pred_label = cv_data.get('Final_Judgement', {}).get('Answer')
                    
                    if true_label and pred_label:
                        y_true.append(true_label)
                        y_pred.append(pred_label)
                
                except json.JSONDecodeError:
                    print(f"Error reading JSON from file: {filename}")
                except Exception as e:
                    print(f"Error processing file {filename}: {str(e)}")
    
    if y_true and y_pred:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label="True")
        recall = recall_score(y_true, y_pred, pos_label="True")
        f1 = f1_score(y_true, y_pred, pos_label="True")
        conf_matrix = confusion_matrix(y_true, y_pred, labels=["True", "False"])
        
        total_true = sum(1 for label in y_true if label == "True")
        total_false = sum(1 for label in y_true if label == "False")
        
        table = PrettyTable()
        table.field_names = ["Metric", "Value"]
        table.add_row(["Total Files Processed", len(y_true)])
        table.add_row(["Total True Labels", total_true])
        table.add_row(["Total False Labels", total_false])
        table.add_row(["Accuracy", f"{accuracy:.2%}"])
        table.add_row(["Precision", f"{precision:.2%}"])
        table.add_row(["Recall", f"{recall:.2%}"])
        table.add_row(["F1 Score", f"{f1:.2%}"])
        
        table.add_row(["", ""])
        table.add_row(["Confusion Matrix", ""])
        table.add_row(["True Negative", conf_matrix[1][1]])
        table.add_row(["False Positive", conf_matrix[1][0]])
        table.add_row(["False Negative", conf_matrix[0][1]])
        table.add_row(["True Positive", conf_matrix[0][0]])
        
        return table
    
    return None









