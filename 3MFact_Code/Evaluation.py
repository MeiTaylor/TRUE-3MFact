import os
import logging

from Evaluation_Relate.accuracy import calculate_acc_metrics
from Evaluation_Relate.pre import process_all_folders, extract_cv_result_files
from Evaluation_Relate.run_evaluation import evaluate_folder, calculate_average_eval_scores
from Config import DATASET_CONFIG









if __name__ == "__main__":


    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='Evaluation.log',
        filemode='a'
    )




    root_dir = DATASET_CONFIG["root_dir"]
    test_folder = os.path.join(root_dir, DATASET_CONFIG["data_dir"]["test"])


    cv_result_folder = os.path.join(root_dir, DATASET_CONFIG["output_dir"]["test"])

    evaluation_folder = os.path.join(root_dir, DATASET_CONFIG["evaluation_output_dir"]["test"])
    


    process_all_folders(cv_result_folder)

    extract_cv_result_files(cv_result_folder, evaluation_folder)



    results = calculate_acc_metrics(test_folder, evaluation_folder)
    if results:

        print(results)
        logging.info("Acc Metrics Results:")
        logging.info(str(results))  
    else:
        print("No valid comparisons could be made.")


    try:

        evaluate_folder(evaluation_folder, test_folder)

        average_scores = calculate_average_eval_scores(evaluation_folder)

        logging.info("Average Evaluation Scores:")
        logging.info(average_scores)

        
       
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
