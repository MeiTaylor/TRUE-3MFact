import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu
import json

import requests


import time
import logging
import pytz
from datetime import datetime
import traceback

from Config import MODEL_CONFIG, DATASET_CONFIG, VIDEO_DESCRIPTOR_CONFIG

model_path = MODEL_CONFIG['video_lmm']['local_model_path']

# Load model and tokenizer once
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, 
                                  attn_implementation='sdpa', torch_dtype=torch.bfloat16) 
model = model.eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

MAX_NUM_FRAMES = 64  # if cuda OOM set a smaller number

def uniform_sample(l, n):
    gap = len(l) / n
    idxs = [int(i * gap + gap / 2) for i in range(n)]
    return [l[i] for i in idxs]

def encode_video(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    return frames

def process_image(image_path, question):
    image = Image.open(image_path).convert('RGB')
    msgs = [{'role': 'user', 'content': [image, question]}]

    # Process the image
    answer = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer
    )

    return answer

def process_video(video_path, question):
    frames = encode_video(video_path)
    msgs = [{'role': 'user', 'content': frames + [question]}]
    
    # Set decode params for video
    params = {
        "use_image_id": False,
        "max_slice_nums": 1  # use 1 if cuda OOM and video resolution > 448*448
    }

    # Process the video
    answer = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer,
        **params
    )
    
    return answer


























import os
import cv2
import numpy as np
from typing import List
import glob

import logging




class Time:
    def __init__(self, milliseconds: float):
        self.second, self.millisecond = divmod(milliseconds, 1000)
        self.minute, self.second = divmod(self.second, 60)
        self.hour, self.minute = divmod(self.minute, 60)

    def __str__(self):
        return f'{str(int(self.hour)) + "h-" if self.hour else ""}' \
               f'{str(int(self.minute)) + "m-" if self.minute else ""}' \
               f'{str(int(self.second)) + "s-" if self.second else ""}' \
               f'{str(int(self.millisecond)) + "ms"}'

class Frame:
    def __init__(self, no: int, hist: List):
        self.no = no
        self.hist = hist

class FrameCluster:
    def __init__(self, cluster: List[Frame], center: Frame):
        self.cluster = cluster
        self.center = center

    def re_center(self):
        hist_sum = [0] * len(self.cluster[0].hist)
        for i in range(len(self.cluster[0].hist)):
            for j in range(len(self.cluster)):
                hist_sum[i] += self.cluster[j].hist[i]
        self.center.hist = [i / len(self.cluster) for i in hist_sum]

    def keyframe_no(self) -> int:
        no = self.cluster[0].no
        max_similar = 0
        for frame in self.cluster:
            similar = similarity(frame.hist, self.center.hist)
            if similar > max_similar:
                max_similar, no = similar, frame.no
        return no

def similarity(frame1, frame2):
    s = np.vstack((frame1, frame2)).min(axis=0)
    similar = np.sum(s)
    return similar



def extract_keyframes_with_binary_search(video_path: str, min_keyframes=4, max_keyframes=6, min_threshold=0.3, max_threshold=0.99, max_iterations=20) -> None:
    video_dir, video_name = os.path.split(video_path)
    video_base_name = os.path.splitext(video_name)[0]
    target_path = os.path.join(video_dir, video_base_name)
    
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    left, right = min_threshold, max_threshold
    optimal_threshold = (left + right) / 2

    for _ in range(max_iterations):
        threshold = (left + right) / 2
        frames = handle_video_frames(video_path)
        clusters = frames_cluster(frames, threshold)
        
        for file in os.listdir(target_path):
            os.remove(os.path.join(target_path, file))
        store_keyframe(video_path, target_path, clusters)

        num_images = len(os.listdir(target_path))

        if min_keyframes <= num_images <= max_keyframes:
            optimal_threshold = threshold
            break  
        elif num_images < min_keyframes:
            left = threshold 
        else:
            right = threshold

    return target_path


def handle_video_frames(video_path: str) -> List[Frame]:
    """
    处理视频获取所有帧的HSV直方图
    :param video_path: 视频路径
    :return: 帧对象数组
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)  
    height, width = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH) 


    no = 1
    frames = list()

    nex, frame = cap.read()
    while nex:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # BGR -> HSV 转换颜色空间
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 5, 5], [0, 256, 0, 256, 0, 256])
        flatten_hists = hist.flatten()
        flatten_hists /= height * width
        frames.append(Frame(no, flatten_hists))

        no += 1
        nex, frame = cap.read()

    cap.release()
    return frames

def frames_cluster(frames: List[Frame], threshold: float) -> List[FrameCluster]:
    ret_clusters = [FrameCluster([frames[0]], frames[0])]  
    for frame in frames[1:]:
        max_ratio, clu_idx = 0, -1
        for i, clu in enumerate(ret_clusters):
            sim_ratio = similarity(frame.hist, clu.center.hist)
            if sim_ratio > max_ratio:
                max_ratio, clu_idx = sim_ratio, i

        if max_ratio < threshold:
            ret_clusters.append(FrameCluster([frame], frame))
        else:
            ret_clusters[clu_idx].cluster.append(frame)
            ret_clusters[clu_idx].re_center()

    return ret_clusters


def store_keyframe(video_path: str, target_path: str, frame_clusters: List[FrameCluster]) -> None:
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  

    keyframe_nos = set([cluster.keyframe_no() for cluster in frame_clusters]) 

    no, saved_count = 1, 1 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if no in keyframe_nos:
            cv2.imwrite(f'{target_path}/{saved_count}.jpg', frame)  
            saved_count += 1
        no += 1

    cap.release()







from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter

def katna_keyframes_extraction(video_file_path, no_of_frames_to_returned):
    vd = Video()
    
    video_dir, video_name = os.path.split(video_file_path)
    video_base_name = os.path.splitext(video_name)[0]
    target_path = os.path.join(video_dir, video_base_name)




    if os.path.exists(target_path) and len([f for f in os.listdir(target_path) if f.endswith('.jpeg')]) >= no_of_frames_to_returned:
        logging.info(f"Keyframes already extracted and present in {target_path}")
        return target_path
    
    disk_writer = KeyFrameDiskWriter(location=target_path)

 

    logging.info(f"Input video file path = {video_file_path}")
 
    vd.extract_video_keyframes(
        no_of_frames=no_of_frames_to_returned, file_path=video_file_path,
        writer=disk_writer
    )
    logging.info(f"video {video_base_name}：Keyframes extracted successfully")

    reorder_and_rename_images(target_path)
    
    return target_path

def reorder_and_rename_images(directory_path):
    images = sorted(glob.glob(os.path.join(directory_path, "*.jpeg")), key=os.path.getmtime)
    
    for i, image_path in enumerate(images, start=1):
        new_name = os.path.join(directory_path, f"{i}.jpeg")
        os.rename(image_path, new_name)
    

    logging.info("Images have been renamed successfully.")
























def gpt_summary(video_llava_answer, key_frame_highlights):

    api_key = MODEL_CONFIG["llm"]["model_key"]
    headers = {
        "Authorization": 'Bearer ' + api_key,
    }

    params = {
    "messages": [
        {
            "role": "system",
            "content": """
            Your task is to generate a coherent, logically structured, and accurate video description. This description must:
            1. Strictly adhere to the provided information, with absolutely no speculation or additions.
            2. Integrate the overall video content analysis with detailed information from 7 key frames.
            3. Maintain the highest level of accuracy as the paramount principle.
            4. Create a fluid, logically clear narrative that encompasses all critical details.
            5. Range between 100-500 words, ensuring comprehensiveness while avoiding redundancy.
            """
        },
        {
            "role": "user",
            "content": f"""
            Based on the following information, craft a cohesive and accurate video description:

            Overall video content: {video_llava_answer}

            Highlights from 7 key frames: {key_frame_highlights}

            Your description must:
            1. Synthesize the overall content and information from 7 key frames into a single, cohesive narrative.
            2. Adhere strictly to facts, with absolutely no speculation.
            3. Organize content chronologically or logically, ensuring narrative continuity and fluency.
            4. Include all significant actions, scenes, and visual element details.
            5. Maintain an objective and accurate tone throughout.
            6. Ensure each detail is directly supported by the provided information.
            7. Create a description comprehensible to someone who has not viewed the video.

            Final output: A logically clear, cohesive, and accurate video description encompassing the entire video content.

            Remember: Accuracy is the highest priority, followed by comprehensiveness and coherence.
            """
        }
    ],
    "model": "gpt-4o-mini"
}

    response = requests.post(
        "https://aigptx.top/v1/chat/completions",
        headers=headers,
        json=params,
        stream=False
    )
    res = response.json()

    video_summary_answer = res['choices'][0]['message']['content']


    return video_summary_answer



















def pipe_prompt_2_only_accuracy(video_file_path,image_folder_path):

    video_prompt = "Describe the key events in the video chronologically, including time, location, and participants. Focus on observable actions and processes, avoiding speculation. Provide a brief and accurate summary of the video content."

    image_prompt = "Accurately describe this image, including visible subjects, their actions, and the main scene or background information. Extract and describe any visible text. Only describe what can be directly observed in the image. Do not speculate on uncertain details, only describe the most certain elements."

    temperature = 0.2
    all_answers = {
    "Video": {}, 
    "Image": {},     
    "LLM":{}    
    }

    video_summary_answer = process_video(video_file_path, video_prompt)
    all_answers["Video"]["question"] = video_prompt
    all_answers["Video"]["answer"] = video_summary_answer

    image_summary_answer = {}
    
    all_answers["Image"]["question"] = image_prompt

    image_files = [f for f in os.listdir(image_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    sorted_image_files = sorted(image_files)

    for image_path in sorted_image_files:
        image_name = os.path.join(image_folder_path, image_path)
        answer = process_image(image_name, image_prompt)
        all_answers["Image"][image_path] = {"answer": answer}
    

    key_frame_summary = "; ".join([
        f"{image_file}: {details['answer']}"
        for image_file, details in all_answers["Image"].items()
        if 'answer' in details and details['answer']
    ])

    gpt_summary_answer = gpt_summary(video_summary_answer, key_frame_summary)
    all_answers["LLM"]["answer"] = gpt_summary_answer

    return all_answers











def process_folder_videos():
    root_dir = DATASET_CONFIG["root_dir"]
    test_annotation = os.path.join(root_dir, DATASET_CONFIG["annotation"]["test"])
    test_data_dir = os.path.join(root_dir, DATASET_CONFIG["data_dir"]["test"])
    test_video_dir = os.path.join(root_dir, DATASET_CONFIG["video_dir"]["test"])

    test_output_dir = os.path.join(root_dir, DATASET_CONFIG["output_dir"]["test"])
    
    test_VD_result_dir = test_output_dir
    if not os.path.exists(test_VD_result_dir):
        os.makedirs(test_VD_result_dir)

    with open(test_annotation, 'r') as f:
        video_ids = [line.strip() for line in f if line.strip()]

    for i, video_id in enumerate(video_ids):
        try:
            logging.info(f"Processing video {i+1}/{len(video_ids)}: {video_id}")
            start_time = time.time()

            video_file = None
            for ext in ['.mp4', '.mkv']:
                potential_file = os.path.join(test_video_dir, f"{video_id}{ext}")
                if os.path.exists(potential_file):
                    video_file = potential_file
                    break
            
            if not video_file:
                logging.warning(f"No video file found for ID: {video_id}")
                continue

            data_folder = os.path.join(test_video_dir, str(video_id))

            if not os.path.exists(data_folder):
                os.makedirs(data_folder)
            original_output_path = os.path.join(data_folder, f"{video_id}.json")

            logging.info(f"Extracting keyframes for video: {video_id}")
            try:
                no_of_frames_to_returned = VIDEO_DESCRIPTOR_CONFIG.get("keyframes_per_video")
                keyframes_folder = katna_keyframes_extraction(video_file, no_of_frames_to_returned)
                logging.info(f"Keyframes extracted successfully for video: {video_id}")
            except Exception as e:
                logging.error(f"Failed to extract keyframes for video {video_id}: {str(e)}")
                continue

            descriptor_result = pipe_prompt_2_only_accuracy(video_file, keyframes_folder)
            
            with open(original_output_path, 'w', encoding='utf-8-sig') as f:
                json.dump({"Zero-Shot Detailed Inquiry Prompt": descriptor_result}, f, indent=4)

            original_json_path = os.path.join(test_data_dir, f"{video_id}.json")
            try:
                with open(original_json_path, 'r', encoding='utf-8-sig') as f:
                    original_data = json.load(f)
                
                video_info = original_data.get("video_information", {})
                
                video_info["Video_descriptor"] = descriptor_result["LLM"]["answer"]
                
                final_result = {
                    "claim": original_data.get("claim", ""),
                    "Video_information": video_info
                }

                result_json_path = os.path.join(test_VD_result_dir, f"{video_id}.json")
                with open(result_json_path, 'w', encoding='utf-8') as f:
                    json.dump(final_result, f, indent=4, ensure_ascii=False)

                end_time = time.time()
                logging.info(f"Time taken for video {video_id}: {(end_time - start_time)/60:.2f} minutes")
                logging.info(f"----------------------------- Analysis for {video_id} completed successfully -----------------------------")

            except Exception as e:
                logging.error(f"Error processing JSON for video {video_id}: {str(e)}")
                continue

        except Exception as e:
            logging.error(f"Error processing video {video_id}: {str(e)}")
            continue








class BeijingFormatter(logging.Formatter):
    def converter(self, timestamp):
        dt = datetime.fromtimestamp(timestamp, tz=pytz.UTC)
        return dt.astimezone(pytz.timezone('Asia/Shanghai'))

    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        return dt.strftime('%Y-%m-%d %H:%M:%S')

    def format(self, record):
        record.asctime = self.formatTime(record)
        return super().format(record)

logger = logging.getLogger()
logger.setLevel(logging.INFO)




def process_folder_videos_with_logging():
    """
    Wrapper function for process_folder_videos that sets up logging
    and generates a separate log file for this function's operations.
    """
    root_dir = DATASET_CONFIG["root_dir"]
    test_video_dir = os.path.join(root_dir, DATASET_CONFIG["video_dir"]["test"])
    
    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f'process_folder_videos_{log_timestamp}.log'
    log_file_path = os.path.join(test_video_dir, log_file_name)
    
    os.makedirs(test_video_dir, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file_path)
    formatter = BeijingFormatter('%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    try:
        logger.info(f"Starting process_folder_videos at {log_timestamp}")
        logger.info(f"Log file: {log_file_path}")
        
        process_folder_videos()
        
        logger.info("process_folder_videos completed successfully")
    
    except Exception as e:
        logger.error(f"Error in process_folder_videos: {e}")
        logger.error(traceback.format_exc())
        raise
    
    finally:
        logger.removeHandler(file_handler)
        
        file_handler.close()

