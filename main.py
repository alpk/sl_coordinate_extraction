import gc
import logging
import math
import multiprocessing
import os
import pickle
import time
import argparse
import shutil
import sys
import random
from glob import glob

import cv2
import mediapipe as mp
import numpy as np
from joblib import Parallel, delayed
from natsort import natsorted
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count

# Initial script forked from https://github.com/AI4Bharat/OpenHands/blob/main/scripts/mediapipe_extract.py

logging.basicConfig(level=logging.DEBUG)

mp_holistic = mp.solutions.holistic

parser = argparse.ArgumentParser(description='Process a dataset to extract coordinates.')
parser.add_argument('--dataset_name',  type=str, default='CSL')
parser.add_argument('--base_path',  type=str, default="/media/hdd1/data/bsign22k/frames/")
parser.add_argument('--save_path',  type=str, default="/media/ssd1/data/bsign22k/skeleton-mediapipe/")
parser.add_argument('--use_videos',  type=bool, default=False)
parser.add_argument('--color_depth_same_folder',  type=bool, default=False)
parser.add_argument('--folder_order',  type=str, default='class_sign')
parser.add_argument('--get_face_landmarks',  type=bool, default=False)
parser.add_argument('--get_pose_landmarks',  type=bool, default=True)
parser.add_argument('--get_hand_landmarks',  type=bool, default=True)
parser.add_argument('--get_3Dpose_landmarks',  type=bool, default=True)
parser.add_argument('--number_of_cores',  type=int, default=multiprocessing.cpu_count()//2)
parser.add_argument('--clear_dir',  type=bool, default=False)
parser.add_argument('--randomize_order',  type=bool, default=True)


mediapipe_body_names = []
mediapipe_hand_names = []


class Counter(object):
    # https://stackoverflow.com/a/47562583/
    def __init__(self, initval=0):
        self.val = multiprocessing.RawValue("i", initval)
        self.lock = multiprocessing.Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    @property
    def value(self):
        return self.val.value


def process_landmarks(results, mp_holistic, args):
    T = len(results)
    keypoints = {'pose':{},'face':{}, 'hand_left':{}, 'hand_right':{}, 'pose_world':{}}
    confidence = {'pose':{}, 'face':{}, 'hand_left':{}, 'hand_right':{}, 'pose_world':{}}
    if T != 0:
        if args.get_pose_landmarks:
            for ln in mp_holistic.PoseLandmark:
                keypoints['pose'][ln] =  np.zeros((T, 3),float)
                confidence['pose'][ln] = np.zeros((T, 1),float)
                for t in range(T):
                    if results[t].pose_landmarks is not None:
                        keypoints['pose'][ln][t,:] = np.array([results[t].pose_landmarks.landmark[ln].x,
                                                               results[t].pose_landmarks.landmark[ln].y,
                                                               results[t].pose_landmarks.landmark[ln].z])
                        confidence['pose'][ln][t] = results[t].pose_landmarks.landmark[ln].visibility

        if args.get_face_landmarks:
            face_landmarks = ['face_landmark_'+ format(x,'03') for x in range(468)]
            for ln in range(468):
                keypoints['face'][face_landmarks[ln]] = np.zeros((T, 3), float)
                confidence['face'][face_landmarks[ln]] = np.zeros((T, 1), float)
                for t in range(T):
                    if results[t].face_landmarks is not None:
                        keypoints['face'][face_landmarks[ln]][t, :] = np.array([results[t].face_landmarks.landmark[ln].x,
                                                                results[t].face_landmarks.landmark[ln].y,
                                                                results[t].face_landmarks.landmark[ln].z])
                        confidence['face'][face_landmarks[ln]][t] = 1

        if args.get_hand_landmarks:
            for ln in mp_holistic.HandLandmark:
                keypoints['hand_left'][ln] = np.zeros((T, 3), float)
                confidence['hand_left'][ln] = np.zeros((T, 1), float)
                keypoints['hand_right'][ln] = np.zeros((T, 3), float)
                confidence['hand_right'][ln] = np.zeros((T, 1), float)
                for t in range(T):
                    if results[t].left_hand_landmarks is not None:
                        keypoints['hand_left'][ln][t, :] = np.array([results[t].left_hand_landmarks.landmark[ln].x,
                                                                results[t].left_hand_landmarks.landmark[ln].y,
                                                                results[t].left_hand_landmarks.landmark[ln].z])
                        confidence['hand_left'][ln][t] = 1

                    if results[t].right_hand_landmarks is not None:
                        keypoints['hand_right'][ln][t, :] = np.array([results[t].right_hand_landmarks.landmark[ln].x,
                                                                results[t].right_hand_landmarks.landmark[ln].y,
                                                                results[t].right_hand_landmarks.landmark[ln].z])
                        confidence['hand_right'][ln][t] = 1
        if args.get_3Dpose_landmarks:
            for ln in mp_holistic.PoseLandmark:
                keypoints['pose_world'][ln] = np.zeros((T, 3), float)
                confidence['pose_world'][ln] = np.zeros((T, 1), float)
                for t in range(T):
                    if results[t].pose_world_landmarks is not None:
                        keypoints['pose_world'][ln][t, :] = np.array([results[t].pose_world_landmarks.landmark[ln].x,
                                                                results[t].pose_world_landmarks.landmark[ln].y,
                                                                results[t].pose_world_landmarks.landmark[ln].z])
                        confidence['pose_world'][ln][t] = results[t].pose_world_landmarks.landmark[ln].visibility
    return keypoints, confidence



openpose_keys = ['nose', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist',
                 'left_shoulder', 'left_elbow', 'left_wrist', 'middle_hip',
                 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee',
                 'left_ankle', 'right_eye', 'left_eye', 'right_ear', 'left_ear',
                 'left_big_toe', 'left_small_toe', 'left_heel', 'right_big_toe',
                 'right_small_toe', 'right_heel']
body_keys = ['lunate_bone', 'thumb_1', 'thumb_2', 'thumb_3', 'thumb_4',
             'index_finger_5', 'index_finger_6', 'index_finger_7', 'index_finger_8',
             'middle_finger_9', 'middle_finger_10', 'middle_finger_11',
             'middle_finger_12', 'ring_finger_13', 'ring_finger_14', 'ring_finger_15',
             'ring_finger_16', 'little_finger_17', 'little_finger_18', 'little_finger_19',
             'little_finger_20']


def get_holistic_keypoints(frames, args):
    """
    For videos, it's optimal to create with `static_image_mode=False` for each video.
    https://google.github.io/mediapipe/solutions/holistic.html#static_image_mode
    """

    holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=2)

    results = []
    for f in range(frames.shape[0]):
        frame = frames[f, :, :, :]
        results.append(holistic.process(frame))
    holistic.close()
    del holistic
    gc.collect()
    keypoints, confidence = process_landmarks(results, mp_holistic, args)

    return keypoints, confidence



def gen_keypoints_for_frames(frames, save_path, args):
    pose_kps, pose_confs = get_holistic_keypoints(frames, args )

    d = {"keypoints": pose_kps, "confidences": pose_confs}

    with open(save_path + ".pickle", "wb") as f:
        pickle.dump(d, f, protocol=4)


def load_frames_from_video(video_path):
    frames = []
    vidcap = cv2.VideoCapture(video_path)
    while vidcap.isOpened():
        success, img = vidcap.read()
        if not success:
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (640, 480))
        frames.append(img)
    if np.sum(frames[0]) == 0:
        del frames[0]
    vidcap.release()
    #print(video_path, ': ', len(frames),flush=True)
    # cv2.destroyAllWindows()
    return np.asarray(frames)


def load_frames_from_folder(frames_folder, patterns=["*.jpg"]):
    images = []
    for pattern in patterns:
        images.extend(glob(f"{frames_folder}/{pattern}"))
    images = natsorted(list(set(images)))  # remove dupes
    logging.info(frames_folder + ': ' + format(len(images)))
    if not images:
        exit(f"ERROR: No frames in folder: {frames_folder}")

    frames = []
    for img_path in images:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)

    return np.asarray(frames)


def gen_keypoints_for_video(video_path, save_path, args):
    if not os.path.isfile(video_path):
        print("SKIPPING MISSING FILE:", video_path)
        return
    frames = load_frames_from_video(video_path)
    gen_keypoints_for_frames(frames, save_path, args)


def gen_keypoints_for_folder(folder, save_path, file_patterns, args):
    frames = load_frames_from_folder(folder, file_patterns)
    gen_keypoints_for_frames(frames, save_path, args)


def generate_pose(dataset, save_folder, worker_index, num_workers, counter):
    num_splits = math.ceil(len(dataset) / num_workers)
    end_index = min((worker_index + 1) * num_splits, len(dataset))
    for index in range(worker_index * num_splits, end_index):
        imgs, label, video_id = dataset.read_data(index)
        save_path = os.path.join(save_folder, video_id)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        gen_keypoints_for_frames(imgs, save_path)
        counter.increment()


def dump_pose_for_dataset(
        dataset, save_folder, num_workers=multiprocessing.cpu_count()
):
    os.makedirs(save_folder, exist_ok=True)
    processes = []
    counter = Counter()
    for i in tqdm(range(num_workers), desc="Creating sub-processes..."):
        p = multiprocessing.Process(
            target=generate_pose, args=(dataset, save_folder, i, num_workers, counter)
        )
        p.start()
        processes.append(p)

    total_samples = len(dataset)
    with tqdm(total=total_samples) as pbar:
        while counter.value < total_samples:
            pbar.update(counter.value - pbar.n)
            time.sleep(2)

    for i in range(num_workers):
        processes[i].join()
    print(f"Pose data successfully saved to: {save_folder}")





if __name__ == "__main__":
    args = parser.parse_args()
    logging.info('n_cores is: ' + format(args.number_of_cores))


    if args.clear_dir:
        shutil.rmtree(args.save_path,ignore_errors=True)
    os.makedirs(args.save_path, exist_ok=True)

    file_paths = []
    save_paths = []
    if args.folder_order in 'class_sign':

        rand_list = sorted(os.listdir(args.base_path))
        if args.randomize_order:
            random.shuffle(rand_list)

        for cls in rand_list:
            os.makedirs(os.path.join(args.save_path, cls), exist_ok=True)

            file_list = sorted(os.listdir(args.base_path + cls))
            if args.randomize_order:
                random.shuffle(file_list)
            for file in file_list:
                # if "color" in file:
                if not os.path.isfile(
                        os.path.join(args.save_path, cls, file.replace(".avi", "").replace("_color", "")) + '.pkl'):
                    file_paths.append(os.path.join(args.base_path, cls, file))
                    save_paths.append(
                        os.path.join(args.save_path, cls, file.replace(".avi", "").replace("_color", "")))
    else:
        logging.exception('Unsupported dataset folder order')

    if args.use_videos:
        Parallel(n_jobs=args.number_of_cores)(
            delayed(gen_keypoints_for_video)(path, save_path, args)
            for path, save_path in tqdm(zip(file_paths, save_paths))
        )
    else:
        if args.number_of_cores > 1:
            Parallel(n_jobs=args.number_of_cores)(
                delayed(gen_keypoints_for_folder)(path, save_path, ["*.jpg", "*.png"], args)
                for path, save_path in tqdm(zip(file_paths, save_paths)))
        else:
            for path, save_path in tqdm(zip(file_paths, save_paths)):
                gen_keypoints_for_folder(path, save_path, ["*.jpg","*.png"], args)

