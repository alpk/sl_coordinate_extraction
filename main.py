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
parser.add_argument('--base_path',  type=str, default="/media/warp/Databases/sign_language_recognition/raw_datasets/isolated/CSLR/frames/")
parser.add_argument('--save_path',  type=str, default="/media/warp/Databases/sign_language_recognition/raw_datasets/isolated/CSLR/skeleton_mediapipe/")
parser.add_argument('--use_videos',  type=bool, default=False)
parser.add_argument('--color_depth_same_folder',  type=bool, default=False)
parser.add_argument('--folder_order',  type=str, default='class_sign')
parser.add_argument('--N_FACE_LANDMARKS',  type=int, default=468)
parser.add_argument('--N_BODY_LANDMARKS',  type=int, default=33)
parser.add_argument('--N_HAND_LANDMARKS',  type=int, default=21)
parser.add_argument('--number_of_cores',  type=int, default=1)#multiprocessing.cpu_count())
parser.add_argument('--clear_dir',  type=bool, default=False)


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


def process_body_landmarks(component, n_points, landmark_name=None):
    kps = np.zeros((n_points, 3))
    conf = np.zeros(n_points)
    if component is not None:
        if landmark_name is None:
            landmarks = component.landmark
            kps = np.array([[p.x, p.y, p.z] for p in landmarks])
            conf = np.array([p.visibility for p in landmarks])
        else:
            landmarks = component.landmark
            kps = []
            conf = []
            for ln in landmark_name:
                p = landmarks[ln]
                kps.append(np.array([p.x, p.y, p.z]))
                conf.append(np.array(p.visibility))
            kps = np.array(kps)
            conf = np.array(conf)
    return kps, conf


def process_other_landmarks(component, n_points, landmark_name=None):
    kps = np.zeros((n_points, 3))
    conf = np.zeros(n_points)
    if component is not None:
        if landmark_name is None:
            landmarks = component.landmark
            kps = np.array([[p.x, p.y, p.z] for p in landmarks])
            conf = np.ones(n_points)
        else:
            landmarks = component.landmark
            kps = []
            for ln in landmark_name:
                p = landmarks[ln]
                kps.append(np.array([p.x, p.y, p.z]))
            kps = np.array(kps)
            conf = np.ones(n_points)


    return kps, conf


def get_holistic_keypoints(
        frames,args
):
    """
    For videos, it's optimal to create with `static_image_mode=False` for each video.
    https://google.github.io/mediapipe/solutions/holistic.html#static_image_mode
    """
    holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=2)
    pose_landmarks = [x for x in mp_holistic.PoseLandmark]
    hand_landmarks = [x for x in mp_holistic.HandLandmark]

    keypoints = []
    confs = []
    joint_names = []

    for f in range(frames.shape[0]):
        frame = frames[f, :, :, :]
        results = holistic.process(frame)

        names = [str(x) for x in pose_landmarks] + ['face_' + format(i,'03') for i in range(468)] + ['Left_' + str(x) for x in hand_landmarks]\
                + ['Right_' + str(x) for x in hand_landmarks]  + ['World_' + str(x) for x in pose_landmarks]

        body_data, body_conf = process_body_landmarks(
            results.pose_landmarks, args.N_BODY_LANDMARKS, pose_landmarks
        )
        face_data, face_conf = process_other_landmarks(
            results.face_landmarks, args.N_FACE_LANDMARKS, None
        )

        lh_data, lh_conf = process_other_landmarks(
            results.left_hand_landmarks, args.N_HAND_LANDMARKS, hand_landmarks
        )
        rh_data, rh_conf = process_other_landmarks(
            results.right_hand_landmarks, args.N_HAND_LANDMARKS, hand_landmarks
        )
        pw_data, pw_conf = process_body_landmarks(
            results.pose_world_landmarks, args.N_BODY_LANDMARKS, pose_landmarks
        )
        """
        plt.imshow(frame)
        plt.plot(body_data[:,0]*frame.shape[1],body_data[:,1]*frame.shape[0],'r.')
        plt.plot(rh_data[:, 0] * frame.shape[1], rh_data[:, 1] * frame.shape[0], 'b.')
        plt.plot(lh_data[:, 0] * frame.shape[1], lh_data[:, 1] * frame.shape[0], 'c.')
        plt.show()
        """
        data = np.concatenate([body_data, lh_data, rh_data,face_data,pw_data])
        conf = np.concatenate([body_conf, lh_conf, rh_conf,face_conf,pw_conf])


        keypoints.append(data)
        confs.append(conf)
        joint_names.append(names)

    # TODO: Reuse the same object when this issue is fixed: https://github.com/google/mediapipe/issues/2152
    holistic.close()
    del holistic
    gc.collect()

    keypoints = np.stack(keypoints)
    confs = np.stack(confs)
    joint_names = np.stack(joint_names)
    return keypoints, confs, joint_names


def gen_keypoints_for_frames(frames, save_path,args):
    pose_kps, pose_confs, joint_names = get_holistic_keypoints(frames,args)
    body_kps = np.concatenate([pose_kps[:, :33, :], pose_kps[:, 501:, :]], axis=1)

    confs = np.concatenate([pose_confs[:, :33], pose_confs[:, 501:]], axis=1)

    d = {"keypoints": body_kps, "confidences": confs, "joint_names":joint_names}

    with open(save_path + ".pkl", "wb") as f:
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
    # cv2.destroyAllWindows()
    return np.asarray(frames)


def load_frames_from_folder(frames_folder, patterns=["*.jpg"]):
    images = []
    for pattern in patterns:
        images.extend(glob(f"{frames_folder}/{pattern}"))
    images = natsorted(list(set(images)))  # remove dupes
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

        rand_list = os.listdir(args.base_path)
        random.shuffle(rand_list)

        for cls in rand_list:
            os.makedirs(os.path.join(args.save_path, cls), exist_ok=True)

            file_list = os.listdir(args.base_path + cls)
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
        Parallel(n_jobs=args.number_of_cores, backend="loky")(
            delayed(gen_keypoints_for_video)(path, save_path, args)
            for path, save_path in tqdm(zip(file_paths, save_paths), file=sys.stdout)
        )
    else:
        if args.number_of_cores > 1:
            Parallel(n_jobs=args.number_of_cores, backend="loky")(
                delayed(gen_keypoints_for_folder)(path, save_path, ["*.jpg", "*.png"], args)
                for path, save_path in tqdm(zip(file_paths, save_paths), file=sys.stdout))
        else:
            for path, save_path in tqdm(zip(file_paths, save_paths), file=sys.stdout):
                gen_keypoints_for_folder(path, save_path, ["*.jpg","*.png"], args)

