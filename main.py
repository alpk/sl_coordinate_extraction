import logging
import gc
import logging
import math
import multiprocessing
import os
import pickle
import time
import argparse
from glob import glob

import cv2
import mediapipe as mp
import numpy as np
from joblib import Parallel, delayed
from natsort import natsorted
from tqdm.auto import tqdm

logging.basicConfig(level=logging.DEBUG)

mp_holistic = mp.solutions.holistic

parser = argparse.ArgumentParser(description='Process a dataset to extract coordinates.')
parser.add_argument('-dataset_name',  type=str, default='CSL')
parser.add_argument('-base_path',  type=str, default="/media/warp/Databases/sign_language_recognition/raw_datasets/isolated/CSLR/frames/")
parser.add_argument('-save_path',  type=str, default="/media/warp/Databases/sign_language_recognition/raw_datasets/isolated/CSLR/skeleton_mediapipe/")
parser.add_argument('-use_videos',  type=bool, default=False)
parser.add_argument('-color_depth_same_folder',  type=bool, default=False)
parser.add_argument('-folder_order',  type=str, default='class_sign')
parser.add_argument('-N_FACE_LANDMARKS',  type=int, default=468)
parser.add_argument('-N_BODY_LANDMARKS',  type=int, default=33)
parser.add_argument('-N_HAND_LANDMARKS',  type=int, default=21)
parser.add_argument('-number_of_cores',  type=int, default=multiprocessing.cpu_count())


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


def process_body_landmarks(component, n_points):
    kps = np.zeros((n_points, 3))
    conf = np.zeros(n_points)
    if component is not None:
        landmarks = component.landmark
        kps = np.array([[p.x, p.y, p.z] for p in landmarks])
        conf = np.array([p.visibility for p in landmarks])
    return kps, conf


def process_other_landmarks(component, n_points):
    kps = np.zeros((n_points, 3))
    conf = np.zeros(n_points)
    if component is not None:
        landmarks = component.landmark
        kps = np.array([[p.x, p.y, p.z] for p in landmarks])
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
    keypoints = []
    confs = []

    for f in range(frames.shape[0]):
        frame = frames[f, :, :, :]
        results = holistic.process(frame)
        body_data, body_conf = process_body_landmarks(
            results.pose_landmarks, args.N_BODY_LANDMARKS
        )
        """
        face_data, face_conf = process_other_landmarks(
            results.face_landmarks, N_FACE_LANDMARKS
        )
        """
        lh_data, lh_conf = process_other_landmarks(
            results.left_hand_landmarks, args.N_HAND_LANDMARKS
        )
        rh_data, rh_conf = process_other_landmarks(
            results.right_hand_landmarks, args.N_HAND_LANDMARKS
        )
        """
        plt.imshow(frame)
        plt.plot(body_data[:,0]*frame.shape[1],body_data[:,1]*frame.shape[0],'r.')
        plt.plot(rh_data[:, 0] * frame.shape[1], rh_data[:, 1] * frame.shape[0], 'b.')
        plt.plot(lh_data[:, 0] * frame.shape[1], lh_data[:, 1] * frame.shape[0], 'c.')
        plt.show()
        """
        data = np.concatenate([body_data, lh_data, rh_data])
        conf = np.concatenate([body_conf, lh_conf, rh_conf])

        keypoints.append(data)
        confs.append(conf)

    # TODO: Reuse the same object when this issue is fixed: https://github.com/google/mediapipe/issues/2152
    holistic.close()
    del holistic
    gc.collect()

    keypoints = np.stack(keypoints)
    confs = np.stack(confs)
    return keypoints, confs


def gen_keypoints_for_frames(frames, save_path,args):
    pose_kps, pose_confs = get_holistic_keypoints(frames,args)
    body_kps = np.concatenate([pose_kps[:, :33, :], pose_kps[:, 501:, :]], axis=1)

    confs = np.concatenate([pose_confs[:, :33], pose_confs[:, 501:]], axis=1)

    d = {"keypoints": body_kps, "confidences": confs}

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



    # shutil.rmtree(SAVE_DIR,ignore_errors=True)
    os.makedirs(args.save_path, exist_ok=True)

    file_paths = []
    save_paths = []
    if args.folder_order in 'class_sign':
        for cls in sorted(os.listdir(args.base_path)):
            os.makedirs(os.path.join(args.save_path, cls), exist_ok=True)
            for file in sorted(os.listdir(args.base_path + cls)):
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
            for path, save_path in tqdm(zip(file_paths, save_paths))
        )
    else:
        Parallel(n_jobs=args.number_of_cores, backend="loky")(
            delayed(gen_keypoints_for_folder)(path, save_path, ["*.jpg","*.png"], args)
            for path, save_path in tqdm(zip(file_paths, save_paths))
        )
