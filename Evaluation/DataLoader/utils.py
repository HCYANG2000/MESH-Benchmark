import os
import json
import pandas as pd
from PIL import Image
import numpy as np
from decord import VideoReader, cpu

def load_video(video_path, args, frames):
    fixed_length = args['fixed_length'] if 'fixed_length' in args else None
    return_type = args['return_type'] if 'return_type' in args else 'tensor'
    if return_type == 'tensor':
        vr = VideoReader(video_path + '.mp4', ctx=cpu(0))
        total_frame_num = len(vr)
        fps = round(vr.get_avg_fps())
        if fixed_length:
            if frames is None:
                raise ValueError('Frames not provided')
            # If the fixed_length is specified, then cut the video according to the fixed_length
            cut_length = {'short': 24, 'medium': 48, 'long': 96, 'superlong': 192}
            cut_length = cut_length[fixed_length] * 2
            # Get the frame to cut
            # FIXME: The fixed length can only apply to one specific construction of video. It is not general
            frame = (int(frames[0]) - 1) * 2
            # Get the start and end frame to cut
            start_frame = frame - cut_length//2
            end_frame = frame + cut_length//2
            if start_frame < 0:
                start_frame = 0
                end_frame = cut_length
            elif end_frame > len(vr):
                start_frame = len(vr) - cut_length
                end_frame = len(vr)
            # Get the frames and convert it to numpy
            cut_frames = vr.get_batch([i for i in range(start_frame, end_frame, 6)])
            return cut_frames
        else:
            frame_idx = [i for i in range(0, len(vr), fps)]
            # sample_fps = args.for_get_frames_num if total_frame_num > args.for_get_frames_num else total_frame_num
            if len(frame_idx) > args['for_get_frames_num'] or args['force_sample']:
                sample_fps = args['for_get_frames_num']
                uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
                frame_idx = uniform_sampled_frames.tolist()
            spare_frames = vr.get_batch(frame_idx)
            return spare_frames.asnumpy()
    elif return_type == 'path':
        return video_path + '.mp4'
    else:
        raise ValueError('Return type not supported')

# Function to load image data
def load_image(image_path):
    return np.array(Image.open(image_path).convert("RGB"))

# Function to load the sampled frames from video
def load_frames(video_path, args, frames):
    fixed_length = args['fixed_length'] if 'fixed_length' in args else None
    return_type = args['return_type'] if 'return_type' in args else 'tensor'
    # Read all the files in the directory
    files = os.listdir(video_path)
    # Sort the files according to the name, which is the frame number
    files.sort(key=lambda x: int(x.split(".")[0]))
    # For person ablation study only
    if fixed_length:
        if frames is None:
            raise ValueError('Frames not provided')
        if fixed_length == 'repeat':
            if type(frames[0]) is list:
                # Repeat the frame in the middle
                frame = (frames[0][0] + frames[0][1])//2
            else:
                # Repeat the first frame
                frame = int(frames[0]) - 1
            files_paths = [os.path.join(video_path, files[i]) for i in [frame] * args['for_get_frames_num']]
            spare_frames = np.array([load_image(file) for file in files_paths])
        elif fixed_length == 'flex':
            if type(frames[0]) is list:
                # cut the frames according to the list provided
                start_frame = frames[0][0]
                end_frame = frames[0][1]
            elif len(frames) == 2:
                print("WARNING: Incompatible Format. First two elements of the list will be taken")
                start_frame = frames[0]
                end_frame = frames[1]
            else:
                raise ValueError("ERROR: Incompatible Format.")
            files_paths = [os.path.join(video_path, files[i]) for i in range(start_frame, end_frame, args['sample_rate'])]
            spare_frames = np.array([load_image(file) for file in files_paths])
        else:
            # assign the length
            len_dict = {'image': 1, 'ssshort': 6, 'sshort': 12, 'short': 24, 'short_ext':32, 'medium': 48, 'medium_ext': 72, 'long': 96, 'superlong': 192}
            cut_length = len_dict[fixed_length]
            if type(frames[0]) is list:
                # Read the frames in the middle
                frame = (frames[0][0] + frames[0][1])//2
                # There is one video in the dataset that has only 21 frames
                if len(files) < cut_length:
                    cut_length = len(files)
                # The cut length should at least cover the frames provided (Not for ablation study)
                if frames[0][1] - frames[0][0] > cut_length and cut_length > 12:
                    cut_length = frames[0][1] - frames[0][0]
            else:
                # Read the frames
                frame = int(frames[0]) - 1
                # There is one video in the dataset that has only 21 frames
                if len(files) < cut_length:
                    cut_length = len(files)
            # cut the cut_length frames around the frame
            start_frame = frame - cut_length//2
            end_frame = frame + cut_length//2
            # Prevent the start_frame and end_frame from going out of the range
            if start_frame < 0:
                start_frame = 0
                end_frame = cut_length
            elif end_frame > len(files):
                start_frame = len(files) - cut_length
                end_frame = len(files)
            # If the start_frame is equal to the end_frame, then return the frame
            if start_frame == end_frame and cut_length == 1:
                end_frame += 1
            files_paths = [os.path.join(video_path, files[i]) for i in range(start_frame, end_frame, 3)] if type(frames[0]) is not list else [os.path.join(video_path, files[i]) for i in range(start_frame, end_frame, args['sample_rate'])]
            spare_frames = np.array([load_image(file) for file in files_paths])
    else:
        # Convert the frames to int
        frames = [int(frame)-1 for frame in frames] if frames is not None and type(frames[0]) is not list and int(frames[0]) != 0 else None
        # If the number of frames is larger than the number of frames to sample, sample the frames
        if len(files) > args['for_get_frames_num'] or args['force_sample']:
            # Sample the frames
            sample_fps = args['for_get_frames_num']
            # Get the indices of the frames to sample
            frame_idx = [int(i) for i in np.linspace(0, len(files) - 1, sample_fps)]
            # Check if any frame_idx is in the frames
            if frames == None or any([frame in frames for frame in frame_idx]):
                # If any frame is in the frames, then load the frame using frame_idx
                files_paths = [os.path.join(video_path, files[i]) for i in frame_idx]
                spare_frames = np.array([load_image(file) for file in files_paths])
            else:
                # If no frame is in the frames, then calculate the distance between the frames and the frame_idx
                distances = [[abs(frame - frame_idx_i)  for frame_idx_i in frame_idx] for frame in frames]
                # Get the minimum distance in all the distances
                #print(distances)
                min_distance = min([min(distance) for distance in distances])
                #print(min_distance)
                # Find the frame that is closest to the frame_idx
                closest_frame = frames[[min(distance) for distance in distances].index(min_distance)]
                #print(closest_frame)
                replaced_frame_idx = frame_idx[distances[[min(distance) for distance in distances].index(min_distance)].index(min_distance)]
                #print(replaced_frame_idx)
                # Replace the closest frame with the replaced_frame_idx
                frame_idx[frame_idx.index(replaced_frame_idx)] = closest_frame
                # Load all the frames
                files_paths = [os.path.join(video_path, files[i]) for i in frame_idx]
                spare_frames = np.array([load_image(file) for file in files_paths])
        else:
            # Load all the frames
            files_paths = [os.path.join(video_path, file) for file in files]
            spare_frames = np.array([load_image(file) for file in files_paths])
        
    if return_type == 'tensor':
        return spare_frames
    elif return_type == 'path':
        return files_paths
    else:
        raise ValueError('Return type not supported')

def read_videoframes_qa_data(data_path, annotation_path, data_type):
    # Load the annotation data
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # Get the answers, video names, answer candidates, frames, video paths, labels, id
    answers = list(df['answer'])
    video_names = list(df['video_name'])
    ans_candidates = list(df['ans_candidates'])
    frames = list(df['frames']) if 'frames' in df.columns else None
    video_paths = [os.path.join(data_path, video_name) for video_name in video_names]
    labels = list(df['label'])
    data_types = [data_type] * len(df)
    questions = list(df['question'])
    qids = list(df['qid'])

    return {'answers': answers, 'ans_candidates': ans_candidates, 'data_paths': video_paths, 'labels': labels, 'data_types': data_types, 'questions': questions, 'qids':qids, 'extra2': frames}
