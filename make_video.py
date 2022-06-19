import cv2
import os
import progressbar


path_in = 'recordings'
path_out = ''
name = 'test_video_1.mp4'
fps = 20.0

def make_video(path_in, path_out, name):
    frames = []

    print('Scanning folder')
    bar = progressbar.ProgressBar(maxval=len(os.listdir(path_in))).start()

    for file in sorted(os.listdir(path_in)):
        bar.update(len(frames))
        frames.append(cv2.imread('{}/{}'.format(path_in, file)))

    bar.finish()

    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter('{}{}'.format(path_out, name), fourcc, fps, (w, h), isColor=len(frames[0].shape) > 2)

    print('Creating video')
    bar = progressbar.ProgressBar(maxval=len(frames)).start()

    for i, frame in enumerate(frames):
        bar.update(i)
        writer.write(frame)

    bar.finish()


if __name__ == '__main__':
    make_video(path_in, path_out, name)
    print('Complete')
