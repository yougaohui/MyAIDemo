import cv2
import os


def extract_frames(video_path, output_folder, frame_rate):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_rate == 0:
            frame_name = os.path.join(output_folder, f"frame_{count}.jpg")
            cv2.imwrite(frame_name, frame)

        count += 1

    cap.release()
    cv2.destroyAllWindows()


video_path = './video/game.mp4'
output_folder = './output/folder'
frame_rate = 30  # 每 30 帧保存一帧
extract_frames(video_path, output_folder, frame_rate)
