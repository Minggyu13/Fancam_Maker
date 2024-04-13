import cv2
import os
#동영상에서 데이터를 수집하기 위한 코드입니다.
def save_frames(video_path, output_folder, interval=10, total_image = 10):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # 프레임의 너비, 높이 및 프레임 속도 가져오기
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 저장할 폴더가 없다면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0
    image_count = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # 일정한 간격으로 프레임 저장
        if frame_count % (fps * interval) == 0:
            frame_name = f"{video_path.split('.')[0]}_frame_{int(frame_count // (fps * interval)):04d}.png"
            frame_path = os.path.join(output_folder, frame_name)
            cv2.imwrite(frame_path, frame)
            print(f"Saved frame {frame_count} as {frame_name}")
            image_count += 1
        if image_count == total_image:
            break

    cap.release()




if __name__ == "__main__":
    video_path = 'dddddddd.mp4'  # 비디오 파일 경로 지정
    output_folder = "output_frames"  # 프레임을 저장할 폴더 지정

    save_frames(video_path, output_folder, interval=5, total_image=30)