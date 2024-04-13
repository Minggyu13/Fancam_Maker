# import cv2

#코드를 test 해보는 파일입니다.
#
# video_path = 'ive.mp4'
# cap = cv2.VideoCapture(video_path)
#
# while True:
#     ret, frame = cap.read()
#
#     if not ret:
#         print("영상을 받아오지 못했습니다.")
#         break
#
#     cv2.imshow('img2', frame)
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

from moviepy.editor import VideoFileClip

# 원본 비디오와 이미 처리된 비디오 경로
original_video_path = 'ive_4k.mkv'
processed_video_path = 'ive_4k_output.mkv'
output_video_path = 'ive_output_video.mkv'

# 원본 비디오의 오디오 추출
original_video_clip = VideoFileClip(original_video_path)
audio_clip = original_video_clip.audio

# 이미 처리된 비디오에 오디오 삽입
processed_video_clip = VideoFileClip(processed_video_path)
processed_video_clip = processed_video_clip.set_audio(audio_clip)

# 여기에 비트레이트 설정 추가
# 예를 들어, 5000k는 4K 영상에 대해 비교적 낮은 값일 수 있으니, 상황에 맞게 조정이 필요합니다.
processed_video_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac', bitrate='30000k')

# 작업이 끝나면 임시로 생성된 파일을 닫을 수 있습니다.
processed_video_clip.close()
original_video_clip.close()
