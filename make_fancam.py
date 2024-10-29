import cv2
import numpy as np
from idol_recognition_and_localizer import idol_classifier
import os
from moviepy.editor import VideoFileClip

# 보기 = [oneyoung, ray, riz, ujin, fall, iseo]
idol_name = input("만들고 싶은 직캠의 이름을 보기중 선택해서 입력해주세요 : ")

video_path = 'ive_4k_2.mkv'
cap = cv2.VideoCapture(video_path)

output_size = (375, 667) # (width, height)
fit_to = 'height'

# initialize writing video
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('%s_output.mkv' % (video_path.split('.')[0]), fourcc, cap.get(cv2.CAP_PROP_FPS), output_size)

# check file is opened
if not cap.isOpened():
  exit()

# initialize tracker
OPENCV_OBJECT_TRACKERS = {
  "csrt": cv2.TrackerCSRT_create,
  "kcf": cv2.legacy.TrackerKCF_create,
  "mil": cv2.legacy.TrackerMIL_create,
}


tracker = OPENCV_OBJECT_TRACKERS['mil']()

top_bottom_list, left_right_list = [], []
count = 0

# main
ret, img = cap.read()

while True:
  count += 1
  ret, img = cap.read()


  if count == 1:
    output_folder = 'frame_image'
    frame_name = f"{video_path.split('.')[0]}_frame_{int(count):04d}.jpg"
    frame_path = os.path.join(output_folder, frame_name)
    cv2.imwrite(frame_path, img)
    print(f"Saved frame {count} as {frame_name}")
    idol = idol_classifier(frame_path)
    idol_extracted_face = idol.extract_face()
    extracted_face = idol_extracted_face[0]
    face_box = idol_extracted_face[1]
    idol_face_box = None
    for i in range(len(face_box)):

      infer_name = idol.svm_infer(extracted_face[i])
      print(infer_name)

      if infer_name == idol_name:

        idol_face_box = face_box[i]
        #1080p일 때의 박스의 크기 (리팩토링 필요.)
        # idol_face_box[0] -= 25
        # idol_face_box[1] -= 25
        # idol_face_box[2] += 95
        # idol_face_box[3] += 465
        # 4K일 때의 박스 크기
        idol_face_box[0] -= 60
        idol_face_box[1] -= 40
        idol_face_box[2] += 165
        idol_face_box[3] += 865
        break



    if idol_face_box == None: #첫번째 프레임에서 찾고자 하는 아이돌이 없을 때 아이돌을 찾을 때까지 1프레임씩 넘기면서 아이돌이 나타날 때까지 반복합니다.(다른 방법 개선 필요)
      count -= 1
      continue

    try:
      tracker.init(img, idol_face_box)
    except cv2.error as e:
      # 오류 메시지 출력
      print("cv2 error bad allocation", e)
    print(idol_face_box)


  try:
    success, box = tracker.update(img)

  except cv2.error as e:
    # 오류 메시지 출력
    print("cv2 error bad allocation", e)

  print(count)
  if count % 50 == 0:
    # 아이돌을 찾았을 때
    # 아이돌이 겹처지거나 모종의 이유로 못 찾았을 때
    # 예상 못한 label을 예측 했을 때
    output_folder = 'frame_image'
    frame_name = f"{video_path.split('.')[0]}_frame_{int(count):04d}.jpg"
    frame_path = os.path.join(output_folder, frame_name)
    cv2.imwrite(frame_path, img)
    print(f"Saved frame {count} as {frame_name}")
    idol = idol_classifier(frame_path)
    idol_extracted_face = idol.extract_face()
    extracted_face = idol_extracted_face[0]
    face_box = idol_extracted_face[1]

    for i in range(len(face_box)):
      try:
        infer_name = idol.svm_infer(extracted_face[i])

      except ValueError as e:
        # 오류 메시지 출력
        print("확인하지 못한 레이블입니다. (얼굴 가려짐, 겹처짐):", e)
        success, box = tracker.update(img)
        # 오류를 무시하고 다음으로 진행
        continue  # 또는 다른 적절한 처리를 추가할 수 있음
      if infer_name == idol_name:
        print(infer_name)
        idol_face_box = face_box[i]
        #1080p일 떄의 박스 크기 if문으로 바꿀 필요 있음
        # idol_face_box[0] -= 25
        # idol_face_box[1] -= 25
        # idol_face_box[2] += 95
        # idol_face_box[3] += 465

        #4K일 때의 박스 크기
        idol_face_box[0] -= 60
        idol_face_box[1] -= 40
        idol_face_box[2] += 165
        idol_face_box[3] += 865
        print(idol_face_box)
        break

    box = idol_face_box
    del tracker
    tracker = OPENCV_OBJECT_TRACKERS['mil']()
    try:
      tracker.init(img, idol_face_box)
    except cv2.error as e:
      # 오류 메시지 출력
      print("cv2 error bad allocation", e)
      # 추가 작업 수행 또는 예외 처리 방법에 따라 다르게 처리할 수 있습니다.

    print(box)


  # if success:
  left, top, w, h = [int(v) for v in box]
  w, h = 270, 1000 # 너비와 높이 고정

  right = left + w
  bottom = top + h

  # save sizes of image
  top_bottom_list.append(np.array([top, bottom]))
  left_right_list.append(np.array([left, right]))

  # use recent 10 elements for crop (window_size=10)
  if len(top_bottom_list) > 10:
    del top_bottom_list[0]
    del left_right_list[0]

  # compute moving average
  avg_height_range = np.mean(top_bottom_list, axis=0).astype(int)
  avg_width_range = np.mean(left_right_list, axis=0).astype(int)
  avg_center = np.array([np.mean(avg_width_range), np.mean(avg_height_range)]) # (x, y)

  # compute scaled width and height
  scale = 1.3
  avg_height = (avg_height_range[1] - avg_height_range[0]) * scale
  avg_width = (avg_width_range[1] - avg_width_range[0]) * scale

  # compute new scaled ROI
  avg_height_range = np.array([avg_center[1] - avg_height / 2, avg_center[1] + avg_height / 2])
  avg_width_range = np.array([avg_center[0] - avg_width / 2, avg_center[0] + avg_width / 2])

  # fit to output aspect ratio
  if fit_to == 'width':
    avg_height_range = np.array([
      avg_center[1] - avg_width * output_size[1] / output_size[0] / 2,
      avg_center[1] + avg_width * output_size[1] / output_size[0] / 2
    ]).astype(int).clip(0, 9999)

    avg_width_range = avg_width_range.astype(int).clip(0, 9999)
  elif fit_to == 'height':
    avg_height_range = avg_height_range.astype(int).clip(0, 9999)

    avg_width_range = np.array([
      avg_center[0] - avg_height * output_size[0] / output_size[1] / 2,
      avg_center[0] + avg_height * output_size[0] / output_size[1] / 2
    ]).astype(int).clip(0, 9999)

  # print(f'avg_height_range : {avg_height_range} | avg_width_range : {avg_width_range} | avg_cener : {avg_center}')

  result_img = img[avg_height_range[0]:avg_height_range[1], avg_width_range[0]:avg_width_range[1]].copy()

 # resize image to output size
  try:
    result_img = cv2.resize(result_img, output_size,interpolation=cv2.INTER_CUBIC)#interpolation=cv2.INTER_CUBIC
  except cv2.error as e:
    # 오류 메시지 출력
    print("cv2 error ssize.empty error_resize_resize", e)



  # visualize
  pt1 = (int(left), int(top))
  pt2 = (int(right), int(bottom))
  cv2.rectangle(img, pt1, pt2, (255, 255, 255), 3)

  try:
    cv2.imshow('img2', img)
    cv2.imshow('result', result_img)

  except cv2.error as e:
    # 오류 메시지 출력
    print("cv2 error ssize.empty error imshow", e)
    


  # write video
  out.write(result_img)
  if cv2.waitKey(1) == ord('q'):
    break


# release everything
cap.release()
out.release()
cv2.destroyAllWindows()
#

original_video_path = 'ive_4k_2.mkv'
processed_video_path = 'ive_4k_2_output.mkv'
output_video_path = 'ive_output_video.mkv'

# 원본 비디오의 오디오 추출
original_video_clip = VideoFileClip(original_video_path)
audio_clip = original_video_clip.audio

# 이미 처리된 비디오에 오디오 삽입
processed_video_clip = VideoFileClip(processed_video_path)
processed_video_clip = processed_video_clip.set_audio(audio_clip)


processed_video_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac', bitrate='30000k')

processed_video_clip.close()
original_video_clip.close()
