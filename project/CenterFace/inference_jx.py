import sys
import os
import traceback

# 添加当前目录到 Python 模块搜索路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import cv2
import json
import numpy as np
from centerface import CenterFace
from byte_tracker import BYTETracker

# 支持的图片扩展名
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

# BYTETrack 参数配置
class TrackerArgs:
    track_thresh = 0.35  # 适配 CenterFace 的检测阈值
    track_buffer = 30
    match_thresh = 0.8
    mot20 = False

def compute_iou(box1, box2):
    """
    计算两个边界框的 IOU
    box: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def process_image_sequence(image_dir, output_dir):
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Output directory created: {output_dir}")

    # 获取图片列表并按文件名排序，排除以 _det.jpg 结尾的文件
    image_list = sorted([f for f in os.listdir(image_dir) 
                        if f.lower().endswith(IMAGE_EXTENSIONS) and not f.endswith('_det.jpg')])
    print(f"Found {len(image_list)} images in {image_dir}: {image_list[:5]}...")  # 打印前5个文件名

    # 初始化 CenterFace 和 BYTETracker
    landmarks = True
    centerface = CenterFace(landmarks=landmarks)
    tracker = BYTETracker(TrackerArgs())

    try:
        for frame_id, image_name in enumerate(image_list):
            image_path = os.path.join(image_dir, image_name)
            print(f"Processing frame {frame_id}: {image_path}")

            # 读取图片
            src_img = cv2.imread(image_path)
            if src_img is None:
                print(f"Warning: Could not read image {image_path}")
                continue

            h, w = src_img.shape[:2]

            # 准备标注数据结构
            annotation = {
                "frame_id": frame_id,
                "image_name": image_name,
                "image_width": w,
                "image_height": h,
                "faces": []
            }

            # 进行人脸检测
            if landmarks:
                dets, lms = centerface(src_img, h, w, threshold=0.35)
            else:
                dets = centerface(src_img, threshold=0.35)

            # 确保 dets 不为空
            if len(dets) == 0:
                print(f"No faces detected in {image_name}")
            else:
                # 转换为 BYTETrack 输入格式 [x1, y1, x2, y2, score]
                output_results = dets[:, :5]  # [x1, y1, x2, y2, score]
                img_info = [h, w]
                img_size = [h, w]

                # 更新跟踪器
                online_targets = tracker.update(output_results, img_info, img_size)
                print(f"Number of tracked targets: {len(online_targets)}")

                # 处理每个检测到的人脸
                for t, det in enumerate(dets):
                    x1, y1, x2, y2, score = det[:5]

                    # 查找匹配的跟踪ID
                    track_id = -1
                    for track in online_targets:
                        tlbr = track.tlbr  # [x1, y1, x2, y2]
                        iou = compute_iou([x1, y1, x2, y2], tlbr)
                        if iou > 0.5:
                            track_id = track.track_id
                            break

                    face_info = {
                        "track_id": int(track_id),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": float(score)
                    }

                    if landmarks:
                        face_info["landmarks"] = []
                        for j in range(5):  # 5个关键点
                            face_info["landmarks"].append([
                                float(lms[t][j*2]),
                                float(lms[t][j*2+1])
                            ])

                    annotation["faces"].append(face_info)

                    # 在图片上绘制检测和跟踪结果
                    text_y = max(int(y1) - 5, 10)  # 确保文本不超出图像顶部
                    cv2.putText(src_img, f'ID:{track_id} S:{score:.2f}', 
                                (int(x1), text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, (255,0,0), 1)
                    cv2.rectangle(src_img, (int(x1), int(y1)), 
                                  (int(x2), int(y2)), (255,0,0), 2)
                    print(f"Drawing on image: ID={track_id}, Score={score:.2f}, BBox=[{x1},{y1},{x2},{y2}]")

            # 绘制关键点
            if landmarks and len(dets) > 0:
                for lm in lms:
                    for i in range(5):
                        color = (0,255,0) if i%3==0 else (0,255,255) if i%3==2 else (0,0,255)
                        cv2.circle(src_img, (int(lm[i*2]), int(lm[i*2+1])), 2, color, -1)

            # 保存标注后的图片
            output_img_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + '_det.jpg')
            cv2.imwrite(output_img_path, src_img)
            print(f"Saved detected image: {output_img_path}")

            # 保存单个 JSON 文件
            json_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + '_det.json')
            print(f"Attempting to save annotation to: {json_path}")
            try:
                with open(json_path, 'w') as f:
                    json.dump(annotation, f, indent=2)
                print(f"Saved annotation: {json_path}")
            except Exception as e:
                print(f"Failed to save annotation for {image_name}: {str(e)}")
                print(traceback.format_exc())

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        print(traceback.format_exc())
        raise

    print("All processing completed!")

if __name__ == "__main__":
    # 输入和输出目录（相对于脚本所在目录）
    input_dir = os.path.join('..', '..', 'samples', 'cri_images')
    output_dir = os.path.join('..', '..', 'samples', 'cri_images_output')
    process_image_sequence(input_dir, output_dir)