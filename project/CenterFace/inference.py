import os
import cv2
import json
from centerface import CenterFace

# 支持的图片扩展名
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

# 获取图片列表并过滤非图片文件
image_list = [f for f in os.listdir('../../samples/cri_images') 
             if f.lower().endswith(IMAGE_EXTENSIONS)]

for image_name in image_list:
    image_path = os.path.join('../../samples/cri_images', image_name)
    print(f"Processing: {image_path}")
    
    # 读取图片
    src_img = cv2.imread(image_path)
    if src_img is None:
        print(f"Warning: Could not read image {image_path}")
        continue
        
    h, w = src_img.shape[:2]
    landmarks = True
    centerface = CenterFace(landmarks=landmarks)
    
    # 准备标注数据结构
    annotation = {
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

    # 处理每个检测到的人脸
    for i, det in enumerate(dets):
        boxes, score = det[:4], det[4]
        
        face_info = {
            "bbox": [float(boxes[0]), float(boxes[1]), 
                    float(boxes[2]), float(boxes[3])],
            "confidence": float(score)
        }
        
        if landmarks:
            face_info["landmarks"] = []
            for j in range(5):  # 5个关键点
                face_info["landmarks"].append([
                    float(lms[i][j*2]), 
                    float(lms[i][j*2+1])
                ])
        
        annotation["faces"].append(face_info)
        
        # 在图片上绘制检测结果
        cv2.putText(src_img, f'{score:.2f}', (int(boxes[0]), int(boxes[1])-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        cv2.rectangle(src_img, (int(boxes[0]), int(boxes[1])), 
                     (int(boxes[2]), int(boxes[3])), (255,0,0), 2)
    
    # 绘制关键点
    if landmarks:
        for lm in lms:
            for i in range(5):
                color = (0,255,0) if i%3==0 else (0,255,255) if i%3==2 else (0,0,255)
                cv2.circle(src_img, (int(lm[i*2]), int(lm[i*2+1])), 2, color, -1)
    
    # 保存标注后的图片
    output_img_path = os.path.splitext(image_path)[0] + '_det.jpg'
    cv2.imwrite(output_img_path, src_img)
    print(f"Saved detected image: {output_img_path}")
    
    # 保存标注文件(与图片同名，扩展名为.json)
    json_path = os.path.splitext(image_path)[0] + '.json'
    with open(json_path, 'w') as f:
        json.dump(annotation, f, indent=2)
    print(f"Saved annotation: {json_path}\n")

print("All processing completed!")