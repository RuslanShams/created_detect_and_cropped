import uuid
from ultralytics import YOLO, SAM
import numpy as np
from PIL import Image, ImageDraw

def detect_and_segment(file_pic):
    model_yolo = YOLO('yolov8s.pt')
    results = model_yolo.predict(source=file_pic, conf=0.25)
    res_box = results[0].boxes.xyxy.tolist()[0]
    model_sam = SAM('sam_b.pt')
    segmentation_result = model_sam.predict(file_pic, bboxes=res_box)
    return np.array(segmentation_result[0].masks.xy)


def cropped_image(file_pic, coords):
    new_img = f'{uuid.uuid4()}.png'
    image = Image.open(file_pic)
    mask = Image.new('L', image.size)
    draw = ImageDraw.Draw(mask)
    draw.polygon([tuple(point) for point in coords[0]], fill=255)
    result = Image.new('RGBA', image.size, (0, 0, 0, 0))
    result.paste(image, mask=mask)
    result.save(new_img)
    return new_img
