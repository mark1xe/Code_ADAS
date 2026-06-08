from ultralytics import YOLO

# 1. Load lại trọng số tốt nhất mà bạn vừa train xong
model = YOLO('D:/KLTN/opencv/export/v8n_fix/output/best.pt') 

# 2. Export với opset 12 và simplify để OpenCV đọc được
model.export(format='onnx', imgsz=320, opset=12, simplify=True)