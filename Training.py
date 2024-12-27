from ultralytics import YOLO


model = YOLO("yolov8m-pose.pt")  # load a pretrained model


results = model.train(data="config.yaml", epochs=100, imgsz=640, batch=32)
