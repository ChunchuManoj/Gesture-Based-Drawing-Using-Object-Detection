from ultralytics import YOLO

def train_yolo():
    model = YOLO('model\yolo11m.pt')  

    model.train(
        data='data.yaml',
        epochs=80,              # Increased for better convergence
        imgsz=640,
        batch=10,                # Increase batch size if GPU has enough memory
        device='cpu',                # Use GPU (0) if available
        lr0=0.0005,              # Adjusted learning rate
        optimizer='AdamW',       # AdamW is often better than Adam
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        flipud=0.2,              # Reduced vertical flip probability
        fliplr=0.5,
        mosaic=1.0,              # Enable mosaic augmentation (useful for small datasets)
        mixup=0.2                # Helps improve robustness
    )

if __name__ == '__main__':
    train_yolo()
