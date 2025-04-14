from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
import io
import base64
from ultralytics import YOLO
import cv2

# import torchvision.transforms as transforms

app = FastAPI() 
model = None

# Загрузка вашей модели 
def load_model():
    """Загрузка YOLO модели"""
    try:
        model = YOLO('best.pt')  # Загрузка предобученной модели
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# Вызывается при старте приложения
@app.on_event("startup")
async def startup_event():
    global model
    model = load_model()  # Загружаем модель при запуске
    print("Model loaded successfully")

def process_image(image: Image.Image):
    """Обработка изображения с использованием model.predict()"""
    try:
        # Используем predict для получения результатов
        results = model.predict(image)
        
        # Визуализация результатов (с боксами)
        annotated_img = results[0].plot()
        
        # Конвертация обратно в PIL Image (RGB)
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(annotated_img)
    
    except Exception as e:
        print(f"Error processing image: {e}")
        raise

@app.post("/image-to-image/")
async def image_to_image(file: UploadFile = File(...)):
    try:
        # Чтение изображения
        image = Image.open(file.file)
        
        # Обработка изображения
        processed_image = process_image(image)
        
        # Сохранение временного файла
        temp_path = "temp_processed.png"
        processed_image.save(temp_path)
        
        return FileResponse(temp_path, media_type="image/png")
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/image-to-image-base64/")
async def image_to_image_base64(file: UploadFile = File(...)):
    try:
        # Чтение изображения
        image = Image.open(file.file)
        
        # Обработка изображения
        processed_image = process_image(image)
        
        # Конвертация в base64
        buffered = io.BytesIO()
        processed_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return JSONResponse({"image": img_str})
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/base64-to-base64/")
async def base64_to_base64(data: dict):
    try:
        if "image" not in data:
            raise HTTPException(status_code=400, detail="No image provided in base64 format")
        
        # Декодирование base64
        img_bytes = base64.b64decode(data["image"])
        image = Image.open(io.BytesIO(img_bytes))
        
        # Обработка изображения
        processed_image = process_image(image)
        
        # Конвертация обратно в base64
        buffered = io.BytesIO()
        processed_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return JSONResponse({"image": img_str})
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))