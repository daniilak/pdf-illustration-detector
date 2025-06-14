import fitz  # PyMuPDF
import cv2
import numpy as np
from pathlib import Path
import json
from ultralytics import YOLO
from PIL import Image
from os import listdir
from os.path import isfile, join
import shutil

class PDFImageExtractor:
    def __init__(self, model_path='yolov8l.pt'):
        """
        Инициализация экстрактора изображений
        :param model_path: путь к модели YOLOv8
        """
        self.model = YOLO(model_path)
        # Определяем интересующие нас классы
        self.target_classes = {
            # Люди
            'person': [
                'person', 'human', 'man', 'woman', 'boy', 'girl', 'child', 'adult',
                'male', 'female', 'people', 'crowd', 'group', 'family', 'couple',
                'soldier', 'officer', 'worker', 'peasant', 'noble', 'priest',
                'monk', 'nun', 'student', 'teacher', 'doctor', 'nurse',
                'artist', 'musician', 'writer', 'scientist', 'engineer',
                'merchant', 'trader', 'farmer', 'hunter', 'fisherman',
                'guard', 'servant', 'maid', 'butler', 'cook', 'driver',
                'pilot', 'sailor', 'soldier', 'officer', 'general', 'king',
                'queen', 'prince', 'princess', 'noble', 'aristocrat'
            ],
            # Здания и архитектура
            'building': [
                'building', 'house', 'church', 'tower', 'bridge', 'architecture', 
                'monument', 'temple', 'castle', 'palace', 'wall', 'structure',
                'facade', 'construction', 'edifice', 'architecture', 'building',
                'house', 'church', 'tower', 'bridge', 'monument', 'temple',
                'castle', 'palace', 'wall', 'structure', 'facade', 'construction',
                'edifice', 'architecture', 'building', 'house', 'church', 'tower',
                'bridge', 'monument', 'temple', 'castle', 'palace', 'wall',
                'structure', 'facade', 'construction', 'edifice'
            ],
            # Карты и схемы
            'map': ['map', 'chart', 'diagram', 'graph', 'blueprint', 'technical_drawing', 'layout', 'plan', 'scheme'],
            # Узоры и декоративные элементы
            'pattern': ['ornament', 'decoration', 'pattern', 'texture', 'design', 'artwork', 'art', 'drawing', 'illustration'],
            # Транспортные средства
            'vehicle': ['car', 'truck', 'motorcycle', 'bicycle', 'bus', 'train', 'airplane', 'boat'],
            # Дополнительные классы
            'other': ['image', 'photo', 'picture', 'illustration']
        }
        
        print("Available YOLOv8 classes:", self.model.names)
        
    def is_target_object(self, class_name, confidence):
        """
        Проверяет, является ли объект целевым
        """
        if class_name == 'book':
            return False
        print(class_name, confidence)
        for category, classes in self.target_classes.items():
            if class_name.lower() in classes and confidence > 0.2:  # Снижаем порог до 0.2
                return True
        return False
        
    def extract_images_from_pdf(self, pdf_path, output_dir, min_confidence=0.15):
        """
        Извлечение изображений из PDF с помощью YOLOv8
        :param pdf_path: путь к PDF файлу
        :param output_dir: директория для сохранения результатов
        :param min_confidence: минимальный порог уверенности для детекции
        """
        # Создаем директории для сохранения результатов
        pdf_name = Path(pdf_path).stem
        output_dir = Path(output_dir) / pdf_name
        images_dir = output_dir / 'images'
        detections_dir = output_dir / 'detections'  # Директория для изображений с разметкой
        images_dir.mkdir(parents=True, exist_ok=True)
        detections_dir.mkdir(parents=True, exist_ok=True)
        
        # Создаем временную директорию для страниц
        temp_dir = output_dir / 'temp'
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Открываем PDF
        doc = fitz.open(pdf_path)
        
        # Словарь для хранения всех аннотаций
        all_annotations = {
            'pdf_name': Path(pdf_path).name,
            'total_pages': len(doc),
            'detections': []
        }
        
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Получаем изображение страницы с увеличенным разрешением
                pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img_np = np.array(img)
                
                # Детектируем объекты
                results = self.model(img_np, conf=min_confidence)
                
                # Сохраняем изображение с разметкой
                if len(results) > 0 and len(results[0].boxes) > 0:
                    book_count = 0
                    other_objects_detected = False
                    for box in results[0].boxes:
                        cls = int(box.cls[0])
                        class_name = self.model.names[cls]
                        if class_name == 'book':
                            book_count += 1
                        else:
                            other_objects_detected = True

                    if not (book_count == 1 and not other_objects_detected):
                        result_img = results[0].plot()
                        cv2.imwrite(str(detections_dir / f'page_{page_num}_detection.jpg'), result_img)
                    
                    # Выводим все обнаруженные классы для отладки
                    detected_classes = set()
                    for box in results[0].boxes:
                        cls = int(box.cls[0])
                        class_name = self.model.names[cls]
                        conf = float(box.conf[0])
                        detected_classes.add(f"{class_name} ({conf:.2f})")
                    print(f"Page {page_num} detected classes: {', '.join(detected_classes)}")
                
                # Обрабатываем результаты
                img_idx = 1  # Счётчик для уникальных имён файлов на странице
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        class_name = self.model.names[cls]
                        
                        # Проверяем, является ли объект целевым
                        if self.is_target_object(class_name, conf):
                            # Вырезаем изображение с небольшим отступом
                            padding = 50  # Увеличиваем отступ до 50 пикселей
                            h, w = img_np.shape[:2]
                            x1 = max(0, int(x1) - padding)
                            y1 = max(0, int(y1) - padding)
                            x2 = min(w, int(x2) + padding)
                            y2 = min(h, int(y2) + padding)
                            
                            crop = img_np[y1:y2, x1:x2]
                            
                            # Проверяем размер вырезанного изображения
                            if crop.size > 0 and crop.shape[0] > 20 and crop.shape[1] > 20:
                                # Увеличиваем контрастность изображения
                                lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
                                l, a, b = cv2.split(lab)
                                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                                cl = clahe.apply(l)
                                enhanced = cv2.merge((cl,a,b))
                                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
                                
                                # Сохраняем изображение с уникальным именем
                                image_path = images_dir / f'page_{page_num}_img_{img_idx}_{class_name}.jpg'
                                img_idx += 1
                                cv2.imwrite(str(image_path), cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))
                                
                                # Добавляем аннотацию
                                annotation = {
                                    'page': page_num,
                                    'image_path': str(image_path.relative_to(output_dir)),
                                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                    'confidence': conf,
                                    'class': class_name,
                                    'category': next((cat for cat, classes in self.target_classes.items() 
                                                    if class_name.lower() in classes), 'other')
                                }
                                all_annotations['detections'].append(annotation)
                
                # Очищаем память
                del img_np
                del img
                del pix
                
        finally:
            # Закрываем PDF
            doc.close()
            
            # Сохраняем все аннотации в один файл
            annotations_file = output_dir / f'{Path(pdf_path).stem}_annotations.json'
            with open(annotations_file, 'w') as f:
                json.dump(all_annotations, f, indent=2)
            
            # Удаляем временную директорию
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

if __name__ == '__main__':
    # Пример использования
    folder = "~/library_data/data/lib_files/0/chkt"
    for filename in [f for f in listdir(folder) if isfile(join(folder, f))]:
        extractor = PDFImageExtractor()
        extractor.extract_images_from_pdf(
            pdf_path=f'{folder}/{filename}',
            output_dir='output_chkt',
            min_confidence=0.2
        ) 