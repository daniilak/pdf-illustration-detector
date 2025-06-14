import fitz  # PyMuPDF
import cv2
import numpy as np
from pathlib import Path
import json
from PIL import Image
from os import listdir
from os.path import isfile, join
import shutil
import logging
import sys
from yoloworld import YOLOWorld
from setproctitle import setproctitle

setproctitle("PDF_IMAGE_EXTRACT_YOLO")

class PDFImageExtractorYOLO:
    def __init__(self, model_path='yolo_world_l.pt', model_type='l'):
        """
        Инициализация экстрактора изображений с использованием YOLO-World
        :param model_path: путь к модели YOLO-World
        :param model_type: тип модели YOLO-World (s, m, l, x)
        """
        # Настраиваем логирование
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('pdf_extraction_yolo.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Инициализация модели YOLO-World: {model_type}")
        self.model = YOLOWorld(model_path)
        self.logger.info("Модель YOLO-World успешно загружена")
        
        # Определяем интересующие нас классы
        self.target_classes = {
            # Люди
            'person': [
                'person', 'human', 'man', 'woman', 'boy', 'girl', 'child', 'adult',
                'male', 'female', 'people', 'crowd', 'group', 'family', 'couple'
            ],
            # Здания и архитектура
            'building': [
                'building', 'house', 'church', 'tower', 'bridge', 'architecture', 
                'monument', 'temple', 'castle', 'palace', 'wall', 'structure'
            ],
            # Карты и схемы
            'map': ['map', 'chart', 'diagram', 'graph', 'blueprint', 'technical_drawing'],
            # Узоры и декоративные элементы
            'pattern': ['ornament', 'decoration', 'pattern', 'texture', 'design', 'artwork'],
            # Транспортные средства
            'vehicle': ['car', 'truck', 'motorcycle', 'bicycle', 'bus', 'train', 'airplane', 'boat'],
            # Иллюстрации
            'illustration': ['picture', 'photo', 'image', 'painting'],
            # Дополнительные классы
            'other': ['image', 'photo', 'picture', 'illustration']
        }
        
    def is_text_region(self, img_array):
        """
        Определяет, является ли область текстовой
        """
        # Конвертируем в оттенки серого
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Применяем адаптивную пороговую обработку
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Находим контуры
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False
            
        # Анализируем контуры
        total_area = img_array.shape[0] * img_array.shape[1]
        text_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Игнорируем слишком маленькие контуры
                continue
                
            # Получаем прямоугольник, описывающий контур
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Текст обычно имеет определенное соотношение сторон
            if 0.1 < aspect_ratio < 10 and area > 100:
                text_contours.append(contour)
        
        # Если текстовые контуры занимают слишком большую часть области
        text_area = sum(cv2.contourArea(c) for c in text_contours)
        text_ratio = text_area / total_area
        
        # Проверяем равномерность распределения яркости
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Проверяем наличие градиентов
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Текст обычно имеет:
        # 1. Высокую плотность текстовых контуров
        # 2. Низкую энтропию (равномерное распределение яркости)
        # 3. Низкую плотность краёв
        is_text = (text_ratio > 0.2 and  # Уменьшаем порог для текстовых контуров
                  entropy < 5.0 and      # Текст имеет низкую энтропию
                  edge_density < 0.1)    # Текст имеет мало краёв
        
        if is_text:
            self.logger.debug(f"Область определена как текст: text_ratio={text_ratio:.2f}, entropy={entropy:.2f}, edge_density={edge_density:.2f}")
        
        return is_text
        
    def is_valid_image_region(self, x1, y1, x2, y2, img_shape, min_area=10000, max_aspect_ratio=3.0):
        """
        Проверяет, является ли область подходящей для изображения
        :param x1, y1, x2, y2: координаты области
        :param img_shape: размеры исходного изображения
        :param min_area: минимальная площадь области
        :param max_aspect_ratio: максимальное соотношение сторон
        :return: bool
        """
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        # Проверяем минимальный размер
        if area < min_area:
            self.logger.debug(f"Область слишком маленькая: {area} < {min_area}")
            return False
            
        # Проверяем соотношение сторон
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > max_aspect_ratio:
            self.logger.debug(f"Неподходящее соотношение сторон: {aspect_ratio:.2f} > {max_aspect_ratio}")
            return False
            
        # Проверяем, что область не слишком близка к краям страницы
        margin = 50
        if (x1 < margin or y1 < margin or 
            x2 > img_shape[1] - margin or 
            y2 > img_shape[0] - margin):
            self.logger.debug("Область слишком близко к краю страницы")
            return False
            
        # Проверяем, что область не слишком большая (не занимает большую часть страницы)
        page_area = img_shape[0] * img_shape[1]
        if area > page_area * 0.8:  # Не более 80% страницы
            self.logger.debug(f"Область слишком большая: {area} > {page_area * 0.8}")
            return False
            
        return True
        
    def extract_images_from_pdf(self, pdf_path, output_dir, min_confidence=0.15):
        """
        Извлечение изображений из PDF с помощью YOLO-World
        :param pdf_path: путь к PDF файлу
        :param output_dir: директория для сохранения результатов
        :param min_confidence: минимальный порог уверенности для детекции
        """
        self.logger.info(f"Начинаем обработку файла: {pdf_path}")
        
        # Создаем директории для сохранения результатов
        pdf_name = Path(pdf_path).stem
        output_dir = Path(output_dir) / pdf_name
        images_dir = output_dir / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Создаем временную директорию для страниц
        temp_dir = output_dir / 'temp'
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Открываем PDF
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        self.logger.info(f"PDF файл открыт. Всего страниц: {total_pages}")
        
        # Словарь для хранения всех аннотаций
        all_annotations = {
            'pdf_name': Path(pdf_path).name,
            'total_pages': total_pages,
            'images': []
        }
        
        try:
            for page_num in range(total_pages):
                self.logger.info(f"Обработка страницы {page_num + 1} из {total_pages}")
                page = doc[page_num]
                
                # Получаем изображение страницы с увеличенным разрешением
                pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img_np = np.array(img)
                
                self.logger.info(f"Размер изображения страницы: {img_np.shape}")
                
                # Получаем предсказания от YOLO-World
                results = self.model(img_np)
                
                # Обрабатываем результаты детекции
                valid_segments = 0
                for detection in results:
                    if detection['confidence'] > min_confidence:
                        # Получаем координаты области
                        x1, y1, x2, y2 = detection['bbox']
                        
                        # Проверяем, является ли область подходящей для изображения
                        if self.is_valid_image_region(x1, y1, x2, y2, img_np.shape):
                            # Вырезаем область
                            region = img_np[int(y1):int(y2), int(x1):int(x2)]
                            
                            # Проверяем, что область не пустая и не является текстом
                            if region.size > 0 and not self.is_text_region(region):
                                valid_segments += 1
                                # Увеличиваем контрастность изображения
                                lab = cv2.cvtColor(region, cv2.COLOR_RGB2LAB)
                                l, a, b = cv2.split(lab)
                                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                                cl = clahe.apply(l)
                                enhanced = cv2.merge((cl,a,b))
                                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
                                
                                # Сохраняем изображение
                                image_path = images_dir / f'page_{page_num}_image_{valid_segments}.jpg'
                                cv2.imwrite(str(image_path), cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))
                                
                                self.logger.info(f"Сохранено изображение: {image_path}")
                                
                                # Добавляем аннотацию
                                annotation = {
                                    'page': page_num,
                                    'image_path': str(image_path.relative_to(output_dir)),
                                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                    'confidence': float(detection['confidence']),
                                    'class': detection['class']
                                }
                                all_annotations['images'].append(annotation)
                
                self.logger.info(f"На странице {page_num + 1} найдено {valid_segments} валидных изображений")
                
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
            
            self.logger.info(f"Сохранены аннотации в файл: {annotations_file}")
            self.logger.info(f"Всего найдено {len(all_annotations['images'])} изображений")
            
            # Удаляем временную директорию
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                self.logger.info("Временная директория удалена")

if __name__ == '__main__':
    # Пример использования
    folder = "/home/daniilak/library_data/data/lib_files/0/chkt"
    for filename in [f for f in listdir(folder) if isfile(join(folder, f))]:
        # if filename != "chkt_0_0000237.pdf":
            # continue
        extractor = PDFImageExtractorYOLO()
        extractor.extract_images_from_pdf(
            pdf_path=f'{folder}/{filename}',
            output_dir='output_chkt_yolo',
            min_confidence=0.2
        )