# Библиотека для извлечения изображений и текста из PDF

Этот проект представляет собой набор инструментов для извлечения изображений и текста из PDF-документов с использованием различных методов компьютерного зрения и обработки изображений.

## Возможности

- Извлечение текста из PDF с помощью Tesseract OCR
- Извлечение изображений с использованием трех различных подходов:
  - YOLOv8 для детекции объектов
  - YOLO-World для детекции объектов с открытым словарем
  - Segment Anything Model (SAM) для сегментации изображений
- Предобработка изображений для улучшения качества
- Сохранение результатов в структурированном формате JSON
- Подробное логирование процесса обработки

## Требования

- Python 3.8+
- PyTorch 2.0.0+
- CUDA (опционально, для ускорения работы моделей)

## Установка

1. Клонируйте репозиторий:
```bash
git clone [url-репозитория]
cd [директория-проекта]
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Скачайте необходимые модели:
- YOLOv8 модели (yolov8n.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
- SAM модель (sam_vit_h_4b8939.pth)
- YOLO-World модель (yolo_world_l.pt)

## Структура проекта

- `pdf_text_extractor.py` - Извлечение текста с помощью Tesseract OCR
- `pdf_image_extractor.py` - Извлечение изображений с помощью YOLOv8
- `pdf_image_extractor_sam.py` - Извлечение изображений с помощью SAM
- `pdf_image_extractor_yolo.py` - Извлечение изображений с помощью YOLO-World
- `requirements.txt` - Зависимости проекта

## Использование

### Извлечение текста

```python
from pdf_text_extractor import PDFTextExtractor

extractor = PDFTextExtractor()
extractor.extract_text_from_pdf(
    pdf_path='путь/к/файлу.pdf',
    output_dir='выходная/директория',
    lang='rus+eng+chv'  # Поддерживаемые языки
)
```

### Извлечение изображений с YOLOv8

```python
from pdf_image_extractor import PDFImageExtractor

extractor = PDFImageExtractor(model_path='yolov8l.pt')
extractor.extract_images_from_pdf(
    pdf_path='путь/к/файлу.pdf',
    output_dir='выходная/директория',
    min_confidence=0.2
)
```

### Извлечение изображений с SAM

```python
from pdf_image_extractor_sam import PDFImageExtractorSAM

extractor = PDFImageExtractorSAM(
    model_path='sam_vit_h_4b8939.pth',
    model_type='vit_h'
)
extractor.extract_images_from_pdf(
    pdf_path='путь/к/файлу.pdf',
    output_dir='выходная/директория',
    min_confidence=0.2
)
```

### Извлечение изображений с YOLO-World

```python
from pdf_image_extractor_yolo import PDFImageExtractorYOLO

extractor = PDFImageExtractorYOLO(
    model_path='yolo_world_l.pt',
    model_type='l'
)
extractor.extract_images_from_pdf(
    pdf_path='путь/к/файлу.pdf',
    output_dir='выходная/директория',
    min_confidence=0.2
)
```

## Выходные данные

Для каждого обработанного PDF-файла создается отдельная директория, содержащая:
- Извлеченные изображения в формате JPG
- JSON-файл с аннотациями, содержащий:
  - Информацию о PDF-файле
  - Координаты найденных изображений
  - Уверенность в детекции
  - Классы объектов (для YOLO)
  - Номера страниц

## Логирование

Все операции логируются в файлы:
- `pdf_text_extraction.log` - для операций с текстом
- `pdf_extraction.log` - для операций с изображениями

## Примечания

- Для работы с русским текстом требуется установленный Tesseract OCR с поддержкой русского языка
- Рекомендуется использовать GPU для ускорения работы моделей
- Минимальные требования к памяти зависят от используемой модели:
  - YOLOv8n: ~2GB RAM
  - YOLOv8l: ~8GB RAM
  - SAM: ~16GB RAM
  - YOLO-World: ~8GB RAM 