import fitz  # PyMuPDF
import cv2
import numpy as np
from pathlib import Path
import json
from PIL import Image
import pytesseract
import logging
import sys
import re
from os import listdir
from os.path import isfile, join
from setproctitle import setproctitle

setproctitle("PDF_TEXT_EXTRACT")

class PDFTextExtractor:
    def __init__(self, tesseract_cmd='tesseract'):
        """
        Инициализация экстрактора текста с использованием Tesseract OCR
        :param tesseract_cmd: путь к исполняемому файлу tesseract
        """
        # Настраиваем логирование
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('pdf_text_extraction.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Настраиваем Tesseract
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.logger.info("Tesseract OCR инициализирован")
        
    def preprocess_image(self, img_array):
        """
        Предобработка изображения для улучшения качества OCR
        """
        # Конвертируем в оттенки серого
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Применяем адаптивную пороговую обработку
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Увеличиваем контрастность
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Применяем размытие для уменьшения шума
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        return denoised
        
    def clean_text(self, text):
        """
        Очистка текста от переносов строк и лишних пробелов
        """
        # Заменяем переносы строк на пробелы
        text = text.replace('\n', ' ')
        
        # Убираем множественные пробелы
        text = re.sub(r'\s+', ' ', text)
        
        # Убираем пробелы в начале и конце
        text = text.strip()
        
        return text
        
    def extract_text_from_pdf(self, pdf_path, output_dir, lang='rus+eng+chv'):
        """
        Извлечение текста из PDF с помощью OCR
        :param pdf_path: путь к PDF файлу
        :param output_dir: директория для сохранения результатов
        :param lang: языки для OCR (по умолчанию русский, английский и чувашский)
        """
        self.logger.info(f"Начинаем обработку файла: {pdf_path}")
        
        # Создаем директории для сохранения результатов
        pdf_name = Path(pdf_path).stem
        output_dir = Path(output_dir) / pdf_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Открываем PDF
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        self.logger.info(f"PDF файл открыт. Всего страниц: {total_pages}")
        
        # Словарь для хранения всех текстов
        all_texts = {
            'pdf_name': Path(pdf_path).name,
            'total_pages': total_pages,
            'pages': []
        }
        
        try:
            for page_num in range(total_pages):
                self.logger.info(f"Обработка страницы {page_num + 1} из {total_pages}")
                page = doc[page_num]
                
                # Получаем изображение страницы с увеличенным разрешением
                pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img_np = np.array(img)
                
                # Предобработка изображения
                processed_img = self.preprocess_image(img_np)
                
                # Распознаем текст
                text = pytesseract.image_to_string(
                    processed_img, 
                    lang=lang,
                    config='--psm 6'  # Предполагаем единый блок текста
                )
                
                # Очищаем текст
                text = self.clean_text(text)
                
                # Сохраняем текст страницы
                page_text = {
                    'page_number': page_num + 1,
                    'text': text
                }
                all_texts['pages'].append(page_text)
                
                self.logger.info(f"Текст страницы {page_num + 1} обработан")
                
                # Очищаем память
                del img_np
                del img
                del pix
                
        finally:
            # Закрываем PDF
            doc.close()
            
            # Сохраняем все тексты в один JSON файл
            json_file = output_dir / f'{pdf_name}_texts.json'
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(all_texts, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Все тексты сохранены в {json_file}")

if __name__ == '__main__':
    # Пример использования
    folder = "/home/daniilak/library_data/data/lib_files/0/chkt"
    for filename in [f for f in listdir(folder) if isfile(join(folder, f))]:
        # if filename != "chkt_0_0000237.pdf":
            # continue
        extractor = PDFTextExtractor()
        extractor.extract_text_from_pdf(
            pdf_path=f'{folder}/{filename}',
            output_dir='output_chkt_text',
            lang='rus+eng+chv'
        )