from langdetect import detect
from deep_translator import GoogleTranslator
from typing import List

def detect_and_translate(text: str) -> str:
    """
    Detect text language and translate to English if it's Vietnamese.
    Returns the original text if it's already in English.
    """
    if not text:
        return text
        
    try:
        lang = detect(text)
        if lang == 'vi':
            translator = GoogleTranslator(source='vi', target='en')
            return translator.translate(text)
        return text
    except Exception as e:
        print(f"❌ Translation error: {str(e)}")
        return text

def translate_list(items: List[str]) -> List[str]:
    """
    Translate a list of strings from Vietnamese to English if needed.
    """
    if not items:
        return items
        
    translated_items = []
    for item in items:
        translated_items.append(detect_and_translate(item))
    return translated_items 