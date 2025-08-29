# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import re
from typing import Any

from loguru import logger

from nemo_curator.stages.text.download import DocumentExtractor

# The majority of this code is taken from the HuggingFace
# implementation of the Wikipedia dataset preparation:
# https://github.com/huggingface/datasets/blob/7e30308f49f8c85dc7a2ab5aafbff04b5d2f38e2/datasets/wikipedia/wikipedia.py

MEDIA_ALIASES = {
    "ab": ["Медиа", "Файл", "Афаил", "Амедиа", "Изображение"],
    "ace": ["Beureukaih", "Gambar", "Alat", "Berkas"],
    "ady": ["Медиа"],
    "af": ["Lêer", "Beeld"],
    "als": ["Medium", "Datei", "Bild"],
    "am": ["ፋይል", "ስዕል"],
    "an": ["Imachen", "Imagen"],
    "ang": ["Ymele", "Biliþ"],
    "ar": ["ميديا", "صورة", "وسائط", "ملف"],
    "arc": ["ܠܦܦܐ", "ܡܝܕܝܐ"],
    "arz": ["ميديا", "صورة", "وسائط", "ملف"],
    "as": ["চিত্ৰ", "चित्र", "চিত্র", "মাধ্যম"],
    "ast": ["Imaxen", "Ficheru", "Imaxe", "Archivu", "Imagen", "Medios"],
    "atj": ["Tipatcimoctakewin", "Natisinahikaniwoc"],
    "av": ["Медиа", "Файл", "Изображение"],
    "ay": ["Medio", "Archivo", "Imagen"],
    "az": ["Mediya", "Şəkil", "Fayl"],
    "azb": ["رسانه", "تصویر", "مدیا", "فایل", "رسانه‌ای"],
    "ba": ["Медиа", "Рәсем", "Файл", "Изображение"],
    "bar": ["Medium", "Datei", "Bild"],
    "bat-smg": ["Vaizdas", "Medėjė", "Abruozdielis"],
    "bcl": ["Medio", "Ladawan"],
    "be": ["Мультымедыя", "Файл", "Выява"],
    "be-x-old": ["Мэдыя", "Файл", "Выява"],
    "bg": ["Медия", "Файл", "Картинка"],
    "bh": ["मीडिया", "चित्र"],
    "bjn": ["Barakas", "Gambar", "Berkas"],
    "bm": ["Média", "Fichier"],
    "bn": ["চিত্র", "মিডিয়া"],
    "bpy": ["ছবি", "মিডিয়া"],
    "br": ["Skeudenn", "Restr"],
    "bs": ["Mediji", "Slika", "Datoteka", "Medija"],
    "bug": ["Gambar", "Berkas"],
    "bxr": ["Файл", "Меди", "Изображение"],
    "ca": ["Fitxer", "Imatge"],
    "cbk-zam": ["Medio", "Archivo", "Imagen"],
    "cdo": ["文件", "媒體", "圖像", "檔案"],
    "ce": ["Хlум", "Медиа", "Сурт", "Файл", "Медйа", "Изображение"],  # noqa: RUF001
    "ceb": ["Payl", "Medya", "Imahen"],
    "ch": ["Litratu"],
    "ckb": ["میدیا", "پەڕگە"],
    "co": ["Immagine"],
    "crh": ["Медиа", "Resim", "Файл", "Fayl", "Ресим"],
    "cs": ["Soubor", "Média", "Obrázok"],
    "csb": ["Òbrôzk", "Grafika"],
    "cu": ["Видъ", "Ви́дъ", "Дѣло", "Срѣдьства"],
    "cv": ["Медиа", "Ӳкерчĕк", "Изображение"],
    "cy": ["Delwedd"],
    "da": ["Billede", "Fil"],
    "de": ["Medium", "Datei", "Bild"],
    "din": ["Ciɛl", "Apamduööt"],
    "diq": ["Medya", "Dosya"],
    "dsb": ["Wobraz", "Dataja", "Bild", "Medija"],
    "dty": ["चित्र", "मिडिया"],
    "dv": ["ފައިލު", "މީޑިއާ", "ފައިލް"],
    "el": ["Εικόνα", "Αρχείο", "Μέσο", "Μέσον"],
    "eml": ["Immagine"],
    "eo": ["Dosiero", "Aŭdvidaĵo"],
    "es": ["Medio", "Archivo", "Imagen"],
    "et": ["Pilt", "Fail", "Meedia"],
    "eu": ["Irudi", "Fitxategi"],
    "ext": ["Archivu", "Imagen", "Mediu"],
    "fa": ["رسانه", "تصویر", "مدیا", "پرونده", "رسانه‌ای"],
    "ff": ["Média", "Fichier"],
    "fi": ["Kuva", "Tiedosto"],
    "fiu-vro": ["Pilt", "Meediä"],
    "fo": ["Miðil", "Mynd"],
    "fr": ["Média", "Fichier"],
    "frp": ["Émâge", "Fichiér", "Mèdia"],
    "frr": ["Medium", "Datei", "Bild"],
    "fur": ["Immagine", "Figure"],
    "fy": ["Ofbyld"],
    "ga": ["Íomhá", "Meán"],
    "gag": ["Mediya", "Medya", "Resim", "Dosya", "Dosye"],
    "gan": ["媒体文件", "文件", "文檔", "档案", "媒體", "图像", "圖像", "媒体", "檔案"],
    "gd": ["Faidhle", "Meadhan"],
    "gl": ["Imaxe", "Ficheiro", "Arquivo", "Imagem"],
    "glk": ["رسانه", "تصویر", "پرونده", "فاىل", "رسانه‌ای", "مديا"],
    "gn": ["Medio", "Imagen", "Ta'ãnga"],
    "gom": ["माध्यम", "मिडिया", "फायल"],
    "gor": ["Gambar", "Berkas"],
    "got": ["𐍆𐌴𐌹𐌻𐌰"],
    "gu": ["દ્રશ્ય-શ્રાવ્ય (મિડિયા)", "દ્રશ્ય-શ્રાવ્ય_(મિડિયા)", "ચિત્ર"],
    "gv": ["Coadan", "Meanyn"],
    "hak": ["文件", "媒體", "圖像", "檔案"],
    "haw": ["Kiʻi", "Waihona", "Pāpaho"],  # noqa: RUF001
    "he": ["תמונה", "קו", "מדיה", "קובץ"],
    "hi": ["मीडिया", "चित्र"],
    "hif": ["file", "saadhan"],
    "hr": ["Mediji", "DT", "Slika", "F", "Datoteka"],
    "hsb": ["Wobraz", "Dataja", "Bild"],
    "ht": ["Imaj", "Fichye", "Medya"],
    "hu": ["Kép", "Fájl", "Média"],
    "hy": ["Պատկեր", "Մեդիա"],
    "ia": ["Imagine", "Multimedia"],
    "id": ["Gambar", "Berkas"],
    "ig": ["Nká", "Midia", "Usòrò", "Ákwúkwó orünotu", "Ákwúkwó_orünotu"],
    "ii": ["媒体文件", "文件", "档案", "图像", "媒体"],
    "ilo": ["Midia", "Papeles"],
    "inh": ["Медиа", "Файл", "Изображение"],
    "io": ["Imajo", "Arkivo"],
    "is": ["Miðill", "Mynd"],
    "it": ["Immagine"],
    "ja": ["メディア", "ファイル", "画像"],
    "jbo": ["velsku", "datnyvei"],
    "jv": ["Barkas", "Medhia", "Gambar", "Médhia"],
    "ka": ["მედია", "სურათი", "ფაილი"],
    "kaa": ["Swret", "Таспа", "سۋرەت", "Taspa", "Su'wret", "Сурет", "تاسپا"],
    "kab": ["Tugna"],
    "kbd": ["Медиа", "Файл"],
    "kbp": ["Média", "Fichier"],
    "kg": ["Fisye"],
    "kk": ["Swret", "سۋرەت", "Таспа", "Taspa", "Сурет", "تاسپا"],
    "kl": ["Billede", "Fiileq", "Fil"],
    "km": ["ឯកសារ", "រូបភាព", "មេឌា", "មីឌា"],
    "kn": ["ಚಿತ್ರ", "ಮೀಡಿಯ"],
    "ko": ["미디어", "파일", "그림"],
    "koi": ["Медиа", "Файл", "Изображение"],
    "krc": ["Медиа", "Файл", "Изображение"],
    "ks": ["میڈیا", "فَیِل"],
    "ksh": ["Beld", "Meedije", "Medie", "Belld", "Medium", "Datei", "Meedijum", "Bild"],
    "ku": ["میدیا", "پەڕگە", "Medya", "Wêne"],
    "kv": ["Медиа", "Файл", "Изображение"],
    "kw": ["Restren"],
    "ky": ["Медиа", "Файл"],
    "la": ["Imago", "Fasciculus"],
    "lad": ["Dossia", "Medya", "Archivo", "Dosya", "Imagen", "Meddia"],
    "lb": ["Fichier", "Bild"],
    "lbe": ["Медиа", "Сурат", "Изображение"],
    "lez": ["Медиа", "Mediya", "Файл", "Şəkil", "Изображение"],
    "lfn": ["Fix"],
    "li": ["Afbeelding", "Plaetje", "Aafbeilding"],
    "lij": ["Immaggine", "Immagine"],
    "lmo": ["Immagine", "Imàjine", "Archivi"],
    "ln": ["Média", "Fichier"],
    "lo": ["ສື່ອ", "ສື່", "ຮູບ"],
    "lrc": ["رسانه", "تصویر", "رسانه‌ای", "جانیا", "أسگ", "ڤارئسگأر"],
    "lt": ["Vaizdas", "Medija"],
    "ltg": ["Medeja", "Fails"],
    "lv": ["Attēls"],
    "mai": ["मेडिया", "फाइल"],
    "map-bms": ["Barkas", "Medhia", "Gambar", "Médhia"],
    "mdf": ["Медиа", "Няйф", "Изображение"],
    "mg": ["Rakitra", "Sary", "Média"],
    "mhr": ["Медиа", "Файл", "Изображение"],
    "min": ["Gambar", "Berkas"],
    "mk": ["Податотека", "Медија", "Медиум", "Слика"],
    "ml": ["പ്രമാണം", "ചി", "മീഡിയ", "പ്ര", "ചിത്രം"],
    "mn": ["Медиа", "Файл", "Зураг"],  # noqa: RUF001
    "mr": ["चित्र", "मिडिया"],
    "mrj": ["Медиа", "Файл", "Изображение"],
    "ms": ["Fail", "Imej"],
    "mt": ["Midja", "Medja", "Stampa"],
    "mwl": ["Multimédia", "Fexeiro", "Ficheiro", "Arquivo", "Imagem"],
    "my": ["ဖိုင်", "မီဒီယာ"],
    "myv": ["Медия", "Артовкс", "Изображение"],
    "mzn": ["رسانه", "تصویر", "مه‌دیا", "مدیا", "پرونده", "رسانه‌ای"],
    "nah": ["Mēdiatl", "Īxiptli", "Imagen"],
    "nap": ["Fiùra", "Immagine"],
    "nds": ["Datei", "Bild"],
    "nds-nl": ["Ofbeelding", "Afbeelding", "Bestaand"],
    "ne": ["मीडिया", "चित्र"],
    "new": ["किपा", "माध्यम"],
    "nl": ["Bestand", "Afbeelding"],
    "nn": ["Fil", "Bilde", "Filpeikar"],
    "no": ["Fil", "Medium", "Bilde"],
    "nov": [],
    "nrm": ["Média", "Fichier"],
    "nso": ["Seswantšho"],
    "nv": ["Eʼelyaaígíí"],  # noqa: RUF001
    "oc": ["Imatge", "Fichièr", "Mèdia"],
    "olo": ["Kuva", "Medii", "Failu"],
    "or": ["ମାଧ୍ୟମ", "ଫାଇଲ"],
    "os": ["Ныв", "Медиа", "Файл", "Изображение"],
    "pa": ["ਤਸਵੀਰ", "ਮੀਡੀਆ"],
    "pcd": ["Média", "Fichier"],
    "pdc": ["Medium", "Datei", "Bild", "Feil"],
    "pfl": ["Dadai", "Medium", "Datei", "Bild"],
    "pi": ["मीडिया", "पटिमा"],
    "pl": ["Plik", "Grafika"],
    "pms": ["Figura", "Immagine"],
    "pnb": ["میڈیا", "تصویر", "فائل"],
    "pnt": ["Εικόνα", "Αρχείον", "Εικόναν", "Μέσον"],
    "ps": ["انځور", "رسنۍ", "دوتنه"],
    "pt": ["Multimédia", "Ficheiro", "Arquivo", "Imagem"],
    "qu": ["Midya", "Imagen", "Rikcha"],
    "rm": ["Multimedia", "Datoteca"],
    "rmy": ["Fişier", "Mediya", "Chitro", "Imagine"],
    "ro": ["Fişier", "Imagine", "Fișier"],
    "roa-rup": ["Fişier", "Imagine", "Fișier"],
    "roa-tara": ["Immagine"],
    "ru": ["Медиа", "Файл", "Изображение"],
    "rue": ["Медіа", "Медиа", "Файл", "Изображение", "Зображення"],
    "rw": ["Dosiye", "Itangazamakuru"],
    "sa": ["चित्रम्", "माध्यमम्", "सञ्चिका", "माध्यम", "चित्रं"],
    "sah": ["Миэдьийэ", "Ойуу", "Билэ", "Изображение"],
    "sat": ["ᱨᱮᱫ", "ᱢᱤᱰᱤᱭᱟ"],
    "sc": ["Immàgini"],
    "scn": ["Immagine", "Mmàggini", "Mèdia"],
    "sd": ["عڪس", "ذريعات", "فائل"],
    "se": ["Fiila"],
    "sg": ["Média", "Fichier"],
    "sh": ["Mediji", "Slika", "Медија", "Datoteka", "Medija", "Слика"],
    "si": ["රූපය", "මාධ්‍යය", "ගොනුව"],
    "sk": ["Súbor", "Obrázok", "Médiá"],
    "sl": ["Slika", "Datoteka"],
    "sq": ["Figura", "Skeda"],
    "sr": [
        "Датотека",
        "Medij",
        "Slika",
        "Медија",
        "Datoteka",
        "Медиј",
        "Medija",
        "Слика",
    ],
    "srn": ["Afbeelding", "Gefre"],
    "stq": ["Bielde", "Bild"],
    "su": ["Média", "Gambar"],
    "sv": ["Fil", "Bild"],
    "sw": ["Faili", "Picha"],
    "szl": ["Plik", "Grafika"],
    "ta": ["படிமம்", "ஊடகம்"],
    "tcy": ["ಮಾದ್ಯಮೊ", "ಫೈಲ್"],
    "te": ["ఫైలు", "దస్త్రం", "బొమ్మ", "మీడియా"],
    "tet": ["Imajen", "Arquivo", "Imagem"],
    "tg": ["Акс", "Медиа"],
    "th": ["ไฟล์", "สื่อ", "ภาพ"],
    "ti": ["ፋይል", "ሜድያ"],
    "tk": ["Faýl"],
    "tl": ["Midya", "Talaksan"],
    "tpi": ["Fail"],
    "tr": ["Medya", "Resim", "Dosya", "Ortam"],
    "tt": ["Медиа", "Рәсем", "Файл", "Räsem", "Изображение"],
    "ty": ["Média", "Fichier"],
    "tyv": ["Медиа", "Файл", "Изображение"],
    "udm": ["Медиа", "Файл", "Суред", "Изображение"],
    "ug": ["ۋاسىتە", "ھۆججەت"],
    "uk": ["Медіа", "Медиа", "Файл", "Изображение", "Зображення"],
    "ur": ["میڈیا", "تصویر", "وسیط", "زریعہ", "فائل", "ملف"],
    "uz": ["Mediya", "Tasvir", "Fayl"],
    "vec": ["Immagine", "Imàjine", "Mèdia"],
    "vep": ["Pilt", "Fail"],
    "vi": ["Phương_tiện", "Tập_tin", "Hình", "Tập tin", "Phương tiện"],
    "vls": ["Afbeelding", "Ofbeeldienge"],
    "vo": ["Ragiv", "Magod", "Nünamakanäd"],
    "wa": ["Imådje"],
    "war": ["Medya", "Fayl", "Paypay"],
    "wo": ["Xibaarukaay", "Dencukaay"],
    "wuu": ["文件", "档案", "图像", "媒体"],
    "xal": ["Аһар", "Боомг", "Изображение", "Зург"],  # noqa: RUF001
    "xmf": ["მედია", "სურათი", "ფაილი"],
    "yi": ["מעדיע", "תמונה", "טעקע", "בילד"],
    "yo": ["Fáìlì", "Amóhùnmáwòrán", "Àwòrán"],
    "za": ["媒体文件", "文件", "档案", "图像", "媒体"],
    "zea": ["Afbeelding", "Plaetje"],
    "zh": ["媒体文件", "F", "文件", "媒體", "档案", "图像", "圖像", "媒体", "檔案"],
    "zh-classical": ["文件", "媒體", "圖像", "檔案"],
    "zh-min-nan": ["tóng-àn", "文件", "媒體", "Mûi-thé", "圖像", "檔案"],
    "zh-yue": [
        "檔",
        "档",
        "文件",
        "图",
        "媒體",
        "圖",
        "档案",
        "图像",
        "圖像",
        "媒体",
        "檔案",
    ],
}

CAT_ALIASES = {
    "ab": ["Категория", "Акатегориа"],
    "ace": ["Kawan", "Kategori"],
    "af": ["Kategorie"],
    "ak": ["Nkyekyem"],
    "als": ["Kategorie"],
    "am": ["መደብ"],
    "an": ["Categoría"],
    "ang": ["Flocc"],
    "ar": ["تصنيف"],
    "arc": ["ܣܕܪܐ"],
    "arz": ["تصنيف"],
    "as": ["CAT", "শ্ৰেণী", "श्रेणी", "শ্রেণী"],
    "ast": ["Categoría"],
    "atj": ["Tipanictawin"],
    "av": ["Категория"],
    "ay": ["Categoría"],
    "az": ["Kateqoriya"],
    "azb": ["بؤلمه"],
    "ba": ["Төркөм", "Категория"],
    "bar": ["Kategorie"],
    "bat-smg": ["Kategorija", "Kateguorėjė"],
    "bcl": ["Kategorya"],
    "be": ["Катэгорыя"],
    "be-x-old": ["Катэгорыя"],
    "bg": ["Категория"],
    "bh": ["श्रेणी"],
    "bjn": ["Tumbung", "Kategori"],
    "bm": ["Catégorie"],
    "bn": ["বিষয়শ্রেণী", "വിഭാഗം"],
    "bpy": ["থাক"],
    "br": ["Rummad"],
    "bs": ["Kategorija"],
    "bug": ["Kategori"],
    "bxr": ["Категори", "Категория"],
    "ca": ["Categoria"],
    "cbk-zam": ["Categoría"],
    "cdo": ["分類"],
    "ce": ["Категори", "Тоба", "Кадегар"],  # noqa: RUF001
    "ceb": ["Kategoriya"],
    "ch": ["Katigoria"],
    "ckb": ["پ", "پۆل"],
    "co": ["Categoria"],
    "crh": ["Категория", "Kategoriya"],
    "cs": ["Kategorie"],
    "csb": ["Kategòrëjô"],
    "cu": ["Катигорї", "Категория", "Катигорїꙗ"],
    "cv": ["Категори"],
    "cy": ["Categori"],
    "da": ["Kategori"],
    "de": ["Kategorie"],
    "din": ["Bekätakthook"],
    "diq": ["Kategoriye", "Kategori"],
    "dsb": ["Kategorija"],
    "dty": ["श्रेणी"],
    "dv": ["ޤިސްމު"],
    "el": ["Κατηγορία"],
    "eml": ["Categoria"],
    "eo": ["Kategorio"],
    "es": ["CAT", "Categoría"],
    "et": ["Kategooria"],
    "eu": ["Kategoria"],
    "ext": ["Categoría", "Categoria"],
    "fa": ["رده"],
    "ff": ["Catégorie"],
    "fi": ["Luokka"],
    "fiu-vro": ["Katõgooria"],
    "fo": ["Bólkur"],
    "fr": ["Catégorie"],
    "frp": ["Catègorie"],
    "frr": ["Kategorie"],
    "fur": ["Categorie"],
    "fy": ["Kategory"],
    "ga": ["Rang", "Catagóir"],
    "gag": ["Kategori", "Kategoriya"],
    "gan": ["分類", "分类"],
    "gd": ["Roinn-seòrsa"],
    "gl": ["Categoría"],
    "glk": ["جرگه", "رده"],
    "gn": ["Ñemohenda"],
    "gom": ["वर्ग", "श्रेणी"],
    "gor": ["Dalala"],
    "got": ["𐌷𐌰𐌽𐍃𐌰"],
    "gu": ["શ્રેણી", "CAT", "શ્રે"],
    "gv": ["Ronney"],
    "hak": ["分類"],
    "haw": ["Māhele"],
    "he": ["קטגוריה", "קט"],
    "hi": ["श्र", "श्रेणी"],
    "hif": ["vibhag"],
    "hr": ["CT", "KT", "Kategorija"],
    "hsb": ["Kategorija"],
    "ht": ["Kategori"],
    "hu": ["Kategória"],
    "hy": ["Կատեգորիա"],
    "ia": ["Categoria"],
    "id": ["Kategori"],
    "ie": ["Categorie"],
    "ig": ["Ébéonọr", "Òtù"],
    "ii": ["分类"],
    "ilo": ["Kategoria"],
    "inh": ["ОагӀат"],
    "io": ["Kategorio"],
    "is": ["Flokkur"],
    "it": ["CAT", "Categoria"],
    "ja": ["カテゴリ"],
    "jbo": ["klesi"],
    "jv": ["Kategori"],
    "ka": ["კატეგორია"],
    "kaa": ["Sanat", "Kategoriya", "Санат", "سانات"],
    "kab": ["Taggayt"],
    "kbd": ["Категория", "Категориэ"],
    "kbp": ["Catégorie"],
    "kg": ["Kalasi"],
    "kk": ["Sanat", "Санат", "سانات"],
    "kl": ["Sumut_atassuseq", "Kategori", "Sumut atassuseq"],
    "km": ["ចំនាត់ថ្នាក់ក្រុម", "ចំណាត់ក្រុម", "ចំណាត់ថ្នាក់ក្រុម"],
    "kn": ["ವರ್ಗ"],
    "ko": ["분류"],
    "koi": ["Категория"],
    "krc": ["Категория"],
    "ks": ["زٲژ"],
    "ksh": [
        "Saachjropp",
        "Saachjrop",
        "Katejori",
        "Kategorie",
        "Saachjrupp",
        "Kattejori",
        "Sachjrop",
    ],
    "ku": ["Kategorî", "پۆل"],
    "kv": ["Категория"],
    "kw": ["Class", "Klass"],
    "ky": ["Категория"],
    "la": ["Categoria"],
    "lad": ["Kateggoría", "Katēggoría", "Categoría"],
    "lb": ["Kategorie"],
    "lbe": ["Категория"],
    "lez": ["Категория"],
    "lfn": ["Categoria"],
    "li": ["Categorie", "Kategorie"],
    "lij": ["Categorîa", "Categoria"],
    "lmo": ["Categuria", "Categoria"],
    "ln": ["Catégorie"],
    "lo": ["ໝວດ"],
    "lrc": ["دأسە"],
    "lt": ["Kategorija"],
    "ltg": ["Kategoreja"],
    "lv": ["Kategorija"],
    "mai": ["CA", "श्रेणी"],
    "map-bms": ["Kategori"],
    "mdf": ["Категорие", "Категория"],
    "mg": ["Sokajy", "Catégorie"],
    "mhr": ["Категория", "Категорий"],
    "min": ["Kategori"],
    "mk": ["Категорија"],
    "ml": ["വിഭാഗം", "വി", "വർഗ്ഗം", "വ"],
    "mn": ["Ангилал"],
    "mr": ["वर्ग"],
    "mrj": ["Категори", "Категория"],
    "ms": ["Kategori"],
    "mt": ["Kategorija"],
    "mwl": ["Catadorie", "Categoria"],
    "my": ["ကဏ္ဍ"],
    "myv": ["Категория"],
    "mzn": ["رج", "رده"],
    "nah": ["Neneuhcāyōtl", "Categoría"],
    "nap": ["Categurìa", "Categoria"],
    "nds": ["Kategorie"],
    "nds-nl": ["Categorie", "Kattegerie", "Kategorie"],
    "ne": ["श्रेणी"],
    "new": ["पुचः"],
    "nl": ["Categorie"],
    "nn": ["Kategori"],
    "no": ["Kategori"],
    "nrm": ["Catégorie"],
    "nso": ["Setensele"],
    "nv": ["Tʼááłáhági_átʼéego", "Tʼááłáhági átʼéego"],  # noqa: RUF001
    "oc": ["Categoria"],
    "olo": ["Kategourii"],
    "or": ["ବିଭାଗ", "ଶ୍ରେଣୀ"],
    "os": ["Категори"],
    "pa": ["ਸ਼੍ਰੇਣੀ"],
    "pcd": ["Catégorie"],
    "pdc": ["Abdeeling", "Kategorie"],
    "pfl": ["Kadegorie", "Sachgrubb", "Kategorie"],
    "pi": ["विभाग"],
    "pl": ["Kategoria"],
    "pms": ["Categorìa"],
    "pnb": ["گٹھ"],
    "pnt": ["Κατηγορίαν"],
    "ps": ["وېشنيزه"],
    "pt": ["Categoria"],
    "qu": ["Katiguriya"],
    "rm": ["Categoria"],
    "rmy": ["Shopni"],
    "ro": ["Categorie"],
    "roa-rup": ["Categorie"],
    "roa-tara": ["Categoria"],
    "ru": ["Категория", "К"],  # noqa: RUF001
    "rue": ["Категория", "Катеґорія"],
    "rw": ["Ikiciro"],
    "sa": ["वर्गः"],
    "sah": ["Категория"],
    "sat": ["ᱛᱷᱚᱠ"],
    "sc": ["Categoria"],
    "scn": ["Catigurìa"],
    "sd": ["زمرو"],
    "se": ["Kategoriija"],
    "sg": ["Catégorie"],
    "sh": ["Kategorija", "Категорија"],
    "si": ["ප්‍රවර්ගය"],
    "sk": ["Kategória"],
    "sl": ["Kategorija"],
    "sq": ["Kategoria", "Kategori"],
    "sr": ["Kategorija", "Категорија"],
    "srn": ["Categorie", "Guru"],
    "stq": ["Kategorie"],
    "su": ["Kategori"],
    "sv": ["Kategori"],
    "sw": ["Jamii"],
    "szl": ["Kategoryjo", "Kategoria"],
    "ta": ["பகுப்பு"],
    "tcy": ["ವರ್ಗೊ"],
    "te": ["వర్గం"],
    "tet": ["Kategoría", "Kategoria"],
    "tg": ["Гурӯҳ"],
    "th": ["หมวดหมู่"],
    "ti": ["መደብ"],
    "tk": ["Kategoriýa"],
    "tl": ["Kategorya", "Kaurian"],
    "tpi": ["Grup"],
    "tr": ["Kategori", "KAT"],
    "tt": ["Төркем", "Törkem", "Категория"],
    "ty": ["Catégorie"],
    "tyv": ["Аңгылал", "Категория"],
    "udm": ["Категория"],
    "ug": ["تۈر"],
    "uk": ["Категория", "Категорія"],
    "ur": ["زمرہ"],
    "uz": ["Turkum", "Kategoriya"],
    "vec": ["Categoria"],
    "vep": ["Kategorii"],
    "vi": ["Thể_loại", "Thể loại"],
    "vls": ["Categorie"],
    "vo": ["Klad"],
    "wa": ["Categoreye"],
    "war": ["Kaarangay"],
    "wo": ["Wàll", "Catégorie"],
    "wuu": ["分类"],
    "xal": ["Янз", "Әәшл"],
    "xmf": ["კატეგორია"],
    "yi": ["קאטעגאריע", "קאַטעגאָריע"],
    "yo": ["Ẹ̀ka"],
    "za": ["分类"],
    "zea": ["Categorie"],
    "zh": ["分类", "分類", "CAT"],
    "zh-classical": ["分類", "CAT"],
    "zh-min-nan": ["分類", "Lūi-pia̍t"],
    "zh-yue": ["分类", "分類", "类", "類"],
}


class WikipediaExtractor(DocumentExtractor):
    """Extractor for Wikipedia articles from MediaWiki XML dumps."""

    def __init__(self, language: str = "en"):
        """
        Initialize the Wikipedia extractor.

        Args:
            language: Language code for the Wikipedia articles
        """
        super().__init__()
        self._language = language

    def _create_filters(self) -> tuple[re.Pattern[str], re.Pattern[str], re.Pattern[str]]:
        """Create regex patterns for filtering Wikipedia content."""
        # Filters for magic words that are parser instructions -- e.g., __NOTOC__
        re_rm_magic = re.compile("__[A-Z]*__", flags=re.UNICODE)

        # Filters for file/image links
        media_prefixes = "|".join([*["File", "Image", "Media"], *MEDIA_ALIASES.get(self._language, [])])
        re_rm_wikilink = re.compile(f"^(?:{media_prefixes}):", flags=re.IGNORECASE | re.UNICODE)

        # Leave category links in-place but remove the category prefixes
        cat_prefixes = "|".join(["Category", *CAT_ALIASES.get(self._language, [])])
        re_clean_wikilink = re.compile(f"^(?:{cat_prefixes}):", flags=re.IGNORECASE | re.UNICODE)

        return re_rm_magic, re_rm_wikilink, re_clean_wikilink

    def _create_filter_functions(self, re_rm_wikilink: re.Pattern[str], re_clean_wikilink: re.Pattern[str]) -> tuple:
        """Create filter functions for Wikipedia content processing."""

        def rm_wikilink(obj: Any) -> bool:  # noqa: ANN401
            return bool(re_rm_wikilink.match(str(obj.title)))

        def rm_tag(obj: Any) -> bool:  # noqa: ANN401
            return str(obj.tag) in {"ref", "table"}

        def is_category(obj: Any) -> bool:  # noqa: ANN401
            return bool(re_clean_wikilink.match(str(obj.title)))

        def clean_wikilink(obj: Any) -> None:  # noqa: ANN401
            text = obj.__strip__()
            text = re.sub(re_clean_wikilink, "", text)
            obj.text = text

        def try_replace_obj(obj: Any) -> None:  # noqa: ANN401
            # For unknown reasons, objects are sometimes not found
            with contextlib.suppress(ValueError):
                clean_wikilink(obj)

        def try_remove_obj(obj: Any, section: Any) -> None:  # noqa: ANN401
            # For unknown reasons, objects are sometimes not found
            with contextlib.suppress(ValueError):
                section.remove(obj)

        return rm_wikilink, rm_tag, is_category, try_replace_obj, try_remove_obj

    def _process_sections(  # noqa: PLR0913
        self,
        wikicode: Any,  # noqa: ANN401
        re_rm_magic: re.Pattern[str],
        rm_wikilink: Any,  # noqa: ANN401
        rm_tag: Any,  # noqa: ANN401
        is_category: Any,  # noqa: ANN401
        try_replace_obj: Any,  # noqa: ANN401
        try_remove_obj: Any,  # noqa: ANN401
    ) -> str:
        """Process sections of the Wikipedia article."""
        section_text = []
        wiki_code_kwargs = {
            "flat": True,
            "include_lead": True,
            "include_headings": True,
        }

        for section in wikicode.get_sections(**wiki_code_kwargs):
            for obj in section.ifilter_wikilinks(recursive=True):
                if rm_wikilink(obj):
                    try_remove_obj(obj, section)
                elif is_category(obj):
                    try_replace_obj(obj)

            for obj_tag in section.ifilter_tags(matches=rm_tag, recursive=True):
                try_remove_obj(obj_tag, section)

            section_text.append(
                re.sub(
                    re_rm_magic,
                    "",
                    section.strip_code().strip(),
                )
            )

        return "\n\n".join(section_text)

    def extract(self, record: dict[str, Any]) -> dict[str, Any] | None:
        """Extract and clean Wikipedia article content.

        Args:
            record: Record containing raw_content field with Wikipedia markup

        Returns:
            Dict with cleaned text and metadata, or None if extraction fails
        """
        raw_content = record.get("raw_content")
        if not raw_content:
            return None

        try:
            # Import mwparserfromhell here to avoid pickling issues
            import mwparserfromhell

            # Parse the Wikipedia markup
            wikicode = mwparserfromhell.parse(raw_content)

            # Create regex patterns for filtering
            re_rm_magic, re_rm_wikilink, re_clean_wikilink = self._create_filters()

            # Create filter functions
            rm_wikilink, rm_tag, is_category, try_replace_obj, try_remove_obj = self._create_filter_functions(
                re_rm_wikilink, re_clean_wikilink
            )

            # Process sections
            text = self._process_sections(
                wikicode, re_rm_magic, rm_wikilink, rm_tag, is_category, try_replace_obj, try_remove_obj
            )

            # Only return if we have meaningful content

            return {
                "text": text,
                "title": record["title"],
                "id": record["id"],
                "url": record["url"],
                "language": record["language"],
                "source_id": record["source_id"],
            }

        except Exception as e:  # noqa: BLE001
            logger.error(f"Error extracting Wikipedia content for article {record.get('title', 'unknown')}: {e}")
            return None

    def input_columns(self) -> list[str]:
        """Define the input columns expected by this extractor."""
        return ["title", "id", "url", "language", "source_id", "raw_content"]

    def output_columns(self) -> list[str]:
        """Define the output columns produced by this extractor."""
        return ["text", "title", "id", "url", "language", "source_id"]
