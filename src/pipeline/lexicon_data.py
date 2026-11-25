"""
Leksikon Sentimen Bahasa Indonesia & Bahasa Daerah Bali (InSet Derivative, Slang, & Academic)
Menyediakan kamus kata positif, negatif, negasi, booster, konjungsi kontras, dan emoji
yang dioptimalkan untuk percakapan media sosial seputar dunia kampus & Undiksha.
"""

from typing import Dict, Set

# Kata Negasi: Membalik polaritas kata sentimen di depannya
NEGATION_WORDS: Set[str] = {
    "tidak", "bukan", "ga", "gak", "nggak", "ndak", "tak", "tanpa",
    "jangan", "belum", "kurang", "tiada", "enggak", "kagak", "ngga",
    "bkn", "tdk", "gk", "g", "ora", "tar", "dudu", "sing", "ten"
}

# Kata Booster / Intensifier: Meningkatkan bobot sentimen
BOOSTER_WORDS: Dict[str, float] = {
    "sangat": 1.6,
    "banget": 1.6,
    "bgt": 1.6,
    "bgtu": 1.3,
    "sekali": 1.4,
    "amat": 1.4,
    "terlalu": 1.3,
    "super": 1.5,
    "benar-benar": 1.6,
    "bener-bener": 1.6,
    "beneran": 1.4,
    "sungguh": 1.5,
    "paling": 1.5,
    "luar biasa": 1.8,
    "ekstra": 1.3,
    "parah": 1.4,
    "total": 1.3,
    "gati": 1.5,     # Bahasa Bali: 'banget/sangat' (contoh: melah gati, jele gati)
    "sajan": 1.5,    # Bahasa Bali: 'benar-benar'
    "pol": 1.4,
    "poll": 1.4,
    "maksimal": 1.4
}

# Konjungsi Kontrastif (Membalik fokus kalimat)
# Contoh: "Kampusnya bagus TAPI pelayanannya lelet" -> Klausa setelah "tapi" mendapat bobot lebih tinggi
CONTRASTIVE_CONJUNCTIONS: Set[str] = {
    "tapi", "tetapi", "namun", "sayangnya", "sedangkan", "padahal",
    "melainkan", "akan tetapi", "sayang"
}

# Emoji Sentimen: Bobot polaritas untuk emoji populer di media sosial
EMOJI_SENTIMENT: Dict[str, float] = {
    # Positif Kuat
    "❤️": 2.0, "💖": 2.0, "💙": 2.0, "💕": 2.0, "🤍": 1.5, "✨": 1.5,
    "😍": 2.5, "🥰": 2.5, "😊": 1.5, "😁": 1.5, "😄": 1.5, "😃": 1.5,
    "👏": 2.2, "🙌": 2.0, "🎉": 2.5, "🥳": 2.2, "🔥": 2.2, "👍": 1.8,
    "💪": 2.0, "🏆": 2.5, "💯": 2.2, "🌟": 1.8, "⭐": 1.5, "🤩": 2.2,
    "🎓": 2.0, "💐": 1.5, "🙏": 1.5, "🤝": 1.5, "🫡": 1.5, "👌": 1.5,
    "😻": 2.0, "🤗": 1.8, "☀️": 1.2, "👑": 2.0, "🥇": 2.5,

    # Negatif Kuat
    "😢": -1.8, "😭": -1.5, "😡": -2.8, "😠": -2.2, "🤬": -3.2,
    "💔": -2.2, "👎": -2.0, "🤮": -2.8, "🤢": -2.2, "🙄": -1.5,
    "😒": -1.8, "😞": -1.8, "😔": -1.8, "😩": -2.0, "😫": -2.0,
    "🤡": -1.8, "💩": -2.2, "🤦": -1.8, "🤦‍♂️": -1.8, "🤦‍♀️": -1.8,
    "😤": -2.0, "💀": -1.2, "🖕": -3.5, "🥱": -1.5
}

# Kamus Kata Positif Komprehensif (+1.0 s/d +4.0)
POSITIVE_LEXICON: Dict[str, float] = {
    # Prestasi & Kualitas Akademik
    "unggul": 3.0, "keunggulan": 3.0, "berkeunggulan": 3.0, "prestasi": 3.0, "berprestasi": 3.2,
    "juara": 3.5, "terbaik": 3.5, "hebat": 3.0, "keren": 3.0, "kerenn": 3.0, "kerennn": 3.0,
    "mantap": 2.8, "mantul": 2.8, "mantapp": 2.8, "luar biasa": 3.5, "sukses": 3.0, "kesuksesan": 3.0,
    "berhasil": 2.8, "berkualitas": 3.0, "favorit": 2.5, "terfavorit": 3.0, "kompeten": 2.8,
    "akreditasi": 2.0, "akreditasi a": 3.0, "inovatif": 2.8, "kreatif": 2.8, "cendekia": 2.5,
    "harmoni": 2.5, "kebanggaan": 3.2, "bangga": 3.0, "membanggakan": 3.5, "maju": 2.5, "kemajuan": 2.8,
    "berkembang": 2.2, "reputasi": 2.8, "populer": 2.0, "teladan": 2.8, "profesional": 2.8,
    "lulusan": 2.0, "kompetitif": 2.5, "integritas": 3.0, "dedikasi": 2.8, "potensial": 2.5,

    # Apresiasi, Syukur & Ucapan
    "selamat": 2.8, "selamatt": 2.8, "congrats": 2.8, "congratulation": 3.0, "congratulations": 3.0,
    "terimakasih": 2.2, "terima kasih": 2.2, "makasih": 2.0, "makasi": 2.0, "thanks": 2.0,
    "syukur": 2.5, "alhamdulillah": 3.0, "astungkara": 3.0, "puji": 2.2, "puji tuhan": 3.0,
    "berkah": 2.8, "salut": 2.8, "apresiasi": 2.8, "semangat": 2.5, "jaya": 3.0, "berjaya": 3.0,
    "top": 2.5, "jos": 2.5, "gokil": 2.2, "kece": 2.5, "gas": 1.5, "gass": 1.8, "gaspol": 2.0,
    "ciamik": 2.5, "paten": 2.5, "keras": 1.5, "respect": 2.5, "berfaedah": 2.5, "berbobot": 2.5,

    # Suasana Kampus, Fasilitas, & Pelayanan
    "nyaman": 2.5, "kenyamanan": 2.5, "bersih": 2.2, "kebersihan": 2.2, "indah": 2.5, "keindahan": 2.5,
    "asri": 2.5, "sejuk": 2.0, "ramah": 2.8, "keramahan": 2.8, "cepat": 2.0, "kecepatan": 2.0,
    "mudah": 2.2, "kemudahan": 2.2, "terbantu": 2.8, "membantu": 2.2, "lengkap": 2.2, "kelengkapan": 2.2,
    "teratur": 2.2, "tertib": 2.2, "aman": 2.2, "keamanan": 2.2, "puas": 3.0, "kepuasan": 3.0,
    "senang": 2.5, "bahagia": 3.0, "gembira": 2.5, "suka": 2.2, "cinta": 3.0, "sayang": 2.5,
    "idaman": 2.8, "menarik": 2.2, "bagus": 2.8, "baguss": 2.8, "baik": 2.2, "elok": 2.2,
    "megah": 2.5, "modern": 2.2, "canggih": 2.8, "optimal": 2.5, "terjangkau": 2.2, "murah": 1.8,
    "kondusif": 2.5, "rapi": 2.2, "estetik": 2.5, "aesthetic": 2.5, "adem": 2.0, "teduh": 2.0,

    # Akademik, Kelulusan & Beasiswa
    "lulus": 2.8, "kelulusan": 2.8, "wisuda": 2.8, "wisudawan": 2.5, "wisudawati": 2.5,
    "cumlaude": 3.8, "ipk": 2.0, "beasiswa": 2.8, "gratis": 2.2, "adil": 2.5, "keadilan": 2.5,
    "transparan": 2.8, "keterbukaan": 2.5, "inspiratif": 3.0, "menginspirasi": 3.0, "edukatif": 2.5,
    "bermanfaat": 2.8, "faedah": 2.2, "berguna": 2.2, "rekomendasi": 2.8, "recomended": 2.8,
    "rekomen": 2.5, "solid": 2.5, "kompak": 2.5, "harmonis": 2.8, "meriah": 2.5, "antusias": 2.8,
    "seru": 2.5, "seruu": 2.5, "asik": 2.2, "asyik": 2.2, "enjoy": 2.2, "paham": 2.0,

    # Bahasa Bali / Khas Singaraja (Positif)
    "luung": 2.8, "luung gati": 3.5, "melah": 2.8, "melah gati": 3.5, "suksma": 2.8,
    "matur suksma": 3.0, "rahajeng": 2.8, "becik": 2.8, "becik pisan": 3.5, "patut": 2.2,
    "alus": 2.5, "rame": 1.8, "ramai": 1.8, "liang": 2.5, "demen": 2.2, "trepti": 2.5,
    "shanti": 2.5, "dharma": 2.2, "asri pisan": 3.0
}

# Kamus Kata Negatif Komprehensif (-1.0 s/d -4.0)
NEGATIVE_LEXICON: Dict[str, float] = {
    # Kekecewaan, Keluhan & Kerusakan
    "kecewa": -3.0, "kekecewaan": -3.0, "kecewa berat": -3.8, "buruk": -2.8, "jelek": -2.8,
    "jelekk": -2.8, "parah": -2.8, "parahh": -3.0, "bobrok": -3.8, "rusak": -2.8,
    "kerusakan": -2.8, "hancur": -3.5, "rugi": -2.8, "merugikan": -3.2, "ampas": -3.0,
    "payah": -2.8, "lemah": -2.2, "kacau": -3.2, "kekacauan": -3.2, "amburadul": -3.5,
    "gagal": -2.8, "kegagalan": -2.8, "zonk": -2.8, "menyesal": -3.2, "nyesel": -3.2,
    "sesal": -2.5, "kapok": -3.2, "ambyar": -2.8, "terpuruk": -3.0, "malang": -2.2,
    "bencana": -3.0, "musibah": -2.5, "tragis": -3.0, "miris": -2.8, "prihatin": -2.2,

    # Masalah Birokrasi, Sistem, & Pelayanan
    "lambat": -2.5, "lemot": -3.0, "lemott": -3.0, "lelet": -3.0, "sulit": -2.2, "susah": -2.2,
    "kesulitan": -2.5, "ribet": -2.8, "ruwet": -2.8, "berbelit": -2.8, "berbelit-belit": -3.0,
    "antri": -1.8, "antre": -1.8, "antrean panjang": -2.5, "lama": -1.8, "lama bgt": -2.5,
    "batal": -2.2, "pembatalan": -2.2, "pending": -1.8, "error": -2.8, "eror": -2.8,
    "down": -2.8, "crash": -2.8, "bug": -2.2, "buggy": -2.5, "server down": -3.2,
    "jutek": -3.0, "kasar": -3.2, "cuek": -2.2, "sombong": -2.8, "curang": -3.8,
    "kecurangan": -3.8, "korupsi": -4.0, "pungli": -3.8, "mahal": -2.5, "kemahalan": -3.0,
    "naik": -1.2, "mahalan": -2.5, "beban": -2.5, "memberatkan": -3.0, "zalim": -3.8,
    "tidak adil": -3.2, "gak adil": -3.2, "diskriminasi": -3.5, "dipersulit": -3.2,
    "dosen killer": -2.8, "pungutan": -2.0, "potongan": -1.8, "birokrasi": -1.5,

    # Fasilitas, Lingkungan & Sarana Kampus
    "kotor": -2.8, "kekotoran": -2.8, "jorok": -2.8, "panas": -1.8, "gerah": -1.8,
    "pengap": -2.0, "bising": -2.0, "rusak-rusak": -2.8, "sempit": -1.8, "kumuh": -3.0,
    "gelap": -1.8, "bahaya": -2.8, "berbahaya": -3.0, "bocor": -2.5, "kebocoran": -2.5,
    "banjir": -2.8, "bau": -2.5, "kumal": -2.2, "mangkrak": -3.2, "hilang": -2.2,
    "kehilangan": -2.5, "maling": -3.5, "parkir semrawut": -2.5, "semrawut": -2.8,

    # Emosi Negatif, Stres, & Slang Sosial Media
    "marah": -2.8, "kemarahan": -2.8, "emosi": -2.5, "kesal": -2.8, "kesel": -2.8,
    "jengkel": -2.8, "gedeg": -2.8, "gregetan": -2.2, "eneg": -2.5, "benci": -3.5,
    "kebencian": -3.5, "bosan": -2.0, "bosen": -2.0, "jenuh": -2.2, "males": -2.2,
    "malas": -2.2, "mager": -1.8, "capek": -2.2, "capekk": -2.5, "cape": -2.2,
    "lelah": -2.0, "mumet": -2.5, "pusing": -2.2, "stres": -2.8, "stress": -2.8,
    "depresi": -3.2, "sedih": -2.5, "kesedihan": -2.5, "tangis": -2.2, "menangis": -2.2,
    "bodoh": -3.2, "tolol": -3.8, "goblok": -3.8, "bego": -3.2, "gaje": -2.0,
    "gajelas": -2.5, "ga jelas": -2.5, "bohong": -3.2, "kebohongan": -3.2, "hoax": -3.2,
    "tipu": -3.5, "penipuan": -3.8, "scam": -3.5, "php": -2.8, "dusta": -3.0,
    "omong kosong": -3.0, "ga mutu": -2.8, "gak mutu": -2.8, "palsu": -3.0,

    # Bahasa Bali / Khas Singaraja (Negatif)
    "sing luung": -3.0, "sing melah": -3.0, "sing becik": -3.0, "jele": -2.8,
    "jele gati": -3.5, "belog": -3.2, "belog polos": -3.5, "lengeh": -3.0,
    "buduh": -3.2, "bengkung": -2.5, "rusuh": -2.8, "pocol": -2.8, "tusing ada": -1.5,
    "nyebelin": -2.5, "sebet": -2.5, "gedeg gati": -3.2
}
