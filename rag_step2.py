import requests
import numpy as np
import faiss
import sys
import os

# =========================
# ⚙️ КОНФИГУРАЦИЯ
# =========================
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "mistral"
CHUNK_SIZE = 500  # Символов (примерно)
OVERLAP = 50      # Символов перекрытия
DOC_PATH = "document.txt"
# Порог дистанции (для нормализованных векторов: 0=идентичны, 2=противоположны)
# Для nomic-embed-text хорошее значение около 0.8 - 1.2 для L2 на нормализованных данных
MAX_DISTANCE_THRESHOLD = 1.2 

session = requests.Session()

# =========================
# 🧩 УМНЫЙ ЧАНКИНГ
# =========================
def chunk_text_smart(text, chunk_size=300, overlap=50):
    """
    Делит текст на чанки, стараясь не разрывать слова.
    """
    if not text:
        return []
    
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    # Простой алгоритм накопления слов
    for word in words:
        word_len = len(word) + 1 # +1 для пробела
        if current_length + word_len > chunk_size and current_chunk:
            # Сохраняем текущий чанк
            full_chunk = " ".join(current_chunk)
            chunks.append(full_chunk)
            
            # Реализация overlap: оставляем последние N слов, чтобы влезли в overlap
            # Это упрощенная логика, но она лучше простого среза
            overlap_len = 0
            new_chunk = []
            for w in reversed(current_chunk):
                if overlap_len + len(w) < overlap:
                    new_chunk.insert(0, w)
                    overlap_len += len(w) + 1
                else:
                    break
            
            current_chunk = new_chunk
            current_length = overlap_len

        current_chunk.append(word)
        current_length += word_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# =========================
# 🧠 EMBEDDINGS (Batch support if possible, or fast loop)
# =========================
def get_embeddings(texts):
    embeddings = []
    print(f"⏳ Генерация эмбеддингов для {len(texts)} текстов...", end="", flush=True)
    
    for i, text in enumerate(texts):
        try:
            # Используем session для ускорения
            response = session.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={
                    "model": EMBED_MODEL,
                    "prompt": text
                },
                timeout=60
            )
            response.raise_for_status()
            emb = response.json()["embedding"]
            embeddings.append(emb)
        except Exception as e:
            print(f"\n❌ Ошибка при генерации эмбеддинга (index {i}): {e}")
            sys.exit(1)
            
    print(" Готово!")
    # Преобразуем в numpy array и нормализуем для косинусного поиска
    mat = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(mat) # <--- ВАЖНО: Нормализация
    return mat

# =========================
# 🧠 GENERATION
# =========================
def ollama_generate(prompt):
    try:
        response = session.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": GEN_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.RequestException as e:
        return f"Ошибка генерации: {e}"

# =========================
# 🚀 MAIN PIPELINE
# =========================
def main():
    # 1. Проверка файла
    if not os.path.exists(DOC_PATH):
        print(f"❌ Файл {DOC_PATH} не найден.")
        sys.exit(1)

    with open(DOC_PATH, "r", encoding="utf-8") as f:
        document_text = f.read()

    if not document_text.strip():
        print("❌ Файл пуст.")
        sys.exit(1)

    # 2. Чанкинг
    text_chunks = chunk_text_smart(document_text, CHUNK_SIZE, OVERLAP)
    
    chunks_meta = [
        {"id": i, "text": txt, "source": DOC_PATH}
        for i, txt in enumerate(text_chunks)
    ]
    print(f"✅ Создано чанков: {len(chunks_meta)}")

    # 3. Эмбеддинги базы знаний
    doc_embeddings = get_embeddings([c["text"] for c in chunks_meta])

    # 4. FAISS Index
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension) # L2 на нормализованных векторах = Cosine
    index.add(doc_embeddings)

    # 5. Интерактив
    while True:
        print("\n" + "="*40)
        question = input("Введите вопрос (или 'q' для выхода): ").strip()
        if question.lower() in ['q', 'exit', 'quit']:
            break
        if not question:
            continue

        # Эмбеддинг вопроса (тоже нормализуем!)
        q_emb = get_embeddings([question]) # Вернет уже нормализованный вектор
        
        # Поиск
        TOP_K = 3
        distances, indices = index.search(q_emb, TOP_K)
        
        best_distance = distances[0][0]
        print(f"🔎 Лучшая дистанция (L2): {best_distance:.4f}")

        # Проверка порога
        # При нормализации: Дистанция 0 = совпадение, 2 = противоположность.
        # Обычно релевантное < 1.0 (зависит от модели)
        if best_distance > MAX_DISTANCE_THRESHOLD:
            print("\n⚠️  Ответ не найден в контексте (дистанция слишком велика).")
            # Можно все равно попробовать ответить, но предупредить
            # continue 
        
        selected_chunks = [chunks_meta[i] for i in indices[0]]
        context_text = "\n---\n".join(c["text"] for c in selected_chunks)

        # Промпт
        prompt = f"""
Ты помощник, отвечающий на вопросы по документации.
Используй ТОЛЬКО следующий контекст для ответа. 
Если информации недостаточно, ответь "Я не знаю ответа на основе предоставленного текста".
Не придумывай факты.

Контекст:
{context_text}

Вопрос: 
{question}
"""
        print("🤔 Думаю...")
        answer = ollama_generate(prompt)
        
        print(f"\n🧠 Ответ:\n{answer.strip()}")
        print("\n📚 Использованные фрагменты:")
        for idx, chunk in enumerate(selected_chunks):
            print(f"[{idx+1}] ID:{chunk['id']} (Dist: {distances[0][idx]:.3f}) -> {chunk['text'][:50]}...")

if __name__ == "__main__":
    main()