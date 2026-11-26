# -*- coding: utf-8 -*-
"""Train and serve the Vietnamese feedback topic classifier."""

import argparse
import json
from functools import lru_cache
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "data" / "train.csv"
DEV_PATH = BASE_DIR / "data" / "validation.csv"
TEST_PATH = BASE_DIR / "data" / "test.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "topic_classifier.keras"
VOCAB_PATH = MODEL_DIR / "vocabulary.json"
RESULT_PATH = BASE_DIR / "result_train.txt"
MAX_LEN = 100

LABEL_MAPPING = {
    0: "Giảng viên",
    1: "Chương trình đào tạo",
    2: "Cơ sở vật chất",
    3: "Khác",
}
NUM_TOPICS = len(LABEL_MAPPING)


def create_dictionary(texts):
    from underthesea import word_tokenize

    words = set()
    for text in texts:
        words.update(token.lower() for token in word_tokenize(str(text), format="text").split())
    return sorted(words)


def encode_text(texts, vocabulary):
    from underthesea import word_tokenize
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    if isinstance(texts, str):
        texts = [texts]
    sequences = []
    for text in texts:
        tokens = word_tokenize(str(text), format="text").split()
        sequences.append([vocabulary.get(token.lower(), 1) for token in tokens])
    return pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post", value=0)


def build_model(vocab_size):
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Dense, Embedding, Flatten, Input
    from tensorflow.keras.optimizers import Adam

    inputs = Input(shape=(MAX_LEN,))
    embedded = Embedding(input_dim=vocab_size, output_dim=300)(inputs)
    outputs = Dense(NUM_TOPICS, activation="softmax")(Flatten()(embedded))
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def calculate_metrics(y_true, y_pred):
    import numpy as np

    confusion = np.zeros((NUM_TOPICS, NUM_TOPICS), dtype=np.int64)
    np.add.at(confusion, (y_true, y_pred), 1)
    true_positives = np.diag(confusion)
    precision = np.divide(
        true_positives,
        confusion.sum(axis=0),
        out=np.zeros(NUM_TOPICS, dtype=float),
        where=confusion.sum(axis=0) != 0,
    )
    recall = np.divide(
        true_positives,
        confusion.sum(axis=1),
        out=np.zeros(NUM_TOPICS, dtype=float),
        where=confusion.sum(axis=1) != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(NUM_TOPICS, dtype=float),
        where=(precision + recall) != 0,
    )
    accuracy = float(true_positives.sum() / confusion.sum())
    return accuracy, precision, recall, f1, confusion.sum(axis=1)


def write_training_results(model, train, train_encoded, validation, validation_encoded, test, test_encoded):
    import numpy as np

    result_lines = [
        "KET QUA DANH GIA MO HINH PHAN LOAI CHU DE",
        "Precision, Recall va F1-score duoc tinh theo macro average.",
        "",
    ]
    for dataset_name, dataset, encoded in (
        ("Train", train, train_encoded),
        ("Validation", validation, validation_encoded),
        ("Final Test", test, test_encoded),
    ):
        y_true = dataset["topic"].astype(int).to_numpy()
        probabilities = model.predict(encoded, verbose=0)
        y_pred = np.argmax(probabilities, axis=1)
        accuracy, precision, recall, f1, support = calculate_metrics(y_true, y_pred)

        result_lines.extend(
            [
                f"[{dataset_name}]",
                f"Accuracy: {accuracy:.4f}",
                f"Precision (macro): {precision.mean():.4f}",
                f"Recall (macro): {recall.mean():.4f}",
                f"F1-score (macro): {f1.mean():.4f}",
                "Theo tung lop:",
            ]
        )
        for class_id, class_name in LABEL_MAPPING.items():
            result_lines.append(
                f"  {class_name}: precision={precision[class_id]:.4f}, "
                f"recall={recall[class_id]:.4f}, f1={f1[class_id]:.4f}, "
                f"support={support[class_id]}"
            )
        result_lines.append("")

    RESULT_PATH.write_text("\n".join(result_lines), encoding="utf-8")


def train_model(epochs=15):
    import pandas as pd
    from tensorflow.keras.callbacks import EarlyStopping

    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(DEV_PATH)
    test = pd.read_csv(TEST_PATH)
    vocabulary = {word: index + 2 for index, word in enumerate(create_dictionary(train["sentence"]))}
    vocabulary.update({"PAD": 0, "UNK": 1})
    model = build_model(len(vocabulary))
    train_encoded = encode_text(train["sentence"], vocabulary)
    validation_encoded = encode_text(validation["sentence"], vocabulary)
    test_encoded = encode_text(test["sentence"], vocabulary)
    model.fit(
        train_encoded,
        train["topic"].astype(int).to_numpy(),
        validation_data=(
            validation_encoded,
            validation["topic"].astype(int).to_numpy(),
        ),
        batch_size=128,
        epochs=epochs,
        callbacks=[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
        verbose=1,
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    VOCAB_PATH.write_text(json.dumps(vocabulary, ensure_ascii=False), encoding="utf-8")
    _load_artifacts.cache_clear()
    write_training_results(
        model,
        train,
        train_encoded,
        validation,
        validation_encoded,
        test,
        test_encoded,
    )
    return model


@lru_cache(maxsize=1)
def _load_artifacts():
    if not MODEL_PATH.exists() or not VOCAB_PATH.exists():
        train_model()
    import tensorflow as tf

    model = tf.keras.models.load_model(MODEL_PATH)
    vocabulary = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    return model, vocabulary


def predict_topic(text):
    """Return (class id, topic name, confidence) for one feedback text."""
    if not text or not text.strip():
        raise ValueError("Nội dung phản hồi không được để trống.")
    import numpy as np

    model, vocabulary = _load_artifacts()
    probabilities = model.predict(encode_text(text, vocabulary), verbose=0)[0]
    class_id = int(np.argmax(probabilities))
    return class_id, LABEL_MAPPING[class_id], float(probabilities[class_id])


def main():
    parser = argparse.ArgumentParser(description="Vietnamese student feedback topic classifier")
    parser.add_argument("--train", action="store_true", help="Train and save the classifier")
    parser.add_argument("--interactive", action="store_true", help="Classify text entered in the terminal")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    args = parser.parse_args()

    if args.train or not args.interactive:
        train_model(epochs=args.epochs)
    if args.interactive:
        while True:
            text = input("Nhập phản hồi (hoặc 'exit' để dừng): ").strip()
            if text.lower() == "exit":
                break
            if text:
                class_id, topic_name, confidence = predict_topic(text)
                print(f"[{topic_name}] (mã {class_id}), độ tin cậy {confidence:.2%}")


if __name__ == "__main__":
    main()
'''
# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from underthesea import word_tokenize

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Map mã nhãn sang tên chủ đề tương ứng
# Map mã nhãn sang tên chủ đề tương ứng
LABEL_MAPPING = {
    0: "Giảng viên",
    1: "Chương trình đào tạo",
    2: "Cơ sở vật chất",
    3: "Khác"
}
NUM_TOPICS = len(LABEL_MAPPING)

# ==========================================
# 1. TẢI FILE DỮ LIỆU TỪ ĐƯỜNG DẪN CỐ ĐỊNH
# ==========================================
# TODO: Thay đổi đường dẫn đến các file dữ liệu của bạn ở đây
TRAIN_PATH = "data/train.csv"        # Hoặc đường dẫn tuyệt đối: "D:/data/train.csv"
DEV_PATH   = "data/validation.csv"          # Đường dẫn file validation/dev
TEST_PATH  = "data/test.csv"         # Đường dẫn file test

def read_data_file(file_path: str) -> pd.DataFrame:
    """Đọc file dữ liệu từ đường dẫn cố định."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file tại đường dẫn: {file_path}")

    ext = os.path.splitext(file_path)[-1].lower()

    if ext == '.csv':
        df = pd.read_csv(file_path)
    elif ext == '.parquet':
        df = pd.read_parquet(file_path)
    elif ext in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Định dạng {ext} không hỗ trợ! Hãy dùng file .csv, .parquet hoặc .xlsx")

    print(f"-> Đã tải thành công file: {os.path.basename(file_path)} ({len(df)} dòng)")
    return df

print("=== ĐANG TẢI DỮ LIỆU TỪ ĐƯỜNG DẪN CỐ ĐỊNH ===")
df_train = read_data_file(TRAIN_PATH)
df_dev   = read_data_file(DEV_PATH)
df_test  = read_data_file(TEST_PATH)

# Lấy dữ liệu 2 cột 'sentence' và 'topic'
X_train, y_train = df_train['sentence'], df_train['topic'].values.flatten()
X_dev, y_dev     = df_dev['sentence'], df_dev['topic'].values.flatten()
X_test, y_test   = df_test['sentence'], df_test['topic'].values.flatten()

# ==========================================
# 2. TIỀN XỬ LÝ VĂN BẢN VỚI UNDERTHESEA
# ==========================================
def create_dictionary_underthesea(texts: pd.Series) -> list:
    """Tạo tập từ vựng sử dụng underthesea."""
    vocab_set = set()
    for text in texts.astype(str):
        tokens = word_tokenize(text, format='text').split()
        vocab_set.update([t.lower() for t in tokens])
    return sorted(list(vocab_set))

def build_vocab_mappings(vocab_list: list):
    """Tạo ánh xạ từ ngữ <-> chỉ số (Index)."""
    w2i = {w: (i + 2) for i, w in enumerate(vocab_list)}
    w2i['PAD'] = 0
    w2i['UNK'] = 1
    i2w = {i: w for w, i in w2i.items()}
    return w2i, i2w

@dataclass
class EncodingConfig:
    max_len: int = 100
    padding_type: str = 'post'
    pad_value: int = 0
    truncating: str = 'post'

config = EncodingConfig()

print("\nĐang tách từ và xây dựng từ điển...")
vocab = create_dictionary_underthesea(X_train)
w2i, i2w = build_vocab_mappings(vocab)
print(f"-> Kích thước từ điển (Underthesea): {len(w2i)}")

def encode_text(texts, vocab_map: dict, config: EncodingConfig) -> np.ndarray:
    """Chuyển đổi chuỗi văn bản thành ma trận chỉ số mã hóa."""
    if isinstance(texts, (str, pd.Series)):
        if isinstance(texts, str):
            texts = [texts]

    encoded_sequences = []
    for text in texts:
        tokens = word_tokenize(str(text), format='text').split()
        sequence = [vocab_map.get(word.lower(), vocab_map.get("UNK", 1)) for word in tokens]
        encoded_sequences.append(sequence)

    return pad_sequences(
        sequences=encoded_sequences,
        maxlen=config.max_len,
        padding=config.padding_type,
        truncating=config.truncating,
        value=config.pad_value
    )

# Mã hóa dữ liệu Train, Validation, Test
X_train_enc = encode_text(X_train, w2i, config)
X_dev_enc   = encode_text(X_dev, w2i, config)
X_test_enc  = encode_text(X_test, w2i, config)

# ==========================================
# 3. XÂY DỰNG VÀ HUẤN LUYỆN MÔ HÌNH
# ==========================================
def build_simple_model(vocab_size: int, num_classes: int, max_len: int = 100, embedding_dim: int = 300, learning_rate: float = 0.001):
    inputs = Input(shape=(max_len,))
    emb = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)
    flat = Flatten()(emb)
    outputs = Dense(num_classes, activation='softmax')(flat)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

vocab_size = len(w2i) + 1
y_train_onehot = to_categorical(y_train, num_classes=NUM_TOPICS)
y_dev_onehot   = to_categorical(y_dev, num_classes=NUM_TOPICS)

model = build_simple_model(vocab_size=vocab_size, num_classes=NUM_TOPICS, max_len=config.max_len)

print("\n--- BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH ---")
history = model.fit(
    X_train_enc, y_train_onehot,
    validation_data=(X_dev_enc, y_dev_onehot),
    batch_size=128,
    epochs=15,
    verbose=1
)

# ==========================================
# 4. TƯƠNG TÁC DỰ ĐOÁN CHỦ ĐỀ TỪ BÀN PHÍM
# ==========================================
def predict_topic(text: str):
    """Hàm phân loại chủ đề cho một câu bất kỳ."""
    encoded_input = encode_text(text, w2i, config)
    prediction = model.predict(encoded_input, verbose=0)

    pred_class = np.argmax(prediction[0])
    confidence = prediction[0][pred_class]
    topic_name = LABEL_MAPPING.get(pred_class, "Không xác định")

    return pred_class, topic_name, confidence

# Vòng lặp tương tác nhập câu từ bàn phím
print("\n" + "="*50)
print(" CHƯƠNG TRÌNH PHẢN HOẠI CHỦ ĐỀ PHẢN HỒI SINH VIÊN ")
print("="*50)

while True:
    user_input = input("\nNhập câu nhận xét (hoặc gõ 'exit' để dừng): ")
    if user_input.strip().lower() == 'exit':
        print("Đã thoát chương trình.")
        break
    if not user_input.strip():
        continue

    class_id, topic_name, confidence = predict_topic(user_input)
    print(f"-> Kết quả dự đoán : [{topic_name}] (Mã nhãn: {class_id})")
    print(f"-> Độ tin cậy      : {confidence * 100:.2f}%")
'''