import os, random, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# ===================== 사용자 설정 =====================
DATA_DIR = "/Users/loonatrium/Documents/icpadevelopment/sehyun/catvsdog/PetImages"
IMG_SIZE = (192, 192)
BATCH_SIZE = 32
# 학습 스케줄
EPOCHS = 80
WARMUP_EPOCHS = 5
FINETUNE_EPOCHS = 75
# 데이터 분할
N_SPLITS = 3
TEST_RATIO = 0.2
VAL_RATIO = 0.1
SEED = 42
RESULTS_CSV = "results_cnn.csv"
RESULTS_PLOT_PATH = "results_plot.png"
VALID_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

# 선택 옵션
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0
UNFREEZE_AT = -40                   # 마지막 40개 레이어만 미세조정
BASE_MODEL_NAME = "EfficientNetB0"  # 또는 "MobileNetV2"
INIT_LR = 3e-4
FT_LR = 1e-4
# ======================================================

# ==== 전역 증강 레이어 ====
RAND_ROT = keras.layers.RandomRotation(
    factor=0.05,        # ≈ ±10도
    fill_mode="reflect"
)

# Pretendard 글꼴 경로 후보
FONT_PATH_CANDIDATES = [
    "/Users/loonatrium/Library/Fonts/Pretendard-Medium.otf",
    "/Users/loonatrium/Documents/developmentprac/Pretendard-Regular.ttf",
]

def set_korean_font():
    from matplotlib import font_manager
    chosen = None
    for p in FONT_PATH_CANDIDATES:
        if os.path.exists(p):
            try:
                font_manager.fontManager.addfont(p)
                font_name = font_manager.FontProperties(fname=p).get_name()
                plt.rcParams["font.family"] = font_name
                chosen = p
                break
            except Exception:
                pass
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 12
    print(f"[폰트] {'Pretendard 적용' if chosen else '기본 폰트 사용'}")

def setup_gpu_and_precision():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        print("[MixedPrecision] mixed_float16")
    except Exception:
        pass
    gpus = tf.config.list_physical_devices("GPU")
    print(f"[GPU] {'사용 가능: ' + str(gpus) if gpus else '감지되지 않음'}")

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def list_images(root):
    root = pathlib.Path(root)
    items, skipped = [], 0
    for label_name, y in [("Cat", 0), ("Dog", 1)]:
        dir_path = root / label_name
        if not dir_path.exists():
            print(f"[경고] 폴더 없음: {dir_path}")
            continue
        for p in dir_path.iterdir():
            if p.is_file() and p.suffix in VALID_EXTS:
                try:
                    # 손상 파일 방지: verify 후 convert/load까지 확인
                    with Image.open(p) as im:
                        im.verify()
                    with Image.open(p) as im:
                        im.convert("RGB").load()
                    items.append((str(p), y))
                except (UnidentifiedImageError, OSError, ValueError):
                    skipped += 1
            else:
                skipped += 1
    print(f"[스캔] 사용 가능 {len(items)}장, 제외 {skipped}장")
    return items

def stratified_split(items, test_ratio=0.2, seed=42):
    idx_cat = [i for i, it in enumerate(items) if it[1] == 0]
    idx_dog = [i for i, it in enumerate(items) if it[1] == 1]
    rnd = random.Random(seed)
    rnd.shuffle(idx_cat); rnd.shuffle(idx_dog)
    def split_one(idxs):
        n = len(idxs); n_test = max(1, int(n * test_ratio)) if n > 0 else 0
        return idxs[n_test:], idxs[:n_test]
    train_cat, test_cat = split_one(idx_cat)
    train_dog, test_dog = split_one(idx_dog)
    train_idx = train_cat + train_dog
    test_idx  = test_cat + test_dog
    rnd.shuffle(train_idx); rnd.shuffle(test_idx)
    return train_idx, test_idx

def make_cumulative(train_idx, n_splits=3):
    if len(train_idx) == 0:
        return [np.array([], dtype=int)], [np.array([], dtype=int)]
    chunks = np.array_split(train_idx, n_splits)
    cumul = [np.concatenate(chunks[:i+1]).astype(int) for i in range(n_splits)]
    return chunks, cumul

# -------- PIL 기반 로더 (TF decode 오류 회피) --------
def _load_image_numpy(path_bytes):
    path = path_bytes.decode("utf-8")
    with Image.open(path) as im:
        im = im.convert("RGB")
        return np.array(im, dtype=np.uint8)

def preprocess_img(path, label):
    img = tf.numpy_function(_load_image_numpy, [path], tf.uint8)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, 0.85, 1.15)
    img = tf.image.random_saturation(img, 0.85, 1.15)
    img = RAND_ROT(img, training=True)
    # 가벼운 줌(센터 크롭) 후 리사이즈
    crop_scale = tf.random.uniform([], 0.9, 1.0)
    h = tf.shape(img)[0]; w = tf.shape(img)[1]
    nh = tf.cast(tf.cast(h, tf.float32) * crop_scale, tf.int32)
    nw = tf.cast(tf.cast(w, tf.float32) * crop_scale, tf.int32)
    img = tf.image.resize_with_crop_or_pad(img, tf.maximum(nh, 1), tf.maximum(nw, 1))
    img = tf.image.resize(img, IMG_SIZE)
    return img, label

def make_dataset(items, indices, training=True, batch_size=BATCH_SIZE):
    if indices is None or len(indices) == 0:
        raise ValueError("선택된 인덱스가 0개입니다. 데이터 분할을 확인하세요.")
    paths = [items[i][0] for i in indices]
    labels = [items[i][1] for i in indices]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, y: (tf.convert_to_tensor(p), tf.cast(y, tf.int32)))
    ds = ds.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(buffer_size=len(indices), seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.ignore_errors()
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def dataset_size(ds):
    card = tf.data.experimental.cardinality(ds).numpy()
    return None if card < 0 else int(card)

# ---------- 클래스 불균형 보정 ----------
def compute_class_weights(indices, items):
    labels = np.array([items[i][1] for i in indices])
    counts = np.bincount(labels, minlength=2)
    total = counts.sum()
    weights = total / (2.0 * np.maximum(counts, 1))
    class_weight = {i: float(w) for i, w in enumerate(weights)}
    print(f"[class_weight] counts={counts.tolist()}  weights={class_weight}")
    return class_weight

# ---------- Focal Loss ----------
class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma=2.0, class_weight=None, name="focal_loss"):
        super().__init__(name=name)
        self.gamma = gamma
        self.class_weight = class_weight or {0:1.0, 1:1.0}
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_oh = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1. - 1e-7)
        ce = -tf.reduce_sum(y_true_oh * tf.math.log(y_pred), axis=-1)
        pt = tf.reduce_sum(y_true_oh * y_pred, axis=-1)
        focal = (1. - pt) ** self.gamma * ce
        weights = tf.gather(tf.constant([self.class_weight.get(i, 1.0) for i in range(y_pred.shape[-1])], dtype=tf.float32), y_true)
        focal = focal * weights
        return tf.reduce_mean(focal)

# ---------- 모델 ----------
def build_model(input_shape=(192,192,3), num_classes=2, base="EfficientNetB0"):
    if base == "MobileNetV2":
        base_model = keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
        preprocess = keras.applications.mobilenet_v2.preprocess_input
    else:
        base_model = keras.applications.EfficientNetB0(input_shape=input_shape, include_top=False, weights="imagenet")
        preprocess = keras.applications.efficientnet.preprocess_input
    base_model.trainable = False  # 웜업 동안 고정

    inputs = layers.Input(shape=input_shape)
    x = preprocess(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)  # mixed fp16 호환
    model = keras.Model(inputs, outputs)
    return model, base_model

def compile_model(model, lr, use_focal, class_weight=None):
    # CosineDecayRestarts 스케줄러
    steps_per_epoch = 1000
    lr_sched = keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=lr, first_decay_steps=steps_per_epoch, t_mul=2.0, m_mul=0.9, alpha=1e-2
    )
    opt = keras.optimizers.Adam(learning_rate=lr_sched)
    if use_focal:
        loss_fn = FocalLoss(gamma=FOCAL_GAMMA, class_weight=class_weight)
        metrics = ["accuracy"]
    else:
        loss_fn = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]
    model.compile(optimizer=opt, loss=loss_fn, metrics=metrics)

def split_train_val_indices(idxs, val_ratio=0.1):
    rnd = random.Random(SEED)
    idxs_copy = list(idxs)
    rnd.shuffle(idxs_copy)
    n_val = max(1, int(len(idxs_copy) * val_ratio))  # 최소 1 보장
    val_idx = idxs_copy[:n_val]
    train_idx = idxs_copy[n_val:]
    return train_idx, val_idx

# ---------- 콜백 (main 바깥에 위치) ----------
def common_callbacks(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(out_dir, "best.keras"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        ),
        # ▼ 이 줄(블록) 삭제
        # keras.callbacks.ReduceLROnPlateau(
        #     monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        # ),
    ]

def main():
    set_korean_font()
    setup_gpu_and_precision()

    # 데이터 수집 및 분할
    items = list_images(DATA_DIR)
    if len(items) == 0:
        raise RuntimeError(f"데이터가 없습니다: {DATA_DIR}")
    train_idx_all, test_idx = stratified_split(items, TEST_RATIO, SEED)
    print(f"train {len(train_idx_all)} | test {len(test_idx)}")
    if len(test_idx) == 0:
        raise RuntimeError("테스트셋이 0장입니다. TEST_RATIO 조정이 필요합니다.")

    chunks, cumul = make_cumulative(train_idx_all, N_SPLITS)
    for i, c in enumerate(chunks, 1): print(f"구간{i}: {len(c)}장")
    for i, cu in enumerate(cumul, 1): print(f"세션{i}(누적): {len(cu)}장")

    # 고정 테스트셋
    ds_test = make_dataset(items, test_idx, training=False, batch_size=BATCH_SIZE)
    if dataset_size(ds_test) == 0:
        raise RuntimeError("예측용 테스트 데이터셋이 비어 있습니다.")

    sessions, accs, f1s = [], [], []
    input_shape = (*IMG_SIZE, 3)

    for s, idxs in enumerate(cumul, start=1):
        print(f"\n[세션 {s}] 누적 학습 샘플: {len(idxs)}")
        if len(idxs) == 0:
            print("[경고] 학습 샘플 0 → 건너뜀"); continue

        # 훈련/검증 분리 (인덱스 길이로 체크)
        train_idx, val_idx = split_train_val_indices(idxs, VAL_RATIO)
        if len(train_idx) == 0 or len(val_idx) == 0:
            print("[경고] 학습/검증 샘플 0 → 건너뜀"); continue

        ds_train = make_dataset(items, train_idx, training=True, batch_size=BATCH_SIZE)
        ds_val   = make_dataset(items, val_idx,   training=False, batch_size=BATCH_SIZE)

        # 클래스 가중치
        class_weight = compute_class_weights(train_idx, items)

        # 모델 구성 및 웜업
        model, base_model = build_model(input_shape=input_shape, num_classes=2, base=BASE_MODEL_NAME)
        compile_model(model, lr=INIT_LR, use_focal=USE_FOCAL_LOSS, class_weight=class_weight)

        cb = common_callbacks(out_dir=f"runs/session_{s}")
        print("[Train] Warmup (frozen base)")
        model.fit(
            ds_train, epochs=WARMUP_EPOCHS, validation_data=ds_val,
            class_weight=None if USE_FOCAL_LOSS else class_weight,  # focal은 내부 가중치 반영
            callbacks=cb, verbose=1
        )

        # 백본 미세조정: 마지막 N개 레이어만 풀기
        for i, layer in enumerate(base_model.layers):
            layer.trainable = (i >= len(base_model.layers) + UNFREEZE_AT)  # 마지막 N개
        compile_model(model, lr=FT_LR, use_focal=USE_FOCAL_LOSS, class_weight=class_weight)

        print("[Train] Finetune (unfrozen tail)")
        model.fit(
            ds_train, epochs=FINETUNE_EPOCHS, validation_data=ds_val,
            class_weight=None if USE_FOCAL_LOSS else class_weight,
            callbacks=cb, verbose=1
        )

        # 평가
        y_true = np.array([items[i][1] for i in test_idx])
        y_prob = model.predict(ds_test, verbose=0)
        if y_prob.shape[0] != y_true.shape[0]:
            n = min(y_prob.shape[0], y_true.shape[0])
            y_prob, y_true = y_prob[:n], y_true[:n]
        y_pred = np.argmax(y_prob, axis=1)

        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, average="macro")

        sessions.append(s); accs.append(acc); f1s.append(f1)
        print(f"[세션 {s}] ACC={acc:.4f}  F1={f1:.4f}")

        # 혼동행렬 저장
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Cat", "Dog"])
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"세션 {s} 혼동행렬")
        out_path = f"confusion_matrix_session{s}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[저장] {out_path}")

    # 결과 저장
    if not sessions:
        raise RuntimeError("학습/평가 결과가 생성되지 않았습니다.")
    df = pd.DataFrame({"session": sessions, "accuracy": accs, "f1": f1s})
    df.to_csv(RESULTS_CSV, index=False)
    print(f"[저장] {RESULTS_CSV}")
    print(df)

    # 결과 그래프 저장
    plt.figure()
    plt.plot(sessions, accs, marker="o", label="Accuracy")
    plt.plot(sessions, f1s, marker="o", label="F1")
    plt.xlabel("세션"); plt.ylabel("점수")
    plt.title("고양이 vs 강아지 - 누적 학습 결과 (개선 버전)")
    plt.grid(True); plt.legend()
    plt.savefig(RESULTS_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[저장] {RESULTS_PLOT_PATH}")

if __name__ == "__main__":
    main()
