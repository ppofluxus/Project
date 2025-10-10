# -*- coding: utf-8 -*-
"""2025.08.29 - 300(Xception) (정리 버전)"""

# ==============================================================================
# 0. 환경 설정 및 라이브러리 임포트
# ==============================================================================

import os
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import font_manager as fm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from PIL import Image
import io

# Pretendard 글꼴 설정
font_path_candidates = [
    "/Users/loonatrium/Library/Fonts/Pretendard-Medium.otf",  # macOS 사용자 폰트 경로
    "/Users/loonatrium/Documents/developmentprac/Pretendard-Regular.ttf"  # 프로젝트 디렉토리 내 Pretendard 폰트 경로
]

font_prop = fm.FontProperties(fname=font_path_candidates[0])
rcParams['font.family'] = font_prop.get_name()

# TensorFlow Metal 플러그인 활성화
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # GPU 메모리 증가 허용
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TensorFlow 로그 레벨 설정

# 데이터셋 경로
DATASET_ZIP_PATH = '/Users/loonatrium/Documents/developmentprac/yerim/Training-300(bi).zip'
EXTRACTED_DATASET_PATH = './Training'

# 압축 파일 경로 설정
ZIP_FILE_PATH = '/Users/loonatrium/Documents/developmentprac/yerim/test.zip'
EXTRACTED_PATH = '/Users/loonatrium/Documents/developmentprac/yerim/test/'

# ==============================================================================
# 1. 데이터셋 준비
# ==============================================================================
# 데이터셋 압축 해제
if not os.path.exists(EXTRACTED_DATASET_PATH):
    with zipfile.ZipFile(DATASET_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall('.')
    print("데이터셋 압축 해제 완료.")
else:
    print("데이터셋이 이미 압축 해제되어 있습니다.")

# 압축 해제된 데이터 경로 설정
TRAINING_DATASET_PATH = EXTRACTED_DATASET_PATH

# 데이터 증강 및 전처리
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    TRAINING_DATASET_PATH,
    target_size=(300, 300),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    TRAINING_DATASET_PATH,
    target_size=(300, 300),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# ==============================================================================
# 2. Xception 모델 정의
# ==============================================================================
base_model = Xception(weights='imagenet', include_top=False, input_shape=(300, 300, 3))

# 모델 구조 확장
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 사전 학습된 층 고정
for layer in base_model.layers:
    layer.trainable = False

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# ==============================================================================
# 3. 모델 학습
# ==============================================================================
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# ==============================================================================
# 4. 학습 결과 시각화
# ==============================================================================
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# ==============================================================================
# 5. 테스트 데이터 준비 및 혼동 행렬 생성
# ==============================================================================
# 압축 해제된 테스트 데이터 경로로 설정
TEST_DATASET_PATH = EXTRACTED_PATH

test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    TEST_DATASET_PATH,
    target_size=(300, 300),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# 모델 예측 수행
y_pred = model.predict(test_generator)
y_pred_classes = (y_pred > 0.5).astype(int).flatten()

# 실제 클래스 가져오기
y_true = test_generator.classes

# 혼동 행렬 생성
cm = confusion_matrix(y_true, y_pred_classes, labels=[0, 1])

# 혼동 행렬 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['no', 'yes'], yticklabels=['no', 'yes'])
plt.title('혼동 행렬')
plt.xlabel('예측 값')
plt.ylabel('실제 값')
plt.show()

# 압축 파일 내부 파일 목록 읽기
with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
    file_list = zip_ref.namelist()
    print(f"압축 파일 내 파일 목록: {file_list[:10]} (총 {len(file_list)}개)")

# 이미지 파일만 필터링
image_files = [f for f in file_list if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
print(f"이미지 파일 목록: {image_files[:10]} (총 {len(image_files)}개)")

# 이미지 로드 예제 (압축 해제 없이 메모리에서 처리)
def load_image_from_zip(zip_path, file_name):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file = zip_ref.open(file_name)
        image = Image.open(file).convert('RGB')  # 파일 핸들을 닫기 전에 이미지 로드
        return image

# 예제: 첫 번째 이미지 파일 로드
example_image = load_image_from_zip(ZIP_FILE_PATH, image_files[0])
example_image.show()

# TensorFlow 데이터 파이프라인에 통합하려면, tf.data.Dataset을 활용하여 처리 가능
# ...추가 구현 필요...
