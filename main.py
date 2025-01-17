import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
import matplotlib.pyplot as plt

# 设置字体为 SimHei (黑体)
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# 解决负号显示问题
matplotlib.rcParams['axes.unicode_minus'] = False

# 加载数据集
def load_data(data_dir):
    images, labels = [], []
    for label in ['benign', 'malignant', 'normal']:
        class_dir = os.path.join(data_dir, label)
        for file_name in os.listdir(class_dir):
            if 'mask' not in file_name:
                file_path = os.path.join(class_dir, file_name)
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (128, 128))  # 调整图像大小以保持一致性
                images.append(image)
                labels.append(label)
    return np.array(images), np.array(labels)

# 预处理数据集
def preprocess_data(images, labels):
    images = images / 255.0  # 对一化像素值
    images = images.reshape(images.shape[0], -1)  # 展平图像

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)  # 编码标签
    return images, labels, label_encoder

# 主程序
data_dir = "./Breast Ultrasound Images Dataset"
images, labels = load_data(data_dir)
images, labels, label_encoder = preprocess_data(images, labels)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)

# 标准化数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================= 降维前的模型 =================
# 训练 SVM 模型（降维前）
svm_model_original = SVC(kernel='linear', C=1.0, random_state=42)
svm_model_original.fit(X_train, y_train)

# 评价 SVM 模型（降维前）
y_pred_svm_original = svm_model_original.predict(X_test)
print("降维前 SVM 分类报告:\n",
      classification_report(y_test, y_pred_svm_original, target_names=label_encoder.classes_, zero_division=0))
print("降维前 SVM 混沌矩阵:\n", confusion_matrix(y_test, y_pred_svm_original))

# 训练随机森林模型（降维前）
rf_model_original = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_original.fit(X_train, y_train)

# 评价随机森林模型（降维前）
y_pred_rf_original = rf_model_original.predict(X_test)
print("\n降维前 随机森林分类报告:\n",
      classification_report(y_test, y_pred_rf_original, target_names=label_encoder.classes_, zero_division=0))
print("\n降维前 随机森林混沌矩阵:\n", confusion_matrix(y_test, y_pred_rf_original))

# ================= PCA 降维后的模型 =================
# PCA 降维
pca = PCA(n_components=50)  # 降到50维
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 训练 SVM 模型（降维后）
svm_model_pca = SVC(kernel='linear', C=1.0, random_state=42)
svm_model_pca.fit(X_train_pca, y_train)

# 评价 SVM 模型（降维后）
y_pred_svm_pca = svm_model_pca.predict(X_test_pca)
print("\n降维后 SVM 分类报告:\n",
      classification_report(y_test, y_pred_svm_pca, target_names=label_encoder.classes_, zero_division=0))
print("降维后 SVM 混沌矩阵:\n", confusion_matrix(y_test, y_pred_svm_pca))

# 训练随机森林模型（降维后）
rf_model_pca = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_pca.fit(X_train_pca, y_train)

# 评价随机森林模型（降维后）
y_pred_rf_pca = rf_model_pca.predict(X_test_pca)
print("\n降维后 随机森林分类报告:\n",
      classification_report(y_test, y_pred_rf_pca, target_names=label_encoder.classes_, zero_division=0))
print("\n降维后 随机森林混沌矩阵:\n", confusion_matrix(y_test, y_pred_rf_pca))


# 可视化部分预测结果
def plot_predictions(X, y_true, y_pred, label_encoder, title):
    plt.figure(figsize=(12, 8))
    for i in range(6):
        idx = np.random.randint(0, len(X))
        image = X[idx].reshape(128, 128)
        true_label = label_encoder.inverse_transform([y_true[idx]])[0]
        pred_label = label_encoder.inverse_transform([y_pred[idx]])[0]

        plt.subplot(2, 3, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f"真实: {true_label}\n预测: {pred_label}")
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


plot_predictions(X_test, y_test, y_pred_svm_original, label_encoder, "降维前 SVM 预测结果")
plot_predictions(X_test, y_test, y_pred_rf_original, label_encoder, "降维前 随机森林预测结果")
plot_predictions(X_test, y_test, y_pred_svm_pca, label_encoder, "降维后 SVM 预测结果")
plot_predictions(X_test, y_test, y_pred_rf_pca, label_encoder, "降维后 随机森林预测结果")
