# -*- coding: utf-8 -*-
"""
基因微阵列疾病分类系统

核心参考文献：
1. 数据标准化与预处理 [REF1]: Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, pp. 2825-2830.
2. RFE特征选择 [REF2]: Guyon et al. (2002). Gene Selection for Cancer Classification. ML 46(1-3). doi:10.1023/A:1012487302797
3. L1正则化 [REF3]: Tibshirani (1996). Regression Shrinkage and Selection via the Lasso. JRSSB 58(1). doi:10.1111/j.2517-6161.1996.tb02080.x
4. 随机森林 [REF4]: Breiman (2001). Random Forests. ML 45(1). doi:10.1023/A:1010933404324
5. SVM算法 [REF5]: Cortes & Vapnik (1995). Support-Vector Networks. ML 20(3). doi:10.1007/BF00994018
6. Borderline-SMOTE [REF6]: Han et al. (2005). Borderline-SMOTE. ICIC. doi:10.1007/11538059_91
7. 集成学习 [REF7]: Dietterich (2000). Ensemble Methods. MCS. doi:10.1007/3-540-45014-9_1
8. 分层交叉验证 [REF8]: Kohavi (1995). A Study of Cross-Validation. IJCAI.
"""

# ----------------------
# 模块导入
# ----------------------
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, RFE, f_classif
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from collections import Counter
import joblib
import os
import warnings
import re
from tqdm import tqdm

# SHAP可解释性分析支持
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

warnings.filterwarnings('ignore')
np.random.seed(42)  # 保证可重复性

# ----------------------
# 目录配置
# ----------------------
MODEL_DIR = "bio_models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ----------------------
# 数据加载与预处理模块
# ----------------------
def load_data(verbose=True):
    """
    数据加载与预处理模块 [REF1]
    实现功能：
    1. 基因表达数据标准化
    2. 标签格式验证与清洗
    3. 自动过滤非数字列名（解决样本编号问题）
    """

    def clean_and_validate_labels(df):
        """标签清洗流程（处理特殊字符和空值）"""
        df['class'] = df['class'].astype(str).str.strip().str.upper()
        df['class'] = df['class'].apply(lambda x: re.sub(r'[^A-Za-z]', '', x))
        df = df[df['class'] != '']
        return df

    # 加载基因表达数据（基因在行，样本在列）
    train_data = pd.read_csv('pp5i_train.gr.csv', index_col=0)

    # 关键步骤：过滤非数字列名（避免样本编号泄露）
    numeric_cols = train_data.columns.map(lambda x: str(x).isdigit())
    train_data = train_data.loc[:, numeric_cols]

    # 加载并清洗标签数据
    try:
        train_labels = pd.read_csv(
            'pp5i_train_class.txt',
            header=0,
            names=['class'],
            dtype=str,
            encoding='utf-8-sig'
        )
        train_labels = clean_and_validate_labels(train_labels)
    except FileNotFoundError:
        raise FileNotFoundError("训练标签文件未找到")

    # 标签编码与持久化
    le = LabelEncoder()
    y_train = le.fit_transform(train_labels['class'])
    joblib.dump(le, f"{MODEL_DIR}/label_encoder.joblib")

    # 数据标准化 [REF1]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_data.T)
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.joblib")

    # 数据完整性验证
    assert X_train.shape[0] == len(y_train), "数据维度不匹配"

    if verbose:
        print("\n=== 数据摘要 ===")
        print(f"训练样本数: {X_train.shape[0]}")
        print(f"基因特征数: {X_train.shape[1]}")
        print("类别分布:")
        for cls, count in Counter(le.inverse_transform(y_train)).items():
            print(f"  {cls}: {count}")

    return X_train, y_train, train_data.index, le


# ----------------------
# 特征选择模块 [REF2][REF3]
# ----------------------
class FeatureSelector:
    """
    多模式特征选择器
    实现方法：
    1. 递归特征消除 (RFE) [REF2]
    2. L1正则化选择 [REF3]
    3. ANOVA F值选择（经典方法）
    """

    def __init__(self, method='rfe', k=100):
        self.method = method
        self.k = min(k, 500)  # 安全阈值
        self.selector = None

    def fit_transform(self, X, y):
        """执行特征选择"""
        if self.method == 'rfe':
            # 基于SVM的递归特征消除 [REF2]
            estimator = SVC(kernel='linear', class_weight='balanced', random_state=42)
            self.selector = RFE(estimator, n_features_to_select=self.k)
        elif self.method == 'l1':
            # L1正则化特征选择 [REF3]
            self.selector = LogisticRegression(
                penalty='l1',
                solver='liblinear',
                class_weight='balanced',
                random_state=42
            )
            self.selector.fit(X, y)
            coef = np.abs(self.selector.coef_[0])
            top_idx = np.argsort(coef)[-self.k:]
            self.selected_mask = np.zeros(X.shape[1], dtype=bool)
            self.selected_mask[top_idx] = True
        else:
            self.selector = SelectKBest(f_classif, k=self.k)

        return X[:, self.selected_mask] if self.method == 'l1' else self.selector.fit_transform(X, y)

    def get_feature_indices(self):
        """获取选择特征的索引"""
        if self.method == 'l1':
            return np.where(self.selected_mask)[0]
        elif hasattr(self.selector, 'support_'):
            return np.where(self.selector.support_)[0]
        return self.selector.get_support(indices=True)


# ----------------------
# 分类模型模块 [REF4][REF5][REF6][REF7]
# ----------------------
class DiseaseClassifier:
    """
    疾病分类器（集成学习方法）[REF7]
    核心组件：
    - Borderline-SMOTE处理不平衡数据 [REF6]
    - 随机森林分类器 [REF4]
    - 线性SVM分类器 [REF5]
    """

    def __init__(self, feature_method='rfe'):
        self.feature_selector = FeatureSelector(method=feature_method)
        self.model = self._build_pipeline()
        self.param_grid = self._build_param_grid()
        self.best_score = None

    def _build_pipeline(self):
        """构建预处理和分类流水线"""
        return ImbPipeline([
            ('smote', BorderlineSMOTE(  # 处理类别不平衡 [REF6]
                random_state=42,
                k_neighbors=2,
                m_neighbors=5
            )),
            ('clf', VotingClassifier([  # 集成学习方法 [REF7]
                ('rf', RandomForestClassifier(class_weight='balanced')),  # [REF4]
                ('svm', SVC(kernel='linear', class_weight='balanced', probability=True, random_state=42))  # [REF5]
            ], voting='soft'))
        ])

    def _build_param_grid(self):
        """定义参数搜索空间"""
        return {
            'smote__k_neighbors': [2, 3],
            'smote__m_neighbors': [5, 7],
            'clf__rf__n_estimators': [50, 100],  # 随机森林参数 [REF4]
            'clf__rf__max_depth': [5, 10],
            'clf__svm__C': [0.1, 1]  # SVM正则化参数 [REF5]
        }

    def train(self, X, y):
        """模型训练与参数优化 [REF8]"""
        cv = StratifiedKFold(n_splits=5, shuffle=True)  # 分层交叉验证 [REF8]
        grid = GridSearchCV(
            self.model,
            self.param_grid,
            cv=cv,
            scoring='balanced_accuracy',
            n_jobs=-1,
            verbose=1
        )
        grid.fit(X, y)
        self.best_model = grid.best_estimator_
        self.best_score = grid.best_score_
        return self


# ----------------------
# 可解释性分析模块 [REF4]
# ----------------------
def explain_model(model, feature_names, class_names, method):
    """基于随机森林的特征重要性分析 [REF4]"""
    try:
        rf_model = model.named_steps['clf'].named_estimators_['rf']
        importance = rf_model.feature_importances_

        report = pd.DataFrame({
            'Gene': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)

        report.to_csv(f"{MODEL_DIR}/{method}_importance.csv", index=False)
        print(f"\n=== {method.upper()}关键生物标志物 ===")
        print(report.head(10))
    except Exception as e:
        print(f"\n特征分析失败: {str(e)}")


# ----------------------
# 主流程控制模块
# ----------------------
def main():
    """主执行流程"""
    # 初始化
    X_train, y_train, gene_names, le = load_data()
    class_names = le.classes_

    methods = ['rfe', 'l1', 'ensemble']
    results = []

    # 多方法比较
    for method in tqdm(methods, desc="方法比较进度"):
        print(f"\n{'='*40}")
        print(f" 当前方法: {method.upper()} ".center(40))
        print(f"{'='*40}")

        # 特征选择
        selector = FeatureSelector(method=method, k=50)
        X_selected = selector.fit_transform(X_train, y_train)
        feature_idx = selector.get_feature_indices()
        selected_genes = gene_names[feature_idx]

        # 模型训练
        print("\n[ 模型训练 ]")
        model = DiseaseClassifier(feature_method=method).train(X_selected, y_train)
        results.append((method, model.best_score))

        # 模型持久化
        model_data = {
            'model': model.best_model,
            'features': selected_genes.tolist(),
            'score': model.best_score,
            'classes': class_names.tolist()
        }
        joblib.dump(model_data, f"{MODEL_DIR}/{method}_model.joblib")

        # 可解释性分析
        print("\n[ 生物标志物分析 ]")
        explain_model(model.best_model, selected_genes, class_names, method)

    # 性能报告
    print("\n=== 交叉验证结果 ===")
    print("{:<10} | {:<10}".format('方法', '平衡准确率'))
    print("-"*25)
    for method, score in results:
        print("{:<10} | {:.3f}".format(method.upper(), score))


if __name__ == "__main__":
    main()