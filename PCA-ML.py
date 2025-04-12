# -*- coding: utf-8 -*-
"""
Modified version compatible with Python 3.5 and older Keras/Seaborn versions
Fixed barplot data type error
Added tumor markers (CEA, CA125, CA153) as classification features
Added ROC curve generation and CSV export
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            confusion_matrix, roc_curve, auc, classification_report, 
                            silhouette_score)
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
# 独立的Keras库导入
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import matplotlib
import re
import warnings
warnings.filterwarnings('ignore')

# Windows系统字体设置
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
matplotlib.rcParams['axes.unicode_minus'] = False

# 函数：检查类型和修复列表数据
def is_list_type(obj):
    """检查对象是否为列表类型"""
    return isinstance(obj, list)

# 函数：清洗Excel数据并处理公式
def clean_excel_data(df):
    """清洗Excel数据，处理公式并尽可能转换为数值类型"""
    # 深拷贝防止修改原始数据
    df = df.copy()
    
    for col in df.columns:
        # 检查列是否包含列表
        contains_list = False
        for val in df[col]:
            if is_list_type(val):
                contains_list = True
                break
        
        # 如果包含列表，尝试提取第一个元素
        if contains_list:
            for i, val in enumerate(df[col]):
                if is_list_type(val) and len(val) > 0:
                    df.loc[i, col] = val[0]  # 提取列表的第一个元素
                elif is_list_type(val):
                    df.loc[i, col] = np.nan   # 空列表转为NaN
        
        # 如果是对象类型，检查Excel公式
        if pd.api.types.is_object_dtype(df[col]):
            formula_mask = df[col].astype(str).str.contains('=', na=False)
            if formula_mask.any():
                df.loc[formula_mask, col] = np.nan
    
    return df

# 加载数据并清洗
print("加载和清洗数据...")
try:
    df_roc = pd.read_excel("ROC曲线数据.xlsx")
    print("ROC曲线数据加载成功，形状：", df_roc.shape)
    df_roc = clean_excel_data(df_roc)

    df_tnm = pd.read_excel("BC病理分期.xlsx")
    print("BC病理分期数据加载成功，形状：", df_tnm.shape)
    df_tnm = clean_excel_data(df_tnm)
except Exception as e:
    print("加载数据出错：", str(e))
    import traceback
    traceback.print_exc()
    raise

# 根据TNM分期确定癌症阶段
def determine_stage(tnm_string):
    """根据TNM分期确定乳腺癌阶段"""
    # 如果输入不是字符串，跳过处理
    if not isinstance(tnm_string, str):
        return '未知'
    
    # 解析T、N、M组件
    t_match = re.search(r'T\d', tnm_string)
    n_match = re.search(r'N\d', tnm_string)
    m_match = re.search(r'M\d', tnm_string)
    
    if not all([t_match, n_match, m_match]):
        return '未知'
    
    t = t_match.group()
    n = n_match.group()
    m = m_match.group()
    
    # I期
    if t == 'T1' and n == 'N0' and m == 'M0':
        return 'I期'
    
    # II期
    elif ((t in ['T0', 'T1']) and n == 'N1' and m == 'M0') or \
         (t == 'T2' and n in ['N0', 'N1'] and m == 'M0') or \
         (t == 'T3' and n == 'N0' and m == 'M0'):
        return 'II期'
    
    # III-IV期
    elif ((t in ['T0', 'T1', 'T2']) and n == 'N2' and m == 'M0') or \
         (t == 'T3' and n in ['N1', 'N2'] and m == 'M0') or \
         (t == 'T4' and m == 'M0') or \
         (n == 'N3' and m == 'M0') or \
         (m == 'M1'):
        return 'III期'
    
    else:
        return '未知'

# 处理BC数据（乳腺癌患者）
print("处理乳腺癌患者数据...")
bc_data = df_tnm.copy()
bc_data['Stage'] = bc_data['病理分期'].apply(determine_stage)
bc_data['Type'] = 'BC'

# 处理HC数据（健康对照组）
print("处理健康对照组数据...")
try:
    # 确保索引是字符串类型
    df_roc.index = df_roc.index.astype(str)
    hc_data = df_roc[df_roc.index.str.contains('HC')].copy()
    print("找到 {} 个健康对照样本".format(len(hc_data)))
except Exception as e:
    print("处理健康对照数据时出错:", str(e))
    
    # 备选方案：手动检查第一列中是否有HC标记
    try:
        first_col = df_roc.iloc[:, 0]
        hc_rows = []
        for i, val in enumerate(first_col):
            if isinstance(val, str) and 'HC' in val:
                hc_rows.append(i)
        
        if hc_rows:
            hc_data = df_roc.iloc[hc_rows].copy()
            print("通过第一列找到 {} 个健康对照样本".format(len(hc_data)))
        else:
            # 搜索所有字符串列
            hc_data = None
            for col in df_roc.columns:
                if pd.api.types.is_string_dtype(df_roc[col]):
                    matches = df_roc[df_roc[col].str.contains('HC', na=False)]
                    if not matches.empty:
                        hc_data = matches
                        print("在列 {} 中找到 {} 个健康对照样本".format(col, len(hc_data)))
                        break
            
            if hc_data is None:
                print("无法找到健康对照样本，请检查数据格式")
                hc_data = pd.DataFrame(columns=df_roc.columns)
    except Exception as e2:
        print("备选方案也失败:", str(e2))
        hc_data = pd.DataFrame(columns=df_roc.columns)

# 添加缺失的列
if 'Type' not in hc_data.columns:
    hc_data['Type'] = 'HC'
if 'Stage' not in hc_data.columns:    
    hc_data['Stage'] = '健康'

# 选择用于分析的特征
# 原始循环特征
base_features = ['hsa_circ_0044235 current/μΑ', 'hsa_circ_0000250 current/μΑ']
# 肿瘤标志物特征
tumor_markers = ['CEA', 'CA125', 'CA153']
# 合并所有特征
common_features = base_features.copy()  # 循环RNA特征是基础特征

# 检查并添加肿瘤标志物
for marker in tumor_markers:
    if marker in bc_data.columns:
        # 检查健康对照组是否有此标志物
        if marker in hc_data.columns:
            common_features.append(marker)
            print("添加肿瘤标志物: {} (存在于两个数据集)".format(marker))
        else:
            # 如果健康对照组没有，则添加并填充正常值
            print("健康对照组缺少 {}，将使用参考范围正常值填充".format(marker))
            # 使用参考范围的典型正常值填充
            if marker == 'CEA':
                hc_data[marker] = 2.5  # CEA正常范围通常<5 ng/mL
            elif marker == 'CA125':
                hc_data[marker] = 15.0  # CA125正常范围通常<35 U/mL
            elif marker == 'CA153':
                hc_data[marker] = 15.0  # CA153正常范围通常<30 U/mL
            common_features.append(marker)
    else:
        print("乳腺癌数据中不存在 {}，将忽略此标志物".format(marker))

# 检查特征列是否存在
missing_features = []
for feature in common_features:
    if feature not in bc_data.columns:
        missing_features.append(feature)
    if feature not in hc_data.columns:
        missing_features.append(feature)

if missing_features:
    print("警告: 以下特征在数据中缺失: {}".format(set(missing_features)))
    # 尝试找到匹配的列名
    for dataset in [bc_data, hc_data]:
        for miss_feat in set(missing_features):
            for col in dataset.columns:
                if miss_feat.lower() in col.lower():
                    print("  可能的匹配: {} -> {}".format(miss_feat, col))

# 确保特征列存在于两个数据集中
valid_common_features = []
for feature in common_features:
    if feature in bc_data.columns and feature in hc_data.columns:
        valid_common_features.append(feature)
    else:
        print("忽略特征 {}: 不存在于两个数据集中".format(feature))

if not valid_common_features:
    raise ValueError("没有有效的共同特征可用于分析")

common_features = valid_common_features
print("使用的特征: {}".format(common_features))

# 填充缺失值
for feature in common_features:
    if feature in bc_data.columns and bc_data[feature].isnull().any():
        median_value = bc_data[feature].median()
        print("填充乳腺癌数据中 {} 的缺失值，使用中位数: {}".format(feature, median_value))
        bc_data[feature] = bc_data[feature].fillna(median_value)
    
    if feature in hc_data.columns and hc_data[feature].isnull().any():
        # 对于健康对照组，使用正常参考范围的中值
        if feature == 'CEA':
            fill_value = 2.5
        elif feature == 'CA125':
            fill_value = 15.0
        elif feature == 'CA153':
            fill_value = 15.0
        else:
            fill_value = hc_data[feature].median()
        
        print("填充健康对照数据中 {} 的缺失值，使用值: {}".format(feature, fill_value))
        hc_data[feature] = hc_data[feature].fillna(fill_value)

# 设置扩展特征为所有特征
extended_features = common_features.copy()

# 准备BC数据
bc_common = bc_data[common_features].copy()
bc_common['Type'] = 'BC'
bc_common['Stage'] = bc_data['Stage'].values

# 准备HC数据
hc_common = hc_data[common_features].copy()
hc_common['Type'] = 'HC' 
hc_common['Stage'] = '健康'

# 合并样本进行联合分析
all_samples = pd.concat([bc_common, hc_common], axis=0)

# 删除特征中有NaN值的行
all_samples = all_samples.dropna(subset=common_features)

print("分析数据: {} 个样本, {} 个特征".format(
    len(all_samples), len(common_features)))
print("  乳腺癌样本: {}".format(len(bc_common.dropna(subset=common_features))))
print("  健康对照样本: {}".format(len(hc_common.dropna(subset=common_features))))

# 检查样本数量是否足够
if len(all_samples) < 10:
    raise ValueError("样本数量太少，无法进行有效分析")
if len(bc_common.dropna(subset=common_features)) < 5:
    print("警告: 乳腺癌样本数量太少，结果可能不可靠")
if len(hc_common.dropna(subset=common_features)) < 5:
    print("警告: 健康对照样本数量太少，结果可能不可靠")

# 标准化特征
scaler = StandardScaler()
scaled_features = scaler.fit_transform(all_samples[common_features])

# 应用PCA进行降维与可视化
print("执行PCA降维...")
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_features)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Type'] = all_samples['Type'].values
pca_df['Stage'] = all_samples['Stage'].values

# 应用K-means聚类
print("执行K-means聚类...")
kmeans = KMeans(n_clusters=2, random_state=42)
pca_df['Cluster'] = kmeans.fit_predict(scaled_features)

# 保存PCA聚类结果到CSV
print("保存PCA聚类结果到CSV...")
# 将主成分及聚类结果保存为CSV
pca_results_df = pca_df.copy()
# 添加原始特征
for feature in common_features:
    pca_results_df[feature] = all_samples[feature].values
# 添加主成分解释方差
pca_results_df['PC1_explained_variance'] = pca.explained_variance_ratio_[0]
pca_results_df['PC2_explained_variance'] = pca.explained_variance_ratio_[1]
# 添加样本索引信息
pca_results_df['Sample_ID'] = all_samples.index
# 保存到CSV
pca_results_df.to_csv('pca_clustering_results.csv', index=False)

# 保存聚类中心信息
centroids = kmeans.cluster_centers_
pca_centroids = pca.transform(centroids)
centroids_df = pd.DataFrame(centroids, columns=common_features)
centroids_df['Cluster'] = range(len(centroids))
pca_centroids_df = pd.DataFrame(pca_centroids, columns=['PC1', 'PC2'])
pca_centroids_df['Cluster'] = range(len(pca_centroids))
centroids_df = pd.concat([centroids_df, pca_centroids_df[['PC1', 'PC2']]], axis=1)
centroids_df.to_csv('cluster_centers.csv', index=False)

# 使用扩展特征准备BC样本进行详细分析
if len(extended_features) > len(common_features):
    print("分析乳腺癌样本的扩展特征...")
    valid_extended = [f for f in extended_features if f in bc_data.columns]
    if len(valid_extended) > len(common_features):
        bc_full_features = bc_data[valid_extended].copy()
        bc_full_features = bc_full_features.dropna()
        
        # 标准化BC扩展特征
        bc_scaler = StandardScaler()
        bc_scaled_features = bc_scaler.fit_transform(bc_full_features)
        
        # 尝试3个聚类（对应癌症分期）
        bc_kmeans = KMeans(n_clusters=min(3, len(bc_full_features)), random_state=42)
        bc_clusters = bc_kmeans.fit_predict(bc_scaled_features)
        
        # BC样本的PCA分析
        bc_pca = PCA(n_components=2)
        bc_pca_components = bc_pca.fit_transform(bc_scaled_features)
        bc_pca_df = pd.DataFrame(data=bc_pca_components, columns=['PC1', 'PC2'])
        bc_pca_df['Stage'] = bc_data.loc[bc_full_features.index, 'Stage'].values
        bc_pca_df['Cluster'] = bc_clusters
    else:
        print("没有足够的扩展特征，将使用基本特征")
        bc_full_features = bc_common[common_features].copy()
        bc_full_features = bc_full_features.dropna()
        bc_indices = all_samples[all_samples['Type'] == 'BC'].index
        bc_scaled_features = scaled_features[[i for i, idx in enumerate(all_samples.index) if idx in bc_indices]]
        if len(bc_scaled_features) > 0:
            bc_clusters = kmeans.predict(bc_scaled_features)
            bc_pca_df = pca_df[pca_df['Type'] == 'BC'].copy()
            bc_pca_df['Cluster'] = bc_clusters
        else:
            print("警告: 没有有效的BC样本用于聚类")
            bc_clusters = []
            bc_pca_df = pd.DataFrame(columns=['PC1', 'PC2', 'Type', 'Stage', 'Cluster'])
else:
    # 如果没有扩展特征，仅使用普通特征
    bc_full_features = bc_common[common_features].copy()
    bc_full_features = bc_full_features.dropna()
    bc_indices = all_samples[all_samples['Type'] == 'BC'].index
    bc_scaled_features = scaled_features[[i for i, idx in enumerate(all_samples.index) if idx in bc_indices]]
    if len(bc_scaled_features) > 0:
        bc_clusters = kmeans.predict(bc_scaled_features)
        bc_pca_df = pca_df[pca_df['Type'] == 'BC'].copy()
        bc_pca_df['Cluster'] = bc_clusters
    else:
        print("警告: 没有有效的BC样本用于聚类")
        bc_clusters = []
        bc_pca_df = pd.DataFrame(columns=['PC1', 'PC2', 'Type', 'Stage', 'Cluster'])

# ============== 机器学习模型 ==============

# 准备分类数据（预测癌症与健康）
print("准备机器学习模型数据...")
X = scaled_features
y = (all_samples['Type'] == 'BC').astype(int)  # 1为BC, 0为HC

# 确保有足够的样本进行训练和测试
if len(all_samples) < 10:
    print("警告: 样本数量太少，无法进行有效的机器学习，将跳过模型训练")
    skip_ml = True
else:
    skip_ml = False

if not skip_ml:
    # 将数据分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # 定义集成模型的组成部分
    print("训练集成模型...")
    svm_model = SVC(probability=True, random_state=42)
    gb_model = GradientBoostingClassifier(random_state=42)
    rf_model = RandomForestClassifier(random_state=42)
    lr_model = LogisticRegression(random_state=42, max_iter=1000)

    # 创建投票分类器（集成模型）
    ensemble_model = VotingClassifier(
        estimators=[
            ('svm', svm_model),
            ('gb', gb_model),
            ('rf', rf_model),
            ('lr', lr_model)
        ],
        voting='soft'  # 使用概率估计进行投票
    )

    # 训练集成模型
    ensemble_model.fit(X_train, y_train)

    # 进行预测
    y_pred = ensemble_model.predict(X_test)
    y_pred_proba = ensemble_model.predict_proba(X_test)[:, 1]

    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # 集成模型的交叉验证
    cv = StratifiedKFold(n_splits=min(5, len(y_train) // 2), shuffle=True, random_state=42)
    cv_scores = cross_val_score(ensemble_model, X, y, cv=cv, scoring='accuracy')

    # 训练单独的模型进行比较
    print("训练单独的机器学习模型...")
    models = {
        'SVM': svm_model,
        'Gradient Boosting': gb_model,
        'Random Forest': rf_model,
        'Logistic Regression': lr_model
    }

    model_results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        model_pred = model.predict(X_test)
        model_accuracy = accuracy_score(y_test, model_pred)
        model_results[name] = {
            'accuracy': model_accuracy,
            'predictions': model_pred
        }

    # 神经网络模型
    try:
        print("训练神经网络模型...")
        def create_nn_model(input_dim):
            model = Sequential()
            model.add(Dense(min(32, input_dim*2), activation='relu', input_dim=input_dim))
            model.add(Dropout(0.2))
            model.add(Dense(min(16, input_dim), activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            return model

        # 创建临时目录保存模型
        if not os.path.exists('temp_models'):
            os.makedirs('temp_models')

        # 创建并训练神经网络（使用ModelCheckpoint代替restore_best_weights）
        nn_model = create_nn_model(X_train.shape[1])
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        model_checkpoint = ModelCheckpoint(
            'temp_models/best_model.h5', 
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )

        nn_history = nn_model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=min(8, len(X_train)),
            validation_split=0.2,
            callbacks=[early_stopping, model_checkpoint],
            verbose=0
        )

        # 加载最佳模型
        if os.path.exists('temp_models/best_model.h5'):
            nn_model = load_model('temp_models/best_model.h5')

        # 神经网络预测
        nn_y_pred_raw = nn_model.predict(X_test)
        nn_y_pred = (nn_y_pred_raw > 0.5).astype(int).reshape(-1,)
        nn_y_pred_proba = nn_y_pred_raw.reshape(-1,)

        # 为神经网络计算ROC曲线
        nn_fpr, nn_tpr, nn_thresholds = roc_curve(y_test, nn_y_pred_proba)
        nn_roc_auc = auc(nn_fpr, nn_tpr)
        
        nn_trained = True
    except Exception as e:
        print("神经网络训练失败:", str(e))
        nn_trained = False
        
    # ========== 保存ROC曲线数据到CSV ==========
    print("保存ROC曲线数据到CSV...")
    from scipy import interpolate
    n_points = 1000  # 标准化点数
    standard_fpr = np.linspace(0, 1, n_points)
    
    # 创建包含所有模型ROC数据的DataFrame
    roc_df = pd.DataFrame({'FPR': standard_fpr})
    
    # 修改部分：使用 "extrapolate" 填充插值
    ensemble_interp = interpolate.interp1d(fpr, tpr, bounds_error=False, fill_value="extrapolate")
    roc_df['Ensemble_TPR'] = ensemble_interp(standard_fpr)
    roc_df['Ensemble_AUC'] = roc_auc
    
    # 插值各个单独模型ROC曲线
    for name, model in models.items():
        model_proba = model.predict_proba(X_test)[:, 1]
        model_fpr, model_tpr, _ = roc_curve(y_test, model_proba)
        model_auc = auc(model_fpr, model_tpr)
        model_interp = interpolate.interp1d(model_fpr, model_tpr, bounds_error=False, fill_value="extrapolate")
        roc_df['{}_TPR'.format(name)] = model_interp(standard_fpr)
        roc_df['{}_AUC'.format(name)] = model_auc
    
    # 如果神经网络已训练，添加其ROC数据
    if nn_trained:
        nn_interp = interpolate.interp1d(nn_fpr, nn_tpr, bounds_error=False, fill_value="extrapolate")
        roc_df['Neural_Network_TPR'] = nn_interp(standard_fpr)
        roc_df['Neural_Network_AUC'] = nn_roc_auc
    
    roc_df.to_csv('roc_curves_data.csv', index=False)
    
    # 保存ROC曲线原始点(不插值)
    roc_raw_data = []
    for i in range(len(fpr)):
        roc_raw_data.append({
            'Model': 'Ensemble', 
            'FPR': fpr[i], 
            'TPR': tpr[i]
        })
    for name, model in models.items():
        model_proba = model.predict_proba(X_test)[:, 1]
        model_fpr, model_tpr, _ = roc_curve(y_test, model_proba)
        for i in range(len(model_fpr)):
            roc_raw_data.append({
                'Model': name,
                'FPR': model_fpr[i],
                'TPR': model_tpr[i]
            })
    if nn_trained:
        for i in range(len(nn_fpr)):
            roc_raw_data.append({
                'Model': 'Neural_Network',
                'FPR': nn_fpr[i],
                'TPR': nn_tpr[i]
            })
    pd.DataFrame(roc_raw_data).to_csv('roc_curves_raw_data.csv', index=False)
    
    # 保存模型性能摘要
    model_summary = []
    for name in model_results.keys():
        pred = model_results[name]['predictions']
        model_summary.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, pred),
            'Precision': precision_score(y_test, pred),
            'Recall': recall_score(y_test, pred),
            'F1_Score': f1_score(y_test, pred),
            'AUC': auc(roc_curve(y_test, models[name].predict_proba(X_test)[:, 1])[0], 
                        roc_curve(y_test, models[name].predict_proba(X_test)[:, 1])[1])
        })
    model_summary.append({
        'Model': 'Ensemble',
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1_Score': f1_score(y_test, y_pred),
        'AUC': roc_auc
    })
    if nn_trained:
        model_summary.append({
            'Model': 'Neural_Network',
            'Accuracy': accuracy_score(y_test, nn_y_pred),
            'Precision': precision_score(y_test, nn_y_pred),
            'Recall': recall_score(y_test, nn_y_pred),
            'F1_Score': f1_score(y_test, nn_y_pred),
            'AUC': nn_roc_auc
        })
    pd.DataFrame(model_summary).to_csv('model_performance_summary.csv', index=False)

# ============== 可视化 ==============
print("生成可视化分析图...")

# 1. PCA聚类可视化 (BC vs HC)
plt.figure(figsize=(10, 8))
colors = {'BC': 'red', 'HC': 'blue'}
markers = {'BC': 'o', 'HC': 's'}

for sample_type, group in pca_df.groupby('Type'):
    plt.scatter(group['PC1'], group['PC2'], 
                color=colors.get(sample_type, 'gray'),
                marker=markers.get(sample_type, 'o'),
                label=sample_type,
                alpha=0.7)

plt.title('PCA: 乳腺癌(BC)与健康对照(HC)样本分布', fontsize=15)
plt.xlabel('PCA1', fontsize=12)
plt.ylabel('PCA2', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
# 添加聚类中心
centroids = kmeans.cluster_centers_
pca_centroids = pca.transform(centroids)
plt.scatter(pca_centroids[:, 0], pca_centroids[:, 1], s=100, c='black', marker='X', label='聚类中心')
plt.tight_layout()
plt.savefig('pca_bc_hc_cluster.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. 按聚类结果着色的PCA可视化
plt.figure(figsize=(10, 8))
cluster_array = np.array(pca_df['Cluster'].tolist())
scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], 
                     c=cluster_array, 
                     cmap='jet',
                     alpha=0.7)
plt.title('K-means聚类 (k=2) PCA结果', fontsize=15)
plt.xlabel('PCA1', fontsize=12)
plt.ylabel('PCA2', fontsize=12)
plt.colorbar(scatter, label='聚类编号')
plt.grid(True, linestyle='--', alpha=0.6)
plt.scatter(pca_centroids[:, 0], pca_centroids[:, 1], s=100, c='black', marker='X', label='聚类中心')
plt.legend()
plt.tight_layout()
plt.savefig('pca_kmeans_cluster.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. 乳腺癌患者聚类分析 (按TNM分期着色)
if not bc_pca_df.empty:
    plt.figure(figsize=(10, 8))
    stage_colors = {'I期': 'green', 'II期': 'orange', 'III期': 'red', '未知': 'gray'}

    for stage, group in bc_pca_df.groupby('Stage'):
        color = stage_colors.get(stage, 'purple')
        plt.scatter(group['PC1'], group['PC2'], color=color, label=stage, alpha=0.7)

    plt.title('乳腺癌患者聚类分析 (按TNM分期)', fontsize=15)
    plt.xlabel('PCA1', fontsize=12)
    plt.ylabel('PCA2', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    if 'bc_kmeans' in locals():
        bc_centroids = bc_kmeans.cluster_centers_
        bc_pca_centroids = bc_pca.transform(bc_centroids)
        plt.scatter(bc_pca_centroids[:, 0], bc_pca_centroids[:, 1], s=100, c='black', marker='X', label='聚类中心')
    plt.tight_layout()
    plt.savefig('bc_stage_cluster.png', dpi=300, bbox_inches='tight')
    plt.show()

# 4. 特征相关性热图
if bc_full_features.shape[1] > 1:
    plt.figure(figsize=(10, 8))
    corr = bc_full_features.corr()
    sns.heatmap(corr, cmap='coolwarm', linewidths=0.5, annot=True, fmt=".2f")
    plt.title('特征相关性热图', fontsize=15)
    plt.tight_layout()
    plt.savefig('feature_correlation.svg', format="svg", bbox_inches='tight')
    plt.show()

# 5. 不同阶段乳腺癌的特征分布箱线图
if 'Stage' in bc_data.columns:
    # 创建一个文件夹存放所有CSV文件
    import os
    if not os.path.exists('boxplot_data'):
        os.makedirs('boxplot_data')
        
    for feature in extended_features:
        if feature in bc_data.columns:
            plt.figure(figsize=(10, 6))
            data_to_plot = bc_data[['Stage', feature]].dropna()
            if not data_to_plot.empty and len(data_to_plot['Stage'].unique()) > 1:
                boxplot_data = pd.DataFrame({'Stage': data_to_plot['Stage'], 'Value': data_to_plot[feature]})
                
                # 保存数据为CSV
                csv_filename = 'boxplot_data/boxplot_{}.csv'.format(feature.replace('/', '_'))
                boxplot_data.to_csv(csv_filename, index=False)
                print("已保存{}的数据到{}".format(feature, csv_filename))
                
                # 绘制图表
                sns.boxplot(x='Stage', y='Value', data=boxplot_data)
                plt.title('不同阶段乳腺癌的{}分布'.format(feature), fontsize=15)
                plt.tight_layout()
                plt.savefig('boxplot_{}.png'.format(feature.replace('/', '_')), dpi=300, bbox_inches='tight')
                plt.show()
            else:
                plt.close()

# 6. 肿瘤标志物在不同组别的分布比较
for marker in tumor_markers:
    if marker in common_features:
        plt.figure(figsize=(10, 6))
        plot_data = []
        for idx, row in bc_data.iterrows():
            if not pd.isna(row[marker]):
                plot_data.append({'Group': 'BC', 'Value': row[marker], 'Stage': row['Stage'] if 'Stage' in row else '未知'})
        for idx, row in hc_data.iterrows():
            if not pd.isna(row[marker]):
                plot_data.append({'Group': 'HC', 'Value': row[marker], 'Stage': '健康'})
        plot_df = pd.DataFrame(plot_data)
        if not plot_df.empty:
            ax = sns.boxplot(x='Group', y='Value', data=plot_df)
            plt.title('{0}在乳腺癌与健康对照中的分布'.format(marker), fontsize=15)
            plt.ylabel('{0} 值'.format(marker), fontsize=12)
            plt.xlabel('分组', fontsize=12)
            bc_values = plot_df[plot_df['Group'] == 'BC']['Value']
            hc_values = plot_df[plot_df['Group'] == 'HC']['Value']
            plt.figtext(0.15, 0.01, "BC: n={0}, 均值={1:.2f}, 中位数={2:.2f}".format(len(bc_values), bc_values.mean(), bc_values.median()), ha="left", fontsize=10)
            plt.figtext(0.65, 0.01, "HC: n={0}, 均值={1:.2f}, 中位数={2:.2f}".format(len(hc_values), hc_values.mean(), hc_values.median()), ha="right", fontsize=10)
            plt.tight_layout()
            plt.savefig('boxplot_{}_comparison.png'.format(marker), dpi=300, bbox_inches='tight')
            plt.show()
            
            plt.figure(figsize=(12, 6))
            ax = sns.boxplot(x='Stage', y='Value', data=plot_df)
            plt.title('{0}在不同乳腺癌分期的分布'.format(marker), fontsize=15)
            plt.ylabel('{0} 值'.format(marker), fontsize=12)
            plt.xlabel('分期', fontsize=12)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('boxplot_{}_by_stage.png'.format(marker), dpi=300, bbox_inches='tight')
            plt.show()

if not skip_ml:
    # 7. 集成模型的混淆矩阵
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('集成模型预测混淆矩阵', fontsize=15)
    plt.xlabel('预测类别', fontsize=12)
    plt.ylabel('实际类别', fontsize=12)
    plt.tight_layout()
    plt.savefig('ensemble_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    if nn_trained:
        plt.figure(figsize=(8, 6))
        nn_cm = confusion_matrix(y_test, nn_y_pred)
        sns.heatmap(nn_cm, annot=True, fmt='d', cmap='Blues')
        plt.title('神经网络预测混淆矩阵', fontsize=15)
        plt.xlabel('预测类别', fontsize=12)
        plt.ylabel('实际类别', fontsize=12)
        plt.tight_layout()
        plt.savefig('nn_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    print("\n各个单独模型的混淆矩阵:")
    confusion_matrices = []
    for name, result in model_results.items():
        pred = result['predictions']
        cm = confusion_matrix(y_test, pred)
        print("\n{} 混淆矩阵:".format(name))
        print(cm)
        cm_df = pd.DataFrame(cm, columns=['预测:健康', '预测:乳腺癌'], index=['实际:健康', '实际:乳腺癌'])
        confusion_matrices.append({'Model': name, 'Matrix': cm_df})
    
    print("\n集成模型混淆矩阵:")
    ensemble_cm = confusion_matrix(y_test, y_pred)
    print(ensemble_cm)
    ensemble_cm_df = pd.DataFrame(ensemble_cm, columns=['预测:健康', '预测:乳腺癌'], index=['实际:健康', '实际:乳腺癌'])
    confusion_matrices.append({'Model': 'Ensemble', 'Matrix': ensemble_cm_df})
    
    if nn_trained:
        print("\n神经网络混淆矩阵:")
        nn_cm = confusion_matrix(y_test, nn_y_pred)
        print(nn_cm)
        nn_cm_df = pd.DataFrame(nn_cm, columns=['预测:健康', '预测:乳腺癌'], index=['实际:健康', '实际:乳腺癌'])
        confusion_matrices.append({'Model': 'Neural_Network', 'Matrix': nn_cm_df})
    
    with open('confusion_matrices.csv', 'w', encoding='utf-8') as f:
        for item in confusion_matrices:
            f.write("\n模型: {0}\n".format(item['Model']))
            f.write(item['Matrix'].to_csv())
            f.write("\n")

    # ========= 改进 ROC 曲线生成与 CSV 保存 =========

    from scipy import interpolate
    import numpy as np
    
    n_points = 1000  # 定义统一的插值点数
    standard_fpr = np.linspace(0, 1, n_points)
    
    # 创建 DataFrame 保存所有模型的插值 ROC 数据
    roc_df = pd.DataFrame({'FPR': standard_fpr})
    
    # 对集成模型的 ROC 曲线进行插值，采用线性插值并固定端点（确保 FPR=0 返回 TPR=0，FPR=1 返回 TPR=1）
    ensemble_interp = interpolate.interp1d(fpr, tpr, kind='linear', bounds_error=False, fill_value=(0, 1))
    roc_df['Ensemble_TPR'] = ensemble_interp(standard_fpr)
    roc_df['Ensemble_AUC'] = roc_auc
    
    # 对各单独模型进行插值并保存到 DataFrame
    for name, model in models.items():
        model_proba = model.predict_proba(X_test)[:, 1]
        model_fpr, model_tpr, _ = roc_curve(y_test, model_proba)
        model_auc = auc(model_fpr, model_tpr)
        model_interp = interpolate.interp1d(model_fpr, model_tpr, kind='linear', bounds_error=False, fill_value=(0, 1))
        roc_df['{}_TPR'.format(name)] = model_interp(standard_fpr)
        roc_df['{}_AUC'.format(name)] = model_auc
    
    # 如果神经网络模型已训练，则同样处理其 ROC 曲线
    if nn_trained:
        nn_interp = interpolate.interp1d(nn_fpr, nn_tpr, kind='linear', bounds_error=False, fill_value=(0, 1))
        roc_df['Neural_Network_TPR'] = nn_interp(standard_fpr)
        roc_df['Neural_Network_AUC'] = nn_roc_auc
    
    # 保存所有模型的插值 ROC 数据到 CSV 文件
    roc_df.to_csv('roc_curves_data.csv', index=False)
    print("改进后的 ROC 插值数据已保存为 'roc_curves_data.csv'")
    
    # ========= 改进后的 ROC 曲线绘图 =========
    
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Guessing')
    
    # 绘制集成模型 ROC 曲线，增加 marker 参数和间隔标记（markevery）以便清晰显示
    plt.plot(fpr, tpr, label='集成模型 (AUC = {:.2f})'.format(roc_auc), lw=2, marker='o', markevery=50)
    
    # 遍历绘制各单独模型的 ROC 曲线
    for name, model in models.items():
        model_proba = model.predict_proba(X_test)[:, 1]
        model_fpr, model_tpr, _ = roc_curve(y_test, model_proba)
        model_auc = auc(model_fpr, model_tpr)
        plt.plot(model_fpr, model_tpr, label='{} (AUC = {:.2f})'.format(name, model_auc), 
                 lw=2, marker='s', markevery=50)
    
    # 如果神经网络已训练，也绘制其 ROC 曲线
    if nn_trained:
        plt.plot(nn_fpr, nn_tpr, label='神经网络 (AUC = {:.2f})'.format(nn_roc_auc), 
                 lw=2, marker='^', markevery=50)
    
    plt.xlabel('假阳性率 (FPR)', fontsize=12)
    plt.ylabel('真阳性率 (TPR)', fontsize=12)
    plt.title('改进后的 ROC 曲线', fontsize=15)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('roc_curves_improved.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("改进后的 ROC 曲线图已保存为 'roc_curves_improved.png'")

    # 9. 特征散点图（聚类着色）
    if len(bc_clusters) > 0:
        for i, feature1 in enumerate(extended_features):
            for j, feature2 in enumerate(extended_features):
                if i < j and feature1 in bc_full_features.columns and feature2 in bc_full_features.columns:
                    plt.figure(figsize=(10, 8))
                    scatter = plt.scatter(bc_full_features[feature1], bc_full_features[feature2],
                                          c=np.array(bc_clusters), cmap='jet', alpha=0.7)
                    plt.title('{}与{}的聚类散点图'.format(feature1, feature2), fontsize=15)
                    plt.xlabel(feature1, fontsize=12)
                    plt.ylabel(feature2, fontsize=12)
                    plt.colorbar(scatter, label='聚类编号')
                    plt.grid(True, linestyle='--', alpha=0.6)
                    plt.tight_layout()
                    plt.savefig('scatter_{}_{}.png'.format(feature1.replace('/', '_'), feature2.replace('/', '_')), dpi=300, bbox_inches='tight')
                    plt.show()

    # 10. 模型训练历史（神经网络）
    if nn_trained:
        try:
            plt.figure(figsize=(12, 5))
            if 'acc' in nn_history.history:
                accuracy_key = 'acc'
                val_accuracy_key = 'val_acc'
            else:
                accuracy_key = 'accuracy'
                val_accuracy_key = 'val_accuracy'
            
            plt.subplot(1, 2, 1)
            plt.plot(nn_history.history[accuracy_key], lw=2, marker='o')
            plt.plot(nn_history.history[val_accuracy_key], lw=2, marker='x')
            plt.title('神经网络模型准确率')
            plt.ylabel('准确率')
            plt.xlabel('训练周期')
            plt.legend(['训练集', '验证集'], loc='lower right')

            plt.subplot(1, 2, 2)
            plt.plot(nn_history.history['loss'], lw=2, marker='o')
            plt.plot(nn_history.history['val_loss'], lw=2, marker='x')
            plt.title('神经网络模型损失')
            plt.ylabel('损失')
            plt.xlabel('训练周期')
            plt.legend(['训练集', '验证集'], loc='upper right')
            plt.tight_layout()
            plt.savefig('nn_training_history.png', dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print("绘制训练历史图表时出错：", str(e))

    # 11. 模型比较条形图
    plt.figure(figsize=(12, 8))
    model_names = list(model_results.keys()) + ['Ensemble']
    accuracies = []
    for name in model_results.keys():
        accuracies.append(model_results[name]['accuracy'])
    accuracies.append(accuracy_score(y_test, y_pred))
    
    if nn_trained:
        model_names.append('Neural Network')
        accuracies.append(accuracy_score(y_test, nn_y_pred))

    plot_df = pd.DataFrame({'Model': model_names, 'Accuracy': accuracies})
    sns.barplot(x='Model', y='Accuracy', data=plot_df)
    plt.title('不同模型准确率比较', fontsize=15)
    plt.xlabel('模型', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    plt.ylim(0, 1.0)
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, '{:.4f}'.format(acc), ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 12. 随机森林特征重要性
    if hasattr(rf_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': common_features,
            'Importance': rf_model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title('随机森林特征重要性', fontsize=15)
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        plt.figure(figsize=(12, 6))
        feature_importance['Type'] = feature_importance['Feature'].apply(lambda x: "循环RNA" if x in base_features else "肿瘤标志物")
        sns.barplot(x='Feature', y='Importance', hue='Type', data=feature_importance)
        plt.title('特征重要性对比: 循环RNA vs 肿瘤标志物', fontsize=15)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

# 清理临时文件
try:
    if os.path.exists('temp_models/best_model.h5'):
        os.remove('temp_models/best_model.h5')
    if os.path.exists('temp_models'):
        os.rmdir('temp_models')
except:
    pass

# ============== 打印结果 ==============
print("\n========== 结果摘要 ==========")
silhouette_avg = silhouette_score(scaled_features, pca_df['Cluster'])
print("K-means轮廓系数: {:.4f}".format(silhouette_avg))
type_cluster_cross = pd.crosstab(pca_df['Type'], pca_df['Cluster'])
print("\n类型与聚类结果交叉表:")
print(type_cluster_cross)
stage_map = {'健康': 0, 'I期': 1, 'II期': 2, 'III期': 3, '未知': np.nan}
pca_df['Stage_Num'] = pca_df['Stage'].map(stage_map)
valid_rows = ~pca_df['Stage_Num'].isnull()
if valid_rows.any():
    stage_cluster_corr = np.corrcoef(pca_df.loc[valid_rows, 'Stage_Num'], pca_df.loc[valid_rows, 'Cluster'])[0, 1]
    print("\n分期与聚类的相关系数: {:.4f}".format(stage_cluster_corr))

if not skip_ml:
    print("\n集成模型混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))
    if nn_trained:
        print("\n神经网络混淆矩阵:")
        print(confusion_matrix(y_test, nn_y_pred))
    print("\n各模型性能比较:")
    print("模型名称            准确率      精确率      召回率      F1分数      AUC")
    print("-" * 70)
    for name, result in model_results.items():
        pred = result['predictions']
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred)
        rec = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        model_proba = models[name].predict_proba(X_test)[:, 1]
        model_fpr, model_tpr, _ = roc_curve(y_test, model_proba)
        model_auc = auc(model_fpr, model_tpr)
        print("{0:<20} {1:.4f}      {2:.4f}      {3:.4f}      {4:.4f}      {5:.4f}".format(name, acc, prec, rec, f1, model_auc))
    print("{0:<20} {1:.4f}      {2:.4f}      {3:.4f}      {4:.4f}      {5:.4f}".format(
        "集成模型", accuracy_score(y_test, y_pred), precision_score(y_test, y_pred),
        recall_score(y_test, y_pred), f1_score(y_test, y_pred), roc_auc))
    if nn_trained:
        print("{0:<20} {1:.4f}      {2:.4f}      {3:.4f}      {4:.4f}      {5:.4f}".format(
            "神经网络", accuracy_score(y_test, nn_y_pred), precision_score(y_test, nn_y_pred),
            recall_score(y_test, nn_y_pred), f1_score(y_test, nn_y_pred), nn_roc_auc))
    print("-" * 70)
    print("\n集成模型交叉验证准确率: {:.4f} ± {:.4f}".format(cv_scores.mean(), cv_scores.std()))
    if hasattr(rf_model, 'feature_importances_'):
        print("\n特征重要性:")
        for i, row in feature_importance.iterrows():
            feature_type = "循环RNA" if row['Feature'] in base_features else "肿瘤标志物"
            print("{}: {:.4f} ({})".format(row['Feature'], row['Importance'], feature_type))

print("\n分析完成! 所有图表已保存。")
