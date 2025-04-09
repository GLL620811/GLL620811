from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_accuracy(y_true, y_pred):
    # 计算准确率（Accuracy）
    accuracy = accuracy_score(y_true, y_pred)

    # 计算精确率（Precision）
    precision = precision_score(y_true, y_pred, average='macro')  # 可以选择'macro'、'micro'或'weighted'

    # 计算召回率（Recall）
    recall = recall_score(y_true, y_pred, average='macro')  # 可以选择'macro'、'micro'或'weighted'

    # 计算F1分数
    f1 = f1_score(y_true, y_pred, average='macro')  # 可以选择'macro'、'micro'或'weighted'

    # 打印结果
    print(f"准确率（Accuracy）: {accuracy:.4f}")
    print(f"精确率（Precision）: {precision:.4f}")
    print(f"召回率（Recall）: {recall:.4f}")
    print(f"F1 分数: {f1:.4f}")