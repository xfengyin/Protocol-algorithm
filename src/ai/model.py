# AI决策模型定义
"""LEACH协议算法的AI优化模型定义，用于智能簇首选择和分簇优化。"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def create_cluster_model(model_type: str = 'random_forest'):
    """
    创建用于WSN分簇优化的机器学习模型。
    
    参数:
    model_type: 模型类型，可选值为'random_forest'、'svm'或'mlp'
    
    返回:
    model: 配置好的机器学习模型
    """
    # 创建预处理和模型管道
    if model_type == 'random_forest':
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            ))
        ])
    elif model_type == 'svm':
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42
            ))
        ])
    elif model_type == 'mlp':
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate='adaptive',
                max_iter=200,
                random_state=42
            ))
        ])
    else:
        raise ValueError(f"未知的模型类型: {model_type}")
    
    return model

class ClusterAIModel:
    """WSN分簇优化的AI模型类，封装模型的训练和推理功能。"""
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        初始化AI模型。
        
        参数:
        model_type: 模型类型
        """
        self.model = create_cluster_model(model_type)
        self.model_type = model_type
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        训练AI模型。
        
        参数:
        X_train: 训练数据
        y_train: 训练标签
        X_val: 验证数据（可选）
        y_val: 验证标签（可选）
        
        返回:
        history: 训练历史记录（对于scikit-learn模型，返回None）
        """
        # 训练模型
        self.model.fit(X_train, y_train)
        
        return None
    
    def predict(self, X):
        """
        使用训练好的模型进行预测。
        
        参数:
        X: 输入数据
        
        返回:
        predictions: 预测结果（概率值）
        """
        return self.model.predict_proba(X)[:, 1]  # 返回正类（簇首）的概率
    
    def save_model(self, filepath: str):
        """
        保存模型到文件。
        
        参数:
        filepath: 模型保存路径
        """
        import joblib
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath: str):
        """
        从文件加载模型。
        
        参数:
        filepath: 模型加载路径
        """
        import joblib
        self.model = joblib.load(filepath)
