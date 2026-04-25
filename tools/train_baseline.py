from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, r2_score


def train_baseline(file_path: str, target_column: str, drop_columns: list | str | None = None, use_cv: bool = False) -> str:
    """
    Инструмент для обучения базовой модели градиентного бустинга (регрессия).
    Сохраняет модель на диск и возвращает метрики.

    Args:
        file_path: Путь к CSV файлу.
        target_column: Название целевой переменной.
        drop_columns: Список колонок, которые нужно удалить перед обучением.
        use_cv: Флаг использования кросс-валидации (K-Fold, 5 фолдов).

    Returns:
        Строка с метриками (R2, MAE) и топ-5 самых важных признаков.
    """
    if drop_columns is None:
        drop_columns = []

    if isinstance(drop_columns, str):
        drop_columns = [col.strip() for col in drop_columns.split(',')]

    try:
        df = pd.read_csv(file_path)

        if drop_columns:
            df = df.drop(columns=[col for col in drop_columns if col in df.columns])

        df = df.dropna(subset=[target_column])

        X = df.drop(target_column, axis=1)
        y = df[target_column]

        cat_features = X.select_dtypes(include=['object', 'category', 'str']).columns.tolist()
        for col in cat_features:
            X[col] = X[col].astype(str).fillna('NaN')

        cb_params = {
            'iterations': 500,
            'learning_rate': 0.05,
            'depth': 6,
            'loss_function': 'RMSE',
            'verbose': False,
            'allow_writing_files': False
        }

        if use_cv:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            r2_scores, mae_scores = [], []

            for train_idx, val_idx in kf.split(X):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = CatBoostRegressor(**cb_params)
                model.fit(X_tr, y_tr, cat_features=cat_features)

                preds = model.predict(X_val)
                r2_scores.append(r2_score(y_val, preds))
                mae_scores.append(mean_absolute_error(y_val, preds))

            r2 = np.mean(r2_scores)
            mae = np.mean(mae_scores)

            model_to_save = CatBoostRegressor(**cb_params)
            model_to_save.fit(X, y, cat_features=cat_features)
            eval_method = "Кросс-валидация (5 фолдов)"

        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model_to_save = CatBoostRegressor(**cb_params)
            model_to_save.fit(X_train, y_train, cat_features=cat_features)

            preds = model_to_save.predict(X_test)
            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            eval_method = "Train/Test Split (80/20)"

        model_path = "models/baseline_model.cbm"
        model_to_save.save_model(model_path)

        importances = model_to_save.get_feature_importance()
        fi_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
        top_5_features = fi_df.sort_values(by='importance', ascending=False).head(5)

        top_5_str = ", ".join([f"{row['feature']} ({row['importance']:.1f}%)" for _, row in top_5_features.iterrows()])

        return (f"Обучение завершено. Модель сохранена в '{model_path}'.\n"
                f"Метод оценки: {eval_method}\n"
                f"Метрики:\n"
                f"- R2: {r2:.3f}\n"
                f"- MAE: {mae:.3f}\n"
                f"Топ-5 важных признаков:\n{top_5_str}")

    except Exception as e:
        return f"Критическая ошибка при обучении: {e}"

if __name__ == '__main__':
    print(train_baseline('../prices.csv', 'SalePrice', use_cv=True))