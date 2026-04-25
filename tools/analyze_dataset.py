import os
import pandas as pd
import numpy as np


def analyze_dataset(file_path: str, target_column: str) -> str:
    if not os.path.exists(file_path):
        return f"Ошибка: Файл {file_path} не найден. Проверь путь."

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return f"Ошибка при чтении CSV: {e}"

    if target_column not in df.columns:
        return f"Ошибка: колонка '{target_column}' не найдена. Доступны: {', '.join(df.columns)}"

    report = []
    report.append(f"Датасет загружен. Строк: {df.shape[0]}, Колонок: {df.shape[1]}")

    cat_features = df.drop(columns=[target_column]).select_dtypes(
        include=['object', 'category', 'str']).columns.tolist()
    num_features = df.drop(columns=[target_column]).select_dtypes(include=[np.number]).columns.tolist()

    report.append(f"Категориальные признаки: {cat_features if cat_features else 'Нет'}")

    missing_info = []
    missing_threshold = 0.8
    missing_ratios = df.isna().mean()

    for col, ratio in missing_ratios.items():
        if ratio > missing_threshold:
            missing_info.append(
                f"ВНИМАНИЕ! Признак '{col}' содержит {ratio * 100:.1f}% пропусков. Рекомендуется к удалению."
            )

    if missing_info:
        report.append("\nНайдено критическое количество пропусков:")
        report.extend(missing_info)

    cardinality_info = []
    if cat_features:
        for col in cat_features:
            unique_count = df[col].nunique()
            if unique_count == df.shape[0]:
                cardinality_info.append(
                    f"ВНИМАНИЕ! Категориальный признак '{col}' имеет 100% уникальных значений. "
                    f"Скорее всего, это ID или хэш. Обязателен к удалению."
                )
            elif unique_count > df.shape[0] * 0.9:
                cardinality_info.append(
                    f"ВНИМАНИЕ! Категориальный признак '{col}' имеет очень высокую кардинальность "
                    f"({unique_count} уник. значений). Может привести к переобучению."
                )

    integer_features = df.drop(columns=[target_column]).select_dtypes(include=['int64', 'int32']).columns.tolist()
    for col in integer_features:
        unique_count = df[col].nunique()
        if unique_count == df.shape[0]:
            cardinality_info.append(
                f"ВНИМАНИЕ! Числовой признак '{col}' имеет 100% уникальных значений. "
                f"Скорее всего, это колонка с ID или индексами строк. Обязателен к удалению."
            )

    if cardinality_info:
        report.append("\nПроблемы с кардинальностью:")
        report.extend(cardinality_info)

    low_variance = []
    variance_threshold = 1e-4

    for col in num_features:
        col_var = df[col].var()

        if pd.isna(col_var) or col_var == 0:
            low_variance.append(f"ВНИМАНИЕ! Признак '{col}' константный (дисперсия = 0). Обязателен к удалению.")
        elif col_var < variance_threshold:
            low_variance.append(
                f"ВНИМАНИЕ! Признак '{col}' имеет околонулевую дисперсию ({col_var:.5f}). Является кандидатом на удаление.")

    if low_variance:
        report.append("\nНайдена низкая дисперсия:")
        report.extend(low_variance)
    else:
        report.append("Признаков с нулевой или крайне низкой дисперсией не обнаружено.")

    leaks = []
    for col in num_features:
        if df[col].var() > 0:
            corr = df[col].corr(df[target_column])
            if abs(corr) > 0.9:
                leaks.append(
                    f"ВНИМАНИЕ! Признак '{col}' имеет сильную корреляцию {corr:.2f} с таргетом. Возможна утечка данных!")

    if leaks:
        report.append("Найдены возможные утечки данных (Data Leakage):")
        report.extend(leaks)
    else:
        report.append("Явных утечек данных (ликов) среди числовых признаков не найдено.")

    multicollinearity = []

    corr_matrix = df[num_features].corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    high_corr_threshold = 0.75
    for column in upper_triangle.columns:
        highly_correlated_idx = upper_triangle.index[upper_triangle[column] > high_corr_threshold].tolist()
        for row in highly_correlated_idx:
            val = upper_triangle.loc[row, column]
            multicollinearity.append(
                f"ИНФО: Признаки '{row}' и '{column}' сильно коррелируют между собой ({val:.2f}). "
                f"Возможно дублирование информации."
            )

    if multicollinearity:
        report.append("Найдена мультиколлинеарность:")
        report.extend(multicollinearity)
    else:
        report.append("Сильно скоррелированных между собой числовых признаков не обнаружено.")

    training_recommendation = []
    if df.shape[0] < 50_000:
        training_recommendation.append(
            f"ИНФО: Датасет небольшой ({df.shape[0]} строк). "
            f"Рекомендуется использовать кросс-валидацию (K-Fold CV) для более надежной оценки."
        )
    else:
        training_recommendation.append(
            f"ИНФО: Размер датасета достаточен ({df.shape[0]} строк). "
            f"Рекомендуется использовать стандартный Train/Test Split (например, 80/20)."
        )

    report.append("\nСтратегия обучения:")
    report.extend(training_recommendation)

    return "\n".join(report)


if __name__ == '__main__':
    print(analyze_dataset('../prices.csv', 'SalePrice'))