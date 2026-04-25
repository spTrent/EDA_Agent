import ollama
from tools.analyze_dataset import analyze_dataset
from tools.train_baseline import train_baseline
from tools.save_report import save_report

available_functions = {
    'analyze_dataset': analyze_dataset,
    'train_baseline': train_baseline,
    'save_report': save_report,
}


def run_agent(user_query: str):
    print(f"Пользователь: {user_query}\n")
    print("Агент думает...")

    system_prompt = (
        "Ты - Senior AI-ассистент дата-саентиста. Твоя задача: проанализировать данные, построить бейзлайн регрессии и сохранить результат в файл.\n"
        "СТРОГО СЛЕДУЙ АЛГОРИТМУ ПО ШАГАМ И НЕ ЗАВЕРШАЙ РАБОТУ ПОКА НЕ СОХРАНИШЬ ОТЧЕТ:\n"
        "Шаг 1. Вызови инструмент analyze_dataset.\n"
        "Шаг 2. Изучи ответ. Если найдены признаки с 80%+ пропусков или явные утечки (корреляция > 0.75), "
        "добавь эти колонки в список drop_columns при вызове инструмента train_baseline. Учти рекомендацию по кросс-валидации.\n"
        "Шаг 3. Вызови инструмент train_baseline.\n"
        "Шаг 4. ВНИМАНИЕ: Как только получишь метрики от train_baseline, сформируй текст отчета строго по Markdown-шаблону\n"
        "ОБЯЗАТЕЛЬНО вызови инструмент save_report, передав текст отчета в 'content'.\n"
        "Шаг 5. ТОЛЬКО ПОСЛЕ того как save_report вернет сообщение об успешном сохранении, заверши работу.\n\n"
        "Шаблон отчета (СТРОГО НА РУССКОМ ЯЗЫКЕ). СТРОГО СЛЕДУЙ ЕМУ:\n"
        "#Отчет по датасету [Название файла]\n\n"
        "**1. Обзор данных**\n"
        "- Размер: [X] строк, [Y] признаков.\n"
        "- Целевая переменная: [Target].\n\n"
        "**2. Проблемы и Очистка**\n"
        "- Выявлено: [Кратко перечисли найденные лики, мусорные пропуски или константы. Каждый признак со статистикой].\n"
        "- Удалены признаки: [Список колонок, переданных в drop_columns, и причина.].\n"
        "- Инсайт: [Упомяни мультиколлинеарность, если она была найдена, или напиши 'Данные чистые'.].\n\n"
        "Мультиколлинеарные признаки сгруппируй по мультиколлинеарности"
        "**3. Результаты Baseline-модели (CatBoost)**\n"
        "- Метод оценки: [Train/Test или CV].\n"
        "- R2: [Значение] (Объясняет [X]% дисперсии).\n"
        "- MAE: [Значение] (Средняя ошибка предсказания).\n"
        "- Модель сохранена по пути: [Путь].\n\n"
        "**4. Главные предикторы (Feature Importance)**\n"
        "1. [Признак 1] — [Вес]%\n"
        "2. [Признак 2] — [Вес]%\n"
        "3. [Признак 3] — [Вес]%\n"
        "Подведи итог, интерпретируй результаты."
    )

    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_query}
    ]

    tools = [analyze_dataset, train_baseline, save_report]

    for _ in range(5):
        response = ollama.chat(
            model='qwen2.5:7b',
            messages=messages,
            tools=tools
        )

        messages.append(response['message'])

        if not response['message'].get('tool_calls'):
            print("\nФИНАЛЬНЫЙ ОТВЕТ АГЕНТА:")
            print(response['message']['content'])
            break

        for tool_call in response['message']['tool_calls']:
            func_name = tool_call['function']['name']
            func_args = tool_call['function']['arguments']

            if func_name != 'save_report':
                print(f" -> Агент вызывает инструмент: {func_name} с аргументами {func_args}")
            else:
                print(f" -> Агент сохраняет отчет")

            func_to_call = available_functions.get(func_name)
            if func_to_call:
                try:
                    tool_output = str(func_to_call(**func_args))
                    if func_name == 'train_baseline':
                        tool_output += (
                            "\n\n[СИСТЕМНОЕ СООБЩЕНИЕ]: Метрики успешно получены. "
                            "ТЕПЕРЬ НЕМЕДЛЕННО ВЫЗОВИ ИНСТРУМЕНТ save_report! "
                            "Упакуй весь сформированный Markdown-отчет в аргумент 'content'. "
                            "Тебе строго запрещено писать отчет обычным сообщением!"
                        )
                    elif func_name == 'save_report':
                        tool_output += (
                            "\n\n[СИСТЕМНОЕ СООБЩЕНИЕ]: Файл успешно сохранен. "
                            "Теперь напиши пользователю краткое сообщение о том, что работа завершена, "
                            "и укажи название файла, в котором лежит отчет. Инструменты больше не вызывай."
                        )
                except Exception as e:
                    tool_output = f"Ошибка выполнения {func_name}: {e}"
            else:
                tool_output = f"Инструмент {func_name} не найден."

            print(f" <- Инструмент вернул результат (передаю агенту).")

            messages.append({
                'role': 'tool',
                'content': tool_output,
                'name': func_name
            })


if __name__ == "__main__":
    query = "У меня есть файл prices.csv. Целевая переменная SalePrice. Сделай анализ данных и обучи базовую модель. Обязательно выкини мусорные колонки, если найдешь."
    run_agent(query)