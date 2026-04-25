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
        "Ты - специализированный ассистент дата-саентиста. Твоя задача: провести анализ данных, обучить модель и сохранить отчет.\n"
        "Алгоритм работы (строго по шагам):\n"
        "Шаг 1. Вызови analyze_dataset.\n"
        "Шаг 2. Изучи результаты анализа. Если найдены признаки с 80%+ пропусков или явные утечки (корреляция > 0.75), "
        "обязательно добавь их в список drop_columns при вызове инструмента train_baseline.\n"
        "Шаг 3. Вызови train_baseline для обучения модели.\n"
        "Шаг 4. ВНИМАНИЕ: Как только получишь метрики, ОБЯЗАТЕЛЬНО вызови инструмент save_report. "
        "В аргумент 'content' передай текст отчета. Пиши понятным текстом, без сложных спецсимволов и таблиц.\n"
        "В отчете обязательно укажи:\n"
        "- Размер данных и целевую переменную.\n"
        "- Какие проблемы найдены и какие колонки были удалены.\n"
        "- Метод оценки, метрики R2 и MAE.\n"
        "- Топ-5 самых важных признака.\n"
        "Напиши вывод. Интерпретируй результаты, поразмышляй, почему они такие."
        "Шаг 5. После того как инструмент save_report вернет подтверждение, напиши пользователю: 'Отчет успешно сохранен' и заверши работу."
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