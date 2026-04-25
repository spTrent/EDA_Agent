import os, datetime

def save_report(content: str, filename: str | None = None) -> str:
    """
    Инструмент для сохранения итогового аналитического отчета в файл.

    Args:
        content: Полный текст отчета в формате Markdown.
        filename: Имя файла (по умолчанию final_report.md).
    """
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M")
        filename = f"report_{timestamp}.md"

    if not filename.endswith('.md'):
        filename += '.md'

    try:
        report_dir = "reports"
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)

        full_path = os.path.join(report_dir, filename)

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"Отчет успешно сохранен в файл: {full_path}"
    except Exception as e:
        return f"Ошибка при сохранении файла: {e}"