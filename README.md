# MMAD — Wine Quality

Приложение на Streamlit для выбора хороших вин по результатам лабораторного анализа. Хорошее вино — оценка не ниже 7.

В [wine_app](wine_app/) находятся код, обученная модель, выбранный порог, тестовая выборка и зависимости.

## Запуск

Проверено на Python 3.13.5.

```bash
python -m pip install -r wine_app/requirements.txt
python -m streamlit run wine_app/app.py
```
