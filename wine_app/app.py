import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics


APP_DIR = Path(__file__).parent


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


def predict_batch(model, frame, features, threshold):
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise ValueError('Не хватает столбцов: ' + ', '.join(missing))
    if frame.empty:
        raise ValueError('В файле нет образцов вина.')
    try:
        X = frame[features].apply(pd.to_numeric, errors='raise')
    except (ValueError, TypeError):
        raise ValueError('Все признаки должны содержать числа.') from None
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError('Заполните пропуски и удалите бесконечные значения.')
    if not X['is_red'].isin([0, 1]).all():
        raise ValueError('В столбце is_red допустимы только 0 и 1.')
    result = frame.copy()
    result['Вероятность хорошего вина'] = model.predict_proba(X)[:, 1]
    result['Рекомендовать'] = (
        result['Вероятность хорошего вина'] >= threshold
    ).astype(int)
    return result


st.set_page_config(page_title='Подборка вин', layout='wide')
st.title('Подборка хороших вин')
st.caption('Оцените вино по лабораторному анализу и выберите порог для рекомендаций.')

model = load_model()
test = load_test()
config = json.loads((APP_DIR / 'config.json').read_text(encoding='utf-8'))
X_test = test.drop(columns='target')
y_test = test['target']
features = X_test.columns.tolist()
p_test = test_proba(model, X_test)

threshold = st.sidebar.slider(
    'Порог рекомендации', 0.0, 1.0,
    value=float(config['threshold']), step=0.001, format='%.3f'
)
st.sidebar.caption('Один порог применяется к оценке вина, метрикам и загруженному CSV.')

single_tab, metrics_tab, batch_tab, weights_tab = st.tabs([
    'Оценка одного вина', 'Модельные метрики',
    'Пакетное предсказание', 'Что выучила модель'
])

labels = {
    'fixed_acidity': 'Фиксированная кислотность',
    'volatile_acidity': 'Летучая кислотность',
    'citric_acid': 'Лимонная кислота',
    'residual_sugar': 'Остаточный сахар',
    'chlorides': 'Хлориды',
    'free_sulfur_dioxide': 'Свободный диоксид серы',
    'total_sulfur_dioxide': 'Общий диоксид серы',
    'density': 'Плотность',
    'pH': 'pH',
    'sulphates': 'Сульфаты',
    'alcohol': 'Алкоголь'
}

with single_tab:
    st.subheader('Лабораторные показатели')
    with st.form('wine_form'):
        values = {}
        columns = st.columns(3)
        for i, (feature, label) in enumerate(labels.items()):
            with columns[i % 3]:
                values[feature] = st.number_input(
                    label,
                    min_value=float(X_test[feature].min()),
                    max_value=float(X_test[feature].max()),
                    value=float(X_test[feature].median()),
                    step=float((X_test[feature].max() - X_test[feature].min()) / 100),
                    format='%.5f'
                )
        color = st.selectbox('Цвет вина', ['Белое', 'Красное'])
        values['is_red'] = int(color == 'Красное')
        submitted = st.form_submit_button('Оценить вино')

    if submitted:
        sample = pd.DataFrame([values])[features]
        st.session_state['wine_probability'] = float(model.predict_proba(sample)[0, 1])
    probability = st.session_state.get('wine_probability')
    if probability is not None:
        st.metric('Вероятность хорошего вина', f'{probability:.1%}')
        if probability >= threshold:
            st.success('Рекомендовать вино в подборку')
        else:
            st.info('Не включать в подборку при текущем пороге')

with metrics_tab:
    st.subheader('Качество на тестовой выборке')
    pred = (p_test >= threshold).astype(int)
    precision = metrics.precision_score(y_test, pred, zero_division=0)
    recall = metrics.recall_score(y_test, pred, zero_division=0)
    f1 = metrics.f1_score(y_test, pred, zero_division=0)
    columns = st.columns(3)
    for column, name, value in zip(columns, ['Precision', 'Recall', 'F1'], [precision, recall, f1]):
        column.metric(name, f'{value:.3f}')

    cm = metrics.confusion_matrix(y_test, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    metrics.ConfusionMatrixDisplay(cm, display_labels=['Остальные', 'Хорошие']).plot(
        ax=ax, cmap='Blues', colorbar=False
    )
    ax.set_xlabel('Предсказание')
    ax.set_ylabel('Истинный класс')
    st.pyplot(fig)
    plt.close(fig)

    tn, fp, fn, tp = cm.ravel()
    fpr, tpr, _ = metrics.roc_curve(y_test, p_test)
    pr_precision, pr_recall, _ = metrics.precision_recall_curve(y_test, p_test)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(fpr, tpr, label=f'ROC-AUC = {metrics.roc_auc_score(y_test, p_test):.3f}')
    axes[0].plot([0, 1], [0, 1], 'k--', label='Случайная модель')
    axes[0].scatter(fp / (fp + tn), recall, color='red', label='Текущий порог', zorder=3)
    axes[0].set(xlabel='FPR', ylabel='TPR', title='ROC-кривая')
    axes[1].step(pr_recall, pr_precision, where='post', label=f'AP = {metrics.average_precision_score(y_test, p_test):.3f}')
    axes[1].axhline(y_test.mean(), color='black', linestyle='--', label='Случайная модель')
    axes[1].scatter(recall, precision, color='red', label='Текущий порог', zorder=3)
    axes[1].set(xlabel='Recall', ylabel='Precision', title='PR-кривая')
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.write(
        'Повышение порога уменьшает подборку и обычно повышает точность рекомендаций. '
        'Снижение порога помогает найти больше хороших вин, но увеличивает риск '
        'включить посредственное вино.'
    )

with batch_tab:
    st.subheader('Оценка вин из CSV')
    st.write('Загрузите 12 признаков. is_red: 1 — красное, 0 — белое. Рекомендовать: 1 — да, 0 — нет.')
    st.download_button(
        'Скачать пример CSV',
        X_test.head(5).to_csv(index=False).encode('utf-8-sig'),
        file_name='example.csv', mime='text/csv'
    )
    uploaded = st.file_uploader('CSV с признаками вин', type=['csv'])
    if uploaded is not None:
        try:
            frame = pd.read_csv(uploaded)
            result = predict_batch(model, frame, features, threshold)
        except (ValueError, TypeError, pd.errors.ParserError, UnicodeError) as error:
            st.error(str(error))
        else:
            st.dataframe(result, use_container_width=True)
            st.download_button(
                'Скачать предсказания',
                result.to_csv(index=False).encode('utf-8-sig'),
                file_name='wine_predictions.csv', mime='text/csv'
            )

with weights_tab:
    st.subheader('Коэффициенты после масштабирования')
    coefficients = pd.Series(
        model.named_steps['logisticregression'].coef_[0], index=features
    ).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        coefficients.index, coefficients.values,
        color=['#9f3450' if value < 0 else '#43816b' for value in coefficients]
    )
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Коэффициент')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    positive = coefficients[coefficients > 0].nlargest(3).index
    negative = coefficients[coefficients < 0].nsmallest(3).index
    st.write('Сильнее повышают оценку: ' + ', '.join(labels.get(name, name) for name in positive) + '.')
    st.write('Сильнее понижают оценку: ' + ', '.join(labels.get(name, name) for name in negative) + '.')
