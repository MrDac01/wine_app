"""
Демо-приложение «Подборка хороших вин» (ДЗ-3, часть 4).

Запуск:  streamlit run app.py

Приложение ничего не обучает: оно загружает готовый пайплайн
StandardScaler + LogisticRegression (model.joblib), порог по умолчанию (config.json)
и тестовую выборку (test.csv), сохранённые из ноутбука.

Про кэширование (задание 4.5)
-----------------------------
Streamlit перезапускает этот скрипт сверху вниз при КАЖДОМ действии пользователя
(сдвинули слайдер, ввели число, загрузили файл). Поэтому всё дорогое кэшируем:

* Модель — через st.cache_resource. Это «ресурс»: один общий объект на весь сервер,
  который не копируется и не сериализуется между перезапусками — все сессии получают
  ссылку на один и тот же объект. Модель мы только читаем (predict_proba), не меняем,
  так что делить её между пользователями безопасно. st.cache_data для модели не подходит:
  он при каждом обращении возвращает КОПИЮ результата (через pickle), т.е. на каждый
  перезапуск модель бы заново сериализовалась/десериализовалась — смысл кэша теряется,
  а для объектов, которые плохо сериализуются (соединения с БД, большие модели), он вообще
  не работает.
* Таблицы и вычисленные массивы (test.csv, вероятности на тесте, точки ROC/PR-кривых) —
  через st.cache_data. Это «данные»: результат зависит только от аргументов, и Streamlit
  возвращает каждому вызову свою копию. Если кто-то случайно изменит DataFrame
  (например, добавит столбец), это не испортит кэш для остальных.
* Без декоратора joblib.load выполнялся бы при каждом движении слайдера: модель заново
  читалась бы с диска и распаковывалась, приложение тормозило бы (на больших моделях —
  секунды на каждое действие) и тратило лишнюю память и диск. Вероятности на тесте тоже
  пересчитывались бы каждый раз, хотя от порога они не зависят: при сдвиге порога
  меняется только сравнение proba >= t, а predict_proba заново не вызывается.

Аргумент _model в test_proba начинается с подчёркивания — так Streamlit не пытается
его хешировать (модель нехешируема), и ключ кэша строится только по X.
"""
import io
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

APP_DIR = Path(__file__).parent   # пути строим от папки с app.py, а не от текущей директории

FEATURE_LABELS = {
    'fixed_acidity': 'Фиксированная кислотность, г/л',
    'volatile_acidity': 'Летучая кислотность, г/л',
    'citric_acid': 'Лимонная кислота, г/л',
    'residual_sugar': 'Остаточный сахар, г/л',
    'chlorides': 'Хлориды, г/л',
    'free_sulfur_dioxide': 'Свободный SO₂, мг/л',
    'total_sulfur_dioxide': 'Общий SO₂, мг/л',
    'density': 'Плотность, г/см³',
    'pH': 'pH',
    'sulphates': 'Сульфаты, г/л',
    'alcohol': 'Алкоголь, % об.',
    'is_red': 'Красное вино',
}

st.set_page_config(page_title='Подборка хороших вин', page_icon='🍷', layout='wide')


# ----------------------------------------------------------------------------- загрузка
@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_config():
    return json.loads((APP_DIR / 'config.json').read_text())


@st.cache_data
def load_test():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    """Вероятности класса 1 на тесте — считаются один раз, от порога не зависят."""
    return _model.predict_proba(X)[:, 1]


@st.cache_data
def curves(y_true, proba):
    """Точки ROC- и PR-кривых и площади под ними — тоже не зависят от порога."""
    fpr, tpr, _ = metrics.roc_curve(y_true, proba)
    prec, rec, _ = metrics.precision_recall_curve(y_true, proba)
    return {'fpr': fpr, 'tpr': tpr, 'precision': prec, 'recall': rec,
            'roc_auc': metrics.roc_auc_score(y_true, proba),
            'pr_auc': metrics.average_precision_score(y_true, proba)}


def decimals_of(series, max_dec=5):
    """Сколько знаков после запятой реально встречается в столбце (для шага number_input)."""
    for d in range(max_dec + 1):
        if np.allclose(series, series.round(d), rtol=0, atol=1e-9):
            return d
    return max_dec


model = load_model()
config = load_config()
test_df = load_test()

FEATURES = list(model.feature_names_in_)          # порядок признаков, на котором обучалась модель
X_test, y_test = test_df[FEATURES], test_df['target'].to_numpy()
proba_test = test_proba(model, X_test)

# ----------------------------------------------------------------------------- сайдбар
st.sidebar.header('⚙️ Настройки')
threshold = st.sidebar.slider(
    'Порог отнесения к «хорошим» винам', min_value=0.0, max_value=1.0,
    value=float(config['threshold']), step=0.001, format='%.3f', key='threshold',
    help='Вино попадает в подборку, если вероятность «хорошего» не ниже порога.')
st.sidebar.caption(f"Порог по умолчанию (подобран на валидации по максимуму F1): "
                   f"**{config['threshold']:.3f}**")
st.sidebar.markdown(
    '**Как читать порог:**\n'
    '- ⬆️ выше порог → в подборку попадает меньше вин, зато среди них больше действительно хороших;\n'
    '- ⬇️ ниже порог → подборка шире, но в неё чаще проскакивают посредственные вина.')

st.title('🍷 Подборка хороших вин')
st.caption('Модель: StandardScaler + логистическая регрессия. «Хорошее» вино — оценка экспертов не ниже 7.')

tab_one, tab_metrics, tab_batch, tab_coef = st.tabs(
    ['🍾 Оценка одного вина', '📊 Модельные метрики', '📁 Пакетное предсказание', '🧠 Что выучила модель'])

# ----------------------------------------------------------------------------- 4.1
with tab_one:
    st.subheader('Оценка одного вина')
    st.write('Введите результаты лабораторного анализа. Диапазоны и значения по умолчанию '
             '(медианы) взяты из тестовой выборки.')

    values = {}
    wine_type = st.radio('Тип вина', ['Белое', 'Красное'], horizontal=True)
    values['is_red'] = int(wine_type == 'Красное')

    numeric_features = [f for f in FEATURES if f != 'is_red']
    cols = st.columns(3)
    for i, feat in enumerate(numeric_features):
        col = test_df[feat]
        dec = decimals_of(col)
        with cols[i % 3]:
            values[feat] = st.number_input(
                FEATURE_LABELS.get(feat, feat),
                min_value=float(col.min()), max_value=float(col.max()),
                value=float(col.median()), step=10.0 ** -dec, format=f'%.{dec}f', key=f'in_{feat}')

    wine = pd.DataFrame([values])[FEATURES]
    p_good = float(model.predict_proba(wine)[0, 1])
    is_good = p_good >= threshold

    c1, c2 = st.columns(2)
    c1.metric('Вероятность, что вино хорошее', f'{p_good:.1%}')
    c2.metric('Текущий порог', f'{threshold:.3f}')
    if is_good:
        st.success(f'✅ Берём в подборку: вероятность {p_good:.1%} ≥ порога {threshold:.1%}.')
    else:
        st.warning(f'❌ Не берём в подборку: вероятность {p_good:.1%} < порога {threshold:.1%}.')

# ----------------------------------------------------------------------------- 4.2
with tab_metrics:
    st.subheader('Качество модели на тестовой выборке')
    st.write(f'Тестовая выборка: {len(y_test)} вин, из них хороших — {int(y_test.sum())} '
             f'({y_test.mean():.1%}). Всё ниже пересчитывается при движении слайдера порога.')

    pred = (proba_test >= threshold).astype(int)        # единственное, что зависит от порога
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr_t = fp / (fp + tn) if fp + tn else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric('Precision', f'{precision:.3f}', help='Доля действительно хороших среди попавших в подборку')
    m2.metric('Recall', f'{recall:.3f}', help='Доля хороших вин, которые попали в подборку')
    m3.metric('F1', f'{f1:.3f}', help='Среднее гармоническое precision и recall')
    m4.metric('Вин в подборке', f'{tp + fp} из {len(y_test)}')

    st.info(
        f'Из **{tp + fp}** вин в подборке **{tp}** хорошие и **{fp}** посредственные (FP). '
        f'**{fn}** хороших вин модель пропустила (FN). '
        '\n\n**Для сомелье:** если **поднять** порог, подборка станет короче и «надёжнее» — '
        'посредственных вин в ней будет меньше (precision растёт), но больше хороших вин останется за бортом '
        '(recall падает). Если **опустить** порог — подборка станет шире и пропустит меньше хороших вин, '
        'но в неё чаще будут попадать посредственные.')

    cv = curves(y_test, proba_test)
    g1, g2, g3 = st.columns(3)

    with g1:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        metrics.ConfusionMatrixDisplay(np.array([[tn, fp], [fn, tp]]),
                                       display_labels=['не хорошее', 'хорошее']).plot(
            ax=ax, cmap='Blues', values_format='d', colorbar=False)
        ax.set_xlabel('Предсказание модели')
        ax.set_ylabel('Истинный класс')
        ax.set_title('Матрица ошибок')
        st.pyplot(fig)
        plt.close(fig)

    with g2:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.plot(cv['fpr'], cv['tpr'], lw=2, label=f"ROC (AUC = {cv['roc_auc']:.3f})")
        ax.plot([0, 1], [0, 1], 'k:', label='случайная модель')
        ax.scatter([fpr_t], [recall], s=90, color='red', edgecolor='k', zorder=5,
                   label=f'порог {threshold:.3f}')
        ax.set_xlabel('FPR (доля посредственных в подборке\nот всех посредственных)')
        ax.set_ylabel('TPR = recall')
        ax.set_title('ROC-кривая')
        ax.legend(loc='lower right', fontsize=8)
        st.pyplot(fig)
        plt.close(fig)

    with g3:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.plot(cv['recall'], cv['precision'], lw=2, label=f"PR (AP = {cv['pr_auc']:.3f})")
        ax.axhline(y_test.mean(), color='k', ls=':', label='случайная модель')
        if tp + fp > 0:
            ax.scatter([recall], [precision], s=90, color='red', edgecolor='k', zorder=5,
                       label=f'порог {threshold:.3f}')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_ylim(0, 1.02)
        ax.set_title('PR-кривая')
        ax.legend(loc='upper right', fontsize=8)
        st.pyplot(fig)
        plt.close(fig)

# ----------------------------------------------------------------------------- 4.3
with tab_batch:
    st.subheader('Пакетное предсказание')
    batch_threshold = st.session_state['threshold']     # тот же порог, что и в слайдере сайдбара
    st.write(f'Загрузите CSV-файл с признаками вин (столбец `target` не нужен). '
             f'Используется порог из сайдбара: **{batch_threshold:.3f}**.')
    st.caption('Нужные столбцы: ' + ', '.join(f'`{f}`' for f in FEATURES))

    sample_path = APP_DIR / 'sample_batch.csv'
    if sample_path.exists():
        st.download_button('Скачать пример файла', sample_path.read_bytes(),
                           file_name='sample_batch.csv', mime='text/csv')

    uploaded = st.file_uploader('CSV-файл', type=['csv'])
    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
        except Exception as e:  # битый файл, не CSV, пустой файл и т.п.
            batch = None
            st.error(f'Не удалось прочитать файл как CSV: {e}')

        if batch is not None:
            missing = [f for f in FEATURES if f not in batch.columns]
            if batch.empty:
                st.error('Файл пустой — в нём нет ни одной строки с данными.')
            elif missing:
                st.error('В файле не хватает столбцов: ' + ', '.join(f'`{m}`' for m in missing)
                         + '. Добавьте их и загрузите файл снова.')
            else:
                feats = batch[FEATURES].apply(pd.to_numeric, errors='coerce')
                bad_rows = feats.isna().any(axis=1)
                if bad_rows.all():
                    st.error('Ни в одной строке нет корректных числовых значений всех признаков.')
                else:
                    if bad_rows.any():
                        st.warning(f'Строк с пустыми или нечисловыми значениями: {int(bad_rows.sum())} '
                                   f'(номера: {", ".join(map(str, np.where(bad_rows)[0][:20]))}'
                                   f'{"…" if bad_rows.sum() > 20 else ""}). Для них предсказание не делается.')
                    result = batch.copy()
                    proba = pd.Series(np.nan, index=batch.index)
                    proba[~bad_rows] = model.predict_proba(feats[~bad_rows])[:, 1]
                    result['proba_good'] = proba.round(4)
                    result['decision'] = np.where(proba.isna(), 'нет данных',
                                                  np.where(proba >= batch_threshold, 'в подборку', 'не брать'))

                    n_ok = int((~bad_rows).sum())
                    n_in = int((result['decision'] == 'в подборку').sum())
                    st.write(f'Обработано вин: **{n_ok}**, в подборку попадает: **{n_in}**.')
                    st.dataframe(result, width='stretch')

                    buf = io.StringIO()
                    result.to_csv(buf, index=False)
                    st.download_button('⬇️ Скачать результат (CSV)', buf.getvalue().encode('utf-8'),
                                       file_name='wine_predictions.csv', mime='text/csv')

# ----------------------------------------------------------------------------- 4.4
with tab_coef:
    st.subheader('Что выучила модель')
    lr = model[-1]
    coefs = pd.Series(lr.coef_[0], index=[FEATURE_LABELS.get(f, f) for f in FEATURES]).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = ['#c0392b' if v < 0 else '#27ae60' for v in coefs]
    ax.barh(coefs.index, coefs.values, color=colors)
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel('Коэффициент логистической регрессии (признаки стандартизированы)')
    ax.set_title('Зелёные — «за» хорошее вино, красные — «против»')
    st.pyplot(fig)
    plt.close(fig)

    top_pos = coefs[coefs > 0].sort_values(ascending=False).head(3)
    top_neg = coefs[coefs < 0].sort_values().head(3)
    c1, c2 = st.columns(2)
    c1.markdown('**Сильнее всего голосуют ЗА хорошее вино:**\n'
                + '\n'.join(f'- {k}: **{v:+.2f}**' for k, v in top_pos.items()))
    c2.markdown('**Сильнее всего голосуют ПРОТИВ:**\n'
                + '\n'.join(f'- {k}: **{v:+.2f}**' for k, v in top_neg.items()))
    st.caption(
        'Как читать: признаки перед обучением стандартизированы, поэтому коэффициент показывает, '
        'насколько меняются логит-шансы «хорошего» вина при увеличении признака на одно стандартное '
        'отклонение (при прочих равных). Коэффициент +1 означает рост шансов примерно в e ≈ 2.7 раза. '
        'Осторожно с интерпретацией: признаки коррелируют (например, плотность сильно связана с '
        'алкоголем и сахаром), поэтому отдельный коэффициент — не «чистый» эффект признака.')
