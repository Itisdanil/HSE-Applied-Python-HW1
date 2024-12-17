import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from sklearn.linear_model import LinearRegression


def process_city(city_data):
    """
    Обрабатывает данные температуры по городу и возвращает 
    статистики.
    """
    city_data['rolling_mean'] = city_data['temperature'].rolling(window=30, min_periods=1).mean()
    city_data['rolling_std'] = city_data['temperature'].rolling(window=30, min_periods=1).std()

    city_data['anomaly'] = (
        (city_data['temperature'] > city_data['rolling_mean'] + 2 * city_data['rolling_std']) |
        (city_data['temperature'] < city_data['rolling_mean'] - 2 * city_data['rolling_std'])
    )
    anomalies = city_data[city_data['anomaly']]

    seasonal_profile = city_data.groupby('season')['temperature'].agg(['mean', 'std']).reset_index()
    seasonal_profile[['mean', 'std']] = seasonal_profile[['mean', 'std']].apply(lambda x: round(x, 2))

    X = np.arange(len(city_data)).reshape(-1, 1)
    y = city_data['temperature'].values
    model = LinearRegression()
    model.fit(X, y)
    trend = model.coef_[0]

    avg_temp = round(city_data['temperature'].mean(), 2)
    min_temp = round(city_data['temperature'].min(), 2)
    max_temp = round(city_data['temperature'].max(), 2)

    return {
        'city': city_data['city'].iloc[0],
        'avg_temp': avg_temp,
        'min_temp': min_temp,
        'max_temp': max_temp,
        'seasonal_profile': seasonal_profile,
        'trend': trend,
        'anomalies': anomalies
    }


def get_current_weather(api_key, city):
    """
    Получает текущую погоду в городе через API в OpenWeatherMAp.
    """
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['main']['temp']
    else:
        return None


# ------------------------ Основное тело Streamlit приложения ------------------------ 
st.title("Анализ временных рядов температуры")

uploaded_file = st.file_uploader("Загрузите файл с историческими данными", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        required_columns = ['city', 'timestamp', 'temperature', 'season']
        if not all(column in df.columns for column in required_columns):
            st.error(f"Ошибка: файл должен содержать следующие столбцы: {', '.join(required_columns)}.")
        else:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except Exception as e:
                st.error(f"Ошибка: невозможно преобразовать столбец 'timestamp' в формат даты. Убедитесь, что формат корректен.")

            cities = df['city'].unique()
            selected_city = st.selectbox("Выберите город", cities)

            city_data = df[df['city'] == selected_city]

            api_key = st.text_input("Введите API-ключ OpenWeatherMap")

            result = process_city(city_data)

            st.subheader("Описательная статистика по историческим данным")
            st.write(f"Средняя температура: {result['avg_temp']}")
            st.write(f"Минимальная температура: {result['min_temp']}")
            st.write(f"Максимальная температура: {result['max_temp']}")

            st.subheader("Временной ряд температур с аномалиями и трендом")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=city_data['timestamp'], y=city_data['temperature'], mode='lines', name='Температура'))
            fig.add_trace(go.Scatter(x=city_data['timestamp'], y=city_data['rolling_mean'], mode='lines', name='Скользящее среднее'))
            fig.add_trace(go.Scatter(x=result['anomalies']['timestamp'], y=result['anomalies']['temperature'], mode='markers', name='Аномалии', marker=dict(color='red')))

            trend_line = result['trend'] * np.arange(len(city_data)) + result['avg_temp']
            fig.add_trace(go.Scatter(x=city_data['timestamp'], y=trend_line, mode='lines', name='Тренд', line=dict(color='green', dash='dash')))

            fig.update_layout(width=1000, height=600, title="Температура, скользящее среднее, аномалии и тренд", xaxis_title="Дата", yaxis_title="Температура")
            st.plotly_chart(fig)

            st.subheader("Сезонные профили")
            st.write(result['seasonal_profile'])

            if api_key:
                current_temp = get_current_weather(api_key, selected_city)
                if current_temp is not None:
                    st.subheader("Текущая температура")
                    st.write(f"Текущая температура: {current_temp}°C")

                    current_season = city_data['season'].iloc[-1]  # делаю допущение, что последняя строка совпадает с текущим сезоном
                    season_profile = result['seasonal_profile'][result['seasonal_profile']['season'] == current_season]
                    mean_temp = season_profile['mean'].values[0]
                    std_temp = season_profile['std'].values[0]

                    if (current_temp > mean_temp + 2 * std_temp) or (current_temp < mean_temp - 2 * std_temp):
                        st.write("Текущая температура аномальна для сезона.")
                    else:
                        st.write("Текущая температура нормальна для сезона.")
                else:
                    st.error("Invalid API key. Please see https://openweathermap.org/faq#error401 for more info.")
            else:
                st.warning("API-ключ не введен. Текущая погода не отображается.")

    except Exception as e:
        st.error(f"Ошибка при обработке файла: {e}")
