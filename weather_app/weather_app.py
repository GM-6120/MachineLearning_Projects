import streamlit as st
import requests

def get_weather(city):
    api_key = "5b7c495b5c1d495c8e0194547251606"
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            "city": data['location']['name'],
            "country": data['location']['country'],
            "temp": data['current']['temp_c'],
            "condition": data['current']['condition']['text'],
            "humidity": data['current']['humidity'],
            "wind": data['current']['wind_kph']
        }
    else:
        return None

st.title("🌦️ Simple Weather App")

city = st.text_input("Enter city name")

if city:
    weather = get_weather(city)
    if weather:
        st.subheader(f"Weather in {weather['city']}, {weather['country']}")
        st.write(f"**Temperature:** {weather['temp']} °C")
        st.write(f"**Condition:** {weather['condition']}")
        st.write(f"**Humidity:** {weather['humidity']}%")
        st.write(f"**Wind Speed:** {weather['wind']} kph")
    else:
        st.error("⚠️ Could not fetch weather data. Check city name or API.")

