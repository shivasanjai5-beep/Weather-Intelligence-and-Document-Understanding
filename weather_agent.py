"""Agent 1: weather. Uses OpenWeatherMap first, falls back to Open-Meteo (free, no key)."""
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.express as px
import requests
from dotenv import load_dotenv

from llm import ask_llm

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")
OWM = "https://api.openweathermap.org"

# Open-Meteo gives weather as numbers; this turns them into words
WEATHER_CODES = {0: "Clear", 1: "Mostly clear", 2: "Partly cloudy", 3: "Cloudy",
                 45: "Fog", 48: "Fog", 51: "Drizzle", 53: "Drizzle", 55: "Drizzle",
                 61: "Rain", 63: "Rain", 65: "Heavy rain", 71: "Snow", 73: "Snow",
                 75: "Heavy snow", 80: "Showers", 81: "Showers", 82: "Heavy showers",
                 95: "Thunderstorm", 96: "Thunderstorm", 99: "Thunderstorm"}


class WeatherError(Exception):
    pass


def _get(url: str, params: dict, use_key: bool = False) -> dict:
    """One place for all API calls + error handling."""
    if use_key:
        if not API_KEY:
            raise WeatherError("OPENWEATHER_API_KEY is missing.")
        params["appid"] = API_KEY
    try:
        r = requests.get(url, params=params, timeout=15)
    except requests.RequestException as e:
        raise WeatherError(f"Network error: {e}")
    if r.status_code == 401:
        raise WeatherError("API key rejected or plan does not include this data.")
    if r.status_code != 200:
        raise WeatherError(f"Weather API error {r.status_code}: {r.text[:100]}")
    return r.json()


# =============== Source 1: OpenWeatherMap ===============
def owm_geocode(city):
    data = _get(f"{OWM}/geo/1.0/direct", {"q": city, "limit": 1}, use_key=True)
    if not data:
        raise WeatherError(f"City '{city}' not found.")
    return data[0]["lat"], data[0]["lon"], data[0]["name"]


def fetch_openweather(city, when):
    lat, lon, name = owm_geocode(city)
    if when == "current":
        d = _get(f"{OWM}/data/2.5/weather", {"lat": lat, "lon": lon, "units": "metric"}, True)
        df = pd.DataFrame([{
            "Condition": d["weather"][0]["description"],
            "Temp (°C)": d["main"]["temp"],
            "Feels like (°C)": d["main"]["feels_like"],
            "Humidity (%)": d["main"]["humidity"],
            "Wind (m/s)": d["wind"]["speed"],
        }])
    elif when in ("tomorrow", "forecast"):
        d = _get(f"{OWM}/data/2.5/forecast", {"lat": lat, "lon": lon, "units": "metric"}, True)
        tz = d["city"]["timezone"]
        rows = []
        for i in d["list"]:
            t = datetime.fromtimestamp(i["dt"] + tz, tz=timezone.utc)
            rows.append({"Date": t.date(), "Time": t.strftime("%H:%M"),
                         "Temp (°C)": i["main"]["temp"],
                         "Humidity (%)": i["main"]["humidity"],
                         "Rain chance (%)": round(i.get("pop", 0) * 100),
                         "Condition": i["weather"][0]["main"]})
        df = pd.DataFrame(rows)
    elif when in ("yesterday", "last_10_days"):
        n = 1 if when == "yesterday" else 10
        today = datetime.now(timezone.utc).date()
        rows = []
        for i in range(n, 0, -1):
            day = today - timedelta(days=i)
            d = _get(f"{OWM}/data/3.0/onecall/day_summary",
                     {"lat": lat, "lon": lon, "date": day.isoformat(), "units": "metric"}, True)
            rows.append({"Date": day,
                         "Min Temp (°C)": d["temperature"]["min"],
                         "Max Temp (°C)": d["temperature"]["max"],
                         "Humidity (%)": d["humidity"]["afternoon"],
                         "Rain (mm)": d.get("precipitation", {}).get("total", 0)})
        df = pd.DataFrame(rows)
    else:
        raise WeatherError("I could not understand the time period.")
    return df, name


# =============== Source 2: Open-Meteo (free, no key) ===============
def fetch_openmeteo(city, when):
    g = _get("https://geocoding-api.open-meteo.com/v1/search", {"name": city, "count": 1})
    if not g.get("results"):
        raise WeatherError(f"City '{city}' not found.")
    lat, lon, name = g["results"][0]["latitude"], g["results"][0]["longitude"], g["results"][0]["name"]
    url = "https://api.open-meteo.com/v1/forecast"
    base = {"latitude": lat, "longitude": lon, "timezone": "auto"}

    if when == "current":
        d = _get(url, {**base, "wind_speed_unit": "ms", "current":
                       "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,weather_code"})
        c = d["current"]
        df = pd.DataFrame([{
            "Condition": WEATHER_CODES.get(c["weather_code"], "Unknown"),
            "Temp (°C)": c["temperature_2m"],
            "Feels like (°C)": c["apparent_temperature"],
            "Humidity (%)": c["relative_humidity_2m"],
            "Wind (m/s)": c["wind_speed_10m"],
        }])
    elif when in ("tomorrow", "forecast"):
        d = _get(url, {**base, "forecast_days": 5, "hourly":
                       "temperature_2m,relative_humidity_2m,precipitation_probability,weather_code"})
        h = d["hourly"]
        rows = []
        for i in range(0, len(h["time"]), 3):  # every 3 hours
            day, time = h["time"][i].split("T")
            rows.append({"Date": datetime.fromisoformat(day).date(), "Time": time,
                         "Temp (°C)": h["temperature_2m"][i],
                         "Humidity (%)": h["relative_humidity_2m"][i],
                         "Rain chance (%)": h["precipitation_probability"][i],
                         "Condition": WEATHER_CODES.get(h["weather_code"][i], "Unknown")})
        df = pd.DataFrame(rows)
    elif when in ("yesterday", "last_10_days"):
        n = 1 if when == "yesterday" else 10
        d = _get(url, {**base, "past_days": n, "forecast_days": 1, "daily":
                       "temperature_2m_max,temperature_2m_min,precipitation_sum"})
        dd = d["daily"]
        rows = [{"Date": datetime.fromisoformat(dd["time"][i]).date(),
                 "Min Temp (°C)": dd["temperature_2m_min"][i],
                 "Max Temp (°C)": dd["temperature_2m_max"][i],
                 "Rain (mm)": dd["precipitation_sum"][i]} for i in range(n)]  # skip today
        df = pd.DataFrame(rows)
    else:
        raise WeatherError("I could not understand the time period.")
    return df, name


# =============== Main entry ===============
def run(question: str, city: str, when: str) -> dict:
    """Returns {"text": str, "tables": [df], "figs": [plotly figure]}"""
    result = {"text": "", "tables": [], "figs": []}
    try:
        if not city:
            raise WeatherError("Please mention a city, e.g. 'weather in Chennai today'.")

        # try OpenWeatherMap first, then the free fallback
        source = "OpenWeatherMap"
        try:
            if not API_KEY:
                raise WeatherError("no key")
            df, name = fetch_openweather(city, when)
        except WeatherError:
            source = "Open-Meteo (free, no key)"
            df, name = fetch_openmeteo(city, when)

        if when == "tomorrow":
            dates = sorted(df["Date"].unique())
            df = df[df["Date"] == dates[1]] if len(dates) > 1 else df

        result["tables"].append(df)

        # pick charts that fit the question
        if when == "current":
            row = df.iloc[0]
            chart = pd.DataFrame({"Measure": ["Temp", "Feels like"],
                                  "°C": [row["Temp (°C)"], row["Feels like (°C)"]]})
            result["figs"].append(px.bar(chart, x="Measure", y="°C", title=f"Temperature now in {name}"))
        elif when in ("tomorrow", "forecast"):
            chart_df = df.copy()
            chart_df["Label"] = chart_df["Date"].astype(str) + " " + chart_df["Time"]
            result["figs"].append(px.line(chart_df, x="Label", y="Temp (°C)", markers=True,
                                          title=f"Forecast temperature - {name}"))
            result["figs"].append(px.pie(chart_df, names="Condition", title="Share of conditions"))
        elif when == "last_10_days":
            result["figs"].append(px.line(df, x="Date", y=["Min Temp (°C)", "Max Temp (°C)"],
                                          markers=True, title=f"Temperature trend - {name}"))
            result["figs"].append(px.bar(df, x="Date", y="Rain (mm)", title="Daily rain"))

        # LLM writes the answer using ONLY the real data (no made-up numbers)
        data_text = df.to_string(index=False)
        answer = ask_llm(
            f"Answer the question using ONLY this weather data for {name}. "
            f"Do not invent numbers. Keep it short.\n\nData:\n{data_text}\n\nQuestion: {question}"
        )
        result["text"] = f"{answer}\n\n*Source: {source}*"
    except WeatherError as e:
        result["text"] = f"⚠️ {e}"
    return result
