import requests
import json

weather_stack_api_key = "e6cc3f97e4fdf37ac060175b62c5141a"
weather_stack_base_url = "http://api.weatherstack.com/"

news_api_key = "5cd6d4b1c26343f1a3e92024d2e2a407"
news_api_base_url = "https://newsapi.org/v2/everything"

news_api_params = {
    "q": "agriculture",
    "language": "en",
    "sortBy": "relevancy",
    "pageSize": 100,
    "apiKey": news_api_key
}

jamaican_cities = [
    "Kingston",
    "Montego Bay",
    "Spanish Town",
    "Portmore",
    "Mandeville",
    "Ocho Rios",
    "May Pen",
    "Savanna-la-Mar",
    "Port Antonio",
    "Falmouth",
    "Negril",
    "Lucea",
    "Black River",
    "Morant Bay",
    "Linstead",
    "Old Harbour",
    "Bog Walk",
    "Brown's Town",
    "Annotto Bay",
    "Ewarton",
    "Seaforth",
    "Runaway Bay",
    "St. Ann's Bay",
    "Yallahs",
    "Santa Cruz"
]


def get_agriculture_news():
    data = requests.get(news_api_base_url, params=news_api_params).json()
    return data


def update_agriculture_news():
    data = get_agriculture_news()
    filename = "../agriculture_data.json"
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)


def format_agriculture_news():
    filename = '../agriculture_data.txt'
    json_filename = '../agriculture_data.json'
    filedata = []

    try:
        with open(json_filename, 'r') as file:
            filedata = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        filedata = []

    with open(filename, 'w') as file:
        sect_index = 0
        for article in filedata['articles']:
            sect_index += 1
            file.write(f'\nSection: {sect_index}\n\n')
            for key, value in article.items():
                if type(value) == str:
                    file.write(
                        f"{key}: {value.encode('UTF-8')}")  # while combating possible unicode error since some content from the response is written in markdown format which may use backslash and break the string
                elif type(value) == dict:
                    file.write(f'{key}:\n')
                    for k2, v2 in value.items():
                        file.write(f'{k2}: {v2}\n')
                elif type(value) == list:
                    file.write(f"{key}: ")
                    for i in value:
                        file.write(f"{i}, ")
                file.write("\n")
            file.write('-' * 100)

    with open(filename, 'r+') as file:
        cfile = file.read()
        file.seek(0)
        file.write(f"This Document represents the current publicly available news relating to agriculture \n\n" + cfile)


def get_current_weather(city="Kingston"):
    data = requests.get(f"{weather_stack_base_url}current?access_key={weather_stack_api_key}&query={city}").json()
    return data


def get_weather_forecast(city="Kingston"):
    data = requests.get(f"{weather_stack_base_url}forecast?access_key={weather_stack_api_key}&query={city}").json()
    return data


def update_current_weather():
    filename = '../current_weather_data.json'
    filedata = []

    try:
        with open(filename, 'r') as file:
            filedata = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        filedata = []

    for city in jamaican_cities:
        data = get_current_weather(city)
        filedata.append(data)

    with open(filename, 'w') as file:
        json.dump(filedata, file, indent=4)


def update_weather_forecast():
    filename = '../weather_forecast.json'
    filedata = []

    try:
        with open(filename, 'r') as file:
            filedata = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        filedata = []

    for city in jamaican_cities:
        data = get_weather_forecast(city)
        filedata.append(data)

    with open(filename, 'w') as file:
        json.dump(filedata, file, indent=4)


def format_current_weather_data():
    filename = '../current_weather_data.txt'
    json_filename = '../current_weather_data.json'
    filedata = []

    try:
        with open(json_filename, 'r') as file:
            filedata = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        filedata = []

    with open(filename, 'w') as file:
        sect_index = 0
        for data in filedata:
            sect_index += 1
            file.write(f"Section {sect_index}\n")
            for key, value in data.items():
                if type(value) == str:
                    file.write(f"{key}: {value}")
                elif type(value) == dict:
                    # print(key)
                    file.write(key + "\n")
                    for k2, v2 in value.items():
                        file.write(f"{k2}: {v2}\n")
                elif type(value) == list:
                    file.write(f"{key}: ")
                    for i in value:
                        file.write(f"{i}, ")
                file.write("\n")
            file.write('-' * 100)

    with open(filename, 'r+') as file:
        cfile = file.read()
        file.seek(0)
        file.write(f"This Document Represents the current weather data of the following cities in Jamaica\n\n" + cfile)


def format_weather_forecast():
    filename = '../weather_forecast.txt'
    json_filename = '../weather_forecast.json'
    filedata = []

    try:
        with open(json_filename, 'r') as file:
            filedata = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        filedata = []

    with open(filename, 'w') as file:
        sect_index = 0
        for data in filedata:
            sect_index += 1
            file.write(f"Section {sect_index}\n")
            for key, value in data.items():

                if type(value) == str:
                    file.write(f"{key}: {value}")
                elif type(value) == dict:
                    # print(key)
                    file.write(key + "\n")
                    for k2, v2 in value.items():
                        file.write(f"{k2}: {v2}\n")
                elif type(value) == list:
                    file.write(f"{key}: ")
                    for i in value:
                        file.write(f"{i}, ")
                file.write("\n")
            file.write('-' * 100 + '\n\n')

    with open(filename, 'r+') as file:
        cfile = file.read()
        file.seek(0)
        file.write(
            f"This Document Represents the current weather forecast data of the following cities in Jamaica\n\n" + cfile)


def main():
    update_current_weather() #pull current weather data for each city in Jamaica and store in json
    update_weather_forecast() #pull current weather forecst data for each city in Jamaica and store in json
    format_current_weather_data() #reformats and stores corresponding json data in raw text form
    format_weather_forecast() #reformats and stores corresponding json data in raw text form

    update_agriculture_news() #pull current publicly available news relating to agriculture and store in json
    format_agriculture_news()  # reformats and stores corresponding json data in raw text form


if __name__ == "__main__":
    main()