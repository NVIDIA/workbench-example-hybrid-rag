import requests
import json

#this should send the api a get request then saves the result data to the anime_data.json file
def anime_data_pull(size):

	#size is an integer, the last value for size was 100. Size indicates how many entities to load up
	url = "https://anime-db.p.rapidapi.com/anime"

	querystring = {"page": "1", "size": size, "sortBy": "ranking", "sortOrder": "asc"}

	headers = {
		"x-rapidapi-key": "135634c5bcmsh9ebfb79c20e98f8p15bdf9jsn7fb31eef00ef",
		"x-rapidapi-host": "anime-db.p.rapidapi.com"
	}

	response = requests.get(url, headers=headers, params=querystring)

	if response.status_code == 200: #if fetch was successful
		data = response.json()
		with open('data_source/anime_data/anime_data.json', 'w') as json_file:
			json.dump(data, json_file, indent=4)  # indent=4 is used for pretty-printing the JSON
		print("JSON data saved to 'data.json'")
	else :
		print(f"Request failed with status code {response.status_code}")


#this will format the anime_data.json file into raw text and put it in the anime_list file
def update_anime_list():

	with open('data_source/anime_data/anime_data.json', 'r', encoding='utf-8') as json_file:
		data = json.load(json_file)
		data = data['data']

		with open('data_source/anime_data/anime_list.txt', 'w', encoding='utf-8') as file:
			for anime in data:
				for key, value in anime.items():
					if type(value) != list:
						file.write(f"{key}: {value}")
					else:
						file.write(f"{key}: ")
						for i in value:
							file.write(f"{i}, ")
					file.write("\n")
				file.write('-' * 100)

