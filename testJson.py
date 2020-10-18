import json

a_file = open("sample_file.json", "r")
json_object = json.load(a_file)
a_file.close()
print(json_object)

name = 'Chung'
json_object["d"] = 'toi la {}'.format(name)

a_file = open("sample_file.json", "w")
json.dump(json_object, a_file)
a_file.close()

a_file = open("sample_file.json", "r")
json_object = json.load(a_file)
a_file.close()
print(json_object)
