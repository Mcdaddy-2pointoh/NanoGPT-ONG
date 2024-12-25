from data_processing.loaders import data_size_cal

path = "./data_archive/arrays"
data_size = data_size_cal(path = path)
formatted_number = "{:,}".format(data_size)
print(formatted_number)
