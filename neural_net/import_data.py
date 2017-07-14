import numpy

def read_file(file_name):
    data_file = open(file_name, 'r')
    data_items = data_file.readlines()
    data_file.close()
    return [data_item.rstrip() for data_item in data_items]

def read_csv_file(file_name):
    data_items = read_file(file_name)
    return [data_item.split(',') for data_item in data_items]

def read_mnist_file(file_name):
    data_items = read_csv_file(file_name)
    number_images = [numpy.asfarray(x[1:]).reshape((28, 28)) * 0.99 / 255.0 + 0.0 for x in data_items]
    image_number = [int(x[0]) for x in data_items]
    return (image_number, number_images)
