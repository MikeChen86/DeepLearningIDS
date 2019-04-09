import csv

if __name__ == '__main__':
    with open('Dataset/Multiclass/data.csv', 'w', newline='') as csv_file:
        file_path = 'Dataset/Multiclass/data1.csv'
        header = list(csv.reader(open(file_path, newline='')))[0]
        writer = csv.writer(csv_file)
        writer.writerow(header)

        for each in range(1, 16):
            file_path = 'Dataset/Multiclass/data{}.csv'.format(each)
            rows = list(csv.reader(open(file_path, newline='')))[1:]
            for row in rows:
                writer.writerow(row)
