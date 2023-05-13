with open('dataset.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        dataset.append(row)