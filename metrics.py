def accuracy(outputs, labels):
    sumacc = 0
    for value, label in zip(outputs, labels):
        sumacc += int(round(value) == label)
    return sumacc/len(labels)
