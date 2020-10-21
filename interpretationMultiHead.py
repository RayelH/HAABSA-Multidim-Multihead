import numpy as np

def compute_confidence(file, number_of_heads):
    a = np.loadtxt(file)
    a = np.squeeze(np.amax(a, axis=1))
    b = a.reshape(-1, int(np.size(a)/number_of_heads))
    result = np.mean(b, axis=1)
    return result

def positional_left_context(file, number_of_heads, test_size):
    context = np.loadtxt(file)
    max_boolean = []

    for row in context:
        count = 1
        for element in row:
            if count == 1 and element != 0.0:
                temp = []
            if element != 0.0:
                temp.append(element)

            count += 1

        if temp[-1] == max(temp):
            max_boolean.append(1)
        else:
            max_boolean.append(0)

    max_boolean = np.array(max_boolean)

    b = max_boolean.reshape(-1, int(np.size(max_boolean) / 3))
    c = np.sum(b, axis=1)
    result = (c/test_size)
    return result


def positional_right_context(file, number_of_heads, test_size):
    context = np.loadtxt(file)
    max_boolean = []

    for row in context:
        count = 1
        for element in row:
            if count == 1 and element != 0.0:
                temp = []
            if element != 0.0:
                temp.append(element)

            count += 1

        if temp[0] == max(temp):
            max_boolean.append(1)
        else:
            max_boolean.append(0)

    max_boolean = np.array(max_boolean)

    b = max_boolean.reshape(-1, int(np.size(max_boolean) / 3))
    c = np.sum(b, axis=1)
    result = (c / test_size)
    return result

