# Author: Daniel Bis (dmb16f)
import csv
import math
import sys


def load_data(filename):
    """
    Read in the data from filename

    :param filename:
    :return: list of lines in the text with normalized length in format [["label index:value index:value ..."]]
    """

    list_of_rows = []
    f = open(filename, 'r')
    attr_max = 0
    for line in f:
        row = [x.strip() for x in line.split(' ')]
        if int(row[1][0]) != 1:
            for i in range(1, int(row[1][0])):
                row.insert(i, str(i)+":0")
        for i in range(1, len(row) - 1):
            if int(row[i][0]) +1 != int(row[i+1][0]):  # fill out the empty spots
                for j in range(i+1, int(row[i+1][0])):
                    row.insert(j, str(j) + ":0")
            if int(row[len(row)-1][0]) > attr_max:
                attr_max = int(row[len(row)-1][0])
        list_of_rows.append(row)

    attr_max +=1  # adjust for label
    for row in list_of_rows:
        if (len(row)) > attr_max:
            attr_max = len(row)
        elif len(row) < attr_max:
            for i in range(len(row), attr_max):
                row.append(str(i) + ":0")
        else:
            pass

    return list_of_rows


def convert_to_attr(list_of_rows):

    num_version = []

    for row in list_of_rows:
        temp_r = []
        temp_r.append(int(row[0]))
        for i in range(1, len(row)):
            temp_r.append(int(row[i][2]))
        num_version.append(temp_r)

    return num_version


def split_by_class(dataset):
    splitted = {}
    for i in range(len(dataset)):
        row = dataset[i]
        if row[0] not in splitted:
            splitted[row[0]] = []
        splitted[row[0]].append(row)

    return splitted


def mean(nums):
    return sum(nums)/int(len(nums))


def st_dev(nums):
    average = mean(nums)
    variance = sum([pow(x-average, 2) for x in nums])/int(len(nums) - 1)
    return math.sqrt(variance)


def get_stats(dataset):
    """
    get_stats calculates the mean and standard deviation for each attribute
    I use zip() function to group the attributes together
    I delete stats[0] since these are labels

    :param dataset: matrix in form of [[label, attr1, attr2 ..]]
    :return: list of tuples in form [(mean, standard deviation)]
    """
    stats = [(mean(attr), st_dev(attr)) for attr in zip(*dataset)]
    del(stats[0])
    return stats


def stats_by_class(dataset):

    """
    Calculates mean and standard deviation of each attribute.
    Segregates the results by class.
    :param dataset:  [label, attr1, attr2, ...]
    :return:   {label: [(mean, st_dev) ...], label2: [(mean, st_dev) ...]}
    """

    splitted = split_by_class(dataset)
    stats = {}
    for c, values in splitted.items():
        stats[c] = get_stats(values)
    return stats


def gauss(x, mean, st_dev):
    """
            Gauss = math.exp(-(math.pow(x-mean, 2)/(2*math.pow(st_dev, 2))))
            Using = (1 / (math.sqrt(2*math.pi)* st_dev)) * ex
            :param x = attribute (int)
            :param mean
            :param st_dev

            :returns (float) gaussian probability

    """

    ex = math.exp(-(math.pow(x-mean, 2)/(2*math.pow(st_dev, 2))))
    g = (1 / (math.sqrt(2*math.pi)* st_dev)) * ex
    return g


def combained_probability(stats, row):
    """
    row[i+1] used in the loop to skip the label

    :param stats: {label: [(mean, st_dev) ...], label2: [(mean, st_dev) ...]}
    :param row: a row in data matrix [label attr1 attr2 ...]
    :return: probability of a row belonging to a class {label1: probability1, label2: probability2}
    """
    probabilities = {}
    for c, c_stats in stats.items():
        probabilities[c] = 1
        for i in range(0,len(c_stats)):
            mean, st_dev = c_stats[i]
            x = row[i+1]
            probabilities[c] *= gauss(x, mean, st_dev)
    return probabilities


def do_prediction(stats, row):
    """

    :param stats: {label: [(mean, st_dev) ...], label2: [(mean, st_dev) ...]}
    :param row: a row in data matrix [label attr1 attr2 ...]
    :return: predicted label
    """
    probs = combained_probability(stats, row)
    p_label = None
    p_prob = -1

    for c, prob in probs.items():
        if p_label is None or prob > p_prob:
            p_label = c
            p_prob = prob
    return p_label


def get_predictions(stats, test_data):
    """

    :param stats: {label: [(mean, st_dev) ...], label2: [(mean, st_dev) ...]}
    :param test_data: [[label, attr1, attr2, ...] ... ]
    :return: list of predicted labels [l1, l2, l3 ...]
    """
    predicted = []

    for i in range(len(test_data)):
        temp_pred = do_prediction(stats, test_data[i])
        predicted.append(temp_pred)

    return predicted


def summary(test_data, predictions):
    """
    1. A true positive test result is one that detects the condition when the
    condition is present.
    2. A true negative test result is one that does not detect the condition when
    the condition is absent.
    3. A false positive test result is one that detects the condition when the
    condition is absent.
    4. A false negative test result is one that does not detect the condition when
    the condition is present
    :param test_data: [[label, attr1, attr2, ...] ... ]
    :param predictions: labels [l1, l2, l3 ...]
    :return: true_positive, true_negative, false_positive, false_negative type=int
    """
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for i in range(len(test_data)):
        if predictions[i] == test_data[i][0]:
            if test_data[i][0] == 1:
                true_positive += 1
            else:
                true_negative += 1
        else:
            if test_data[i][0] == -1:
                false_positive += 1
            else:
                false_negative += 1
    return true_positive, true_negative, false_positive, false_negative


def run():

    # get filenames from command line
    training_file = sys.argv[1]
    test_file = sys.argv[2]

    # load data
    training_set_raw = load_data(training_file)
    test_set_raw = load_data(test_file)

    training_set = convert_to_attr(training_set_raw)
    test_set = convert_to_attr(test_set_raw)

    stats = stats_by_class(training_set)

    predictions_test = get_predictions(stats, test_set)
    predictions_train = get_predictions(stats, training_set)
    true_positive_test, true_negative_test, false_positive_test, false_negative_test = summary(test_set, predictions_test)
    true_positive_train, true_negative_train, false_positive_train, false_negative_train = summary(training_set, predictions_train)

    print(true_positive_train, true_negative_train, false_positive_train, false_negative_train)
    print(true_positive_test, true_negative_test, false_positive_test, false_negative_test)


#  run the model
run()
