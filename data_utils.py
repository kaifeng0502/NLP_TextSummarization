def read_samples(filename):
    samples = []
    with open(filename, 'r', encoding='utf8') as file:
        for line in file:
            samples.append(line.strip())
    return samples


def write_samples(samples, file_path, opt='w'):
    with open(file_path, opt, encoding='utf8') as file:
        for line in samples:
            file.write(line)
            file.write('\n')


def partition(samples):
    train, dev, test = [], [], []
    count = 0
    for sample in samples:
        count += 1
        if count % 1000 == 0:
            print(count)
        if count <= 1000:
            test.append(sample)
        elif count <= 6000:
            dev.append(sample)
        else:
            train.append(sample)
    print('train: ', len(train))

    write_samples(train, 'output/train.txt')
    write_samples(dev, 'output/dev.txt')
    write_samples(test, 'output/test.txt')


def isChinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False
