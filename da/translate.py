from googletrans import Translator
import time
import jieba
from data_utils import read_samples, write_samples
translator = Translator()

res = []
samples = read_samples('output/samples.txt')

count = 0
for line in samples:
    count += 1
    if count % 100 == 0:
        print(count)
        write_samples(res, 'output/translated.txt', 'a')
        res = []
    source = str(line)
    try:
        translation = translator.translate(source, dest='ja')
        time.sleep(1)

        translation = translator.translate(translation.text, dest='zh-cn')
        res.append(' '.join(list(jieba.cut(translation.text))))

    except:
        continue
    time.sleep(1)
