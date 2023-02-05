from feature_extractor import ExtractFeatures
import time

if __name__ == '__main__':
    start = time.time()
    print(ExtractFeatures().extract_features('https://www.tanbh.com/?email_id=20230131042622.478f17178c2796ea&utm_medium=email'))
    end = time.time()
    print(end - start)
