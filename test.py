from sklearn.ensemble import RandomForestClassifier
import numpy as np


class Domain:
    def __init__(self, _name, _label):
        self.name = _name
        self.label = _label
        self.length = len(self.name)
        self.number = sum(d.isdigit() for d in self.name)

    def returnName(self):
        return self.name

    def returnData(self):
        return [self.length, self.number]

    def returnLabel(self):
        if self.label == "dga":
            return 1
        else:
            return 0

domainlist = [] 

def initData(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            tokens = line.split(",")
            name = tokens[0]
            label = tokens[1]
            domainlist.append(Domain(name, label))

def main():
    initData("train.txt")
    featurematrix = []
    labellist = []
    for domain in domainlist:
        featurematrix.append(domain.returnData())
        labellist.append(domain.returnLabel())

    clf = RandomForestClassifier(random_state=0)
    clf.fit(featurematrix, labellist)

    with open("test.txt") as f:
        with open("result.txt", 'w') as m:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                length = len(line)
                number = sum(d.isdigit() for d in line)
                if (int(clf.predict([[length, number]])) == 1):
                    content = line + "," + "dga\n"
                    m.write(content)
                else:
                    content = line + "," + "notdga\n"
                    m.write(content)


if __name__ == '__main__':
    main()
