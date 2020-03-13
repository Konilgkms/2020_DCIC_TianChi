import pandas as pd
import os

train_dir = ""
test_dir = ""
testB_dir = ""

test = pd.DataFrame(columns=['渔船ID', 'x', 'y', '速度', '方向', 'time', 'start_x', 'start_y', 'end_x', 'end_y'])

for root, dirs, files in os.walk(test_dir):
    for file in files:
        filename = os.path.join(root, file)
        temp = pd.read_csv(filename)

        size = temp.shape[0]
        start_x = temp['x'][size-1]
        start_y = temp['y'][size-1]
        temp['start_x'] = start_x
        temp['start_y'] = start_y
        end_x = temp['x'][0]
        end_y = temp['y'][0]
        temp['end_x'] = end_x
        temp['end_y'] = end_y

        test = pd.concat([test, temp])
print(test.info())
test.to_csv("../data/test_origin_0221.csv", index=False)

testB = pd.DataFrame(columns=['渔船ID', 'x', 'y', '速度', '方向', 'time', 'start_x', 'start_y', 'end_x', 'end_y'])

for root, dirs, files in os.walk(testB_dir):
    for file in files:
        filename = os.path.join(root, file)
        temp = pd.read_csv(filename)

        size = temp.shape[0]
        start_x = temp['x'][size - 1]
        start_y = temp['y'][size - 1]
        temp['start_x'] = start_x
        temp['start_y'] = start_y
        end_x = temp['x'][0]
        end_y = temp['y'][0]
        temp['end_x'] = end_x
        temp['end_y'] = end_y

        testB = pd.concat([testB, temp])
print(testB.info())
testB.to_csv("../data/testB_origin_0221.csv", index=False)


train = pd.DataFrame(columns=['渔船ID', 'x', 'y', '速度', '方向', 'time', 'type', 'start_x', 'start_y', 'end_x', 'end_y'])

for root, dirs, files in os.walk(train_dir):
    for file in files:
        filename = os.path.join(root, file)
        temp = pd.read_csv(filename)

        size = temp.shape[0]
        start_x = temp['x'][size - 1]
        start_y = temp['y'][size - 1]
        temp['start_x'] = start_x
        temp['start_y'] = start_y
        end_x = temp['x'][0]
        end_y = temp['y'][0]
        temp['end_x'] = end_x
        temp['end_y'] = end_y

        train = pd.concat([train, temp])
print(train.info())



train.to_csv("../data/train_origin_0221.csv", index=False)