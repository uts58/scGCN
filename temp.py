import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from ultralytics import YOLO


def get_count(results_):
    all_data = {}

    for result in results_:
        detected = result.names[result.probs.top1]
        image_ = result.path.split('/')[-1].split('.')[0]
        all_data[image_] = detected

    return all_data


ground_truth = {}
for i, rows in pd.read_csv('/content/ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv').iterrows():
    temp = rows.to_dict()
    image = temp.pop('image')
    for items in temp:
        if temp[items] == 1:
            ground_truth[image] = items

models = ["train", "train2", "train3", "train4", "train5"]
image_folder = '/mmfs1/scratch/utsha.saha/lp/data/test/'

for x in models:
    path_ = f"/mmfs1/scratch/utsha.saha/lp/runs/classify/{x}/weights/best.pt"
    model = YOLO(path_)  # load a pretrained model (recommended for training)
    model.to('cuda')

    results = model(image_folder, save=True)
    pred_truth = get_count(results)

    y_true, y_pred = [], []

    for items in pred_truth:
        y_true.append(ground_truth[items])
        y_pred.append(pred_truth[items])

    report = classification_report(y_true, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(f"{x}.csv")

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['AKIEC', "BCC", "BKL", "DF", "MEL", "NV", "VASC"])
    disp.plot()
    plt.savefig(f"{x}_plot.png")

    print("=============================")
    print("=============================")
