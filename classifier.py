import sklearn
import pandas
import numpy
import torch
import nltk
import sys
import re

# choose the hyperparameters

num_runs = 10
test_size = 0.25
num_epochs = 400
batch_size = 60
learn_rate = 0.0004
momentum = 0.8
dampening = 0.4
weight_decay = 0.2

# maybe gpu acceleration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# same setup to the baseline below ...

html_pattern = re.compile(r"<.*?>")

def remove_html(text):
    return html_pattern.sub("", text)

emoji_pattern = re.compile("["
    u"\U0001F600-\U0001F64F" # emoticons
    u"\U0001F300-\U0001F5FF" # symbols & pictographs
    u"\U0001F680-\U0001F6FF" # transport & map symbols
    u"\U0001F1E0-\U0001F1FF" # flags
    u"\U00002702-\U000027B0" # dingbats
    u"\U000024C2-\U0001F251" # enclosed characters
    "]+", flags = re.UNICODE)

def remove_emoji(text):
    return emoji_pattern.sub("", text)

nltk.download("stopwords")
stop_words = nltk.corpus.stopwords.words("english")
stop_words += ["..."]

def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stop_words])

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

def preprocess_dataset(in_path, out_path):
    data = pandas.read_csv(in_path).sample(frac = 1)
    data["Title+Body"] = data.apply(lambda row: row["Title"] + ". " + row["Body"] if pandas.notna(row["Body"]) else row["Title"], axis = 1)
    data = data.rename(columns = {
        "Unnamed: 0": "id",
        "class": "sentiment",
        "Title+Body": "text"
    })
    data["text"] = data["text"].apply(remove_html).apply(remove_emoji).apply(remove_stopwords).apply(clean_str)
    data.to_csv(out_path, index=False, columns=["id", "sentiment", "text"])

# new approach ahead

class WBCEWithLogitsLoss(torch.nn.Module):
    def __init__(self, w_p, w_n):
        super().__init__()
        self.w_p = w_p
        self.w_n = w_n

    def forward(self, logits, labels, epsilon = 1e-7):
        y = torch.sigmoid(logits.squeeze())
        positive = -1 * torch.mean(self.w_p * labels * torch.log(y + epsilon))
        negative = -1 * torch.mean(self.w_n * (1 - labels) * torch.log((1 - y) + epsilon))
        loss = positive + negative
        return loss

def train_model(dataset_path):
    # load the processed dataset and create sets
    data = pandas.read_csv(dataset_path).fillna("")
    indices = numpy.arange(data.shape[0])
    train_index, test_index = sklearn.model_selection.train_test_split(indices, test_size = test_size)
    train_text = data["text"].iloc[train_index]
    test_text = data["text"].iloc[test_index]
    y_train = torch.Tensor(data["sentiment"].iloc[train_index].to_numpy()).to(device)
    y_test = torch.Tensor(data["sentiment"].iloc[test_index].to_numpy()).to(device)
    # vectorizer
    tfidf = sklearn.feature_extraction.text.TfidfVectorizer(ngram_range = (1, 2), max_features = 1000)
    x_train = torch.Tensor(tfidf.fit_transform(train_text).toarray()).to(device)
    x_test = torch.Tensor(tfidf.transform(test_text).toarray()).to(device)
    train_set = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
    # the model
    model = torch.nn.Sequential(
        torch.nn.Linear(1000, 250),
        torch.nn.Sigmoid(),
        torch.nn.LayerNorm(250),
        torch.nn.Linear(250, 250),
        torch.nn.Sigmoid(),
        torch.nn.LayerNorm(250),
        torch.nn.Linear(250, 50),
        torch.nn.Sigmoid(),
        torch.nn.LayerNorm(50),
        torch.nn.Linear(50, 10),
        torch.nn.Sigmoid(),
        torch.nn.LayerNorm(10),
        torch.nn.Linear(10, 1),
    ).to(device)
    weight = torch.sum(y_train) / len(y_train)
    loss_func = WBCEWithLogitsLoss(1 - weight, weight)
    optimizer = torch.optim.SGD(model.parameters(), lr = learn_rate, momentum = momentum, dampening = dampening, weight_decay = weight_decay)
    # training
    size = len(train_set)
    num_batches = len(train_loader)
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0
        for batch, (x, y) in enumerate(train_loader):
            prediction = model(x)
            loss = loss_func(prediction, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
    # testing
    with torch.no_grad():
        model.eval()
        y_prediction = model(x_test).squeeze()
        # scoring
        y_prediction = y_prediction.cpu().numpy() > 0
        y_test = y_test.cpu().numpy() > 0
        accuracy = sklearn.metrics.accuracy_score(y_test, y_prediction)
        precision = sklearn.metrics.precision_score(y_test, y_prediction, average = "macro", zero_division = 0)
        recall = sklearn.metrics.recall_score(y_test, y_prediction, average = "macro", zero_division = 0)
        f1 = sklearn.metrics.f1_score(y_test, y_prediction, average = "macro", zero_division = 0)
        roc_auc = sklearn.metrics.roc_auc_score(y_test, y_prediction)
        return model, tfidf, accuracy, precision, recall, f1, roc_auc

def save_model(path, model, tfidf):
    torch.save((model, tfidf), path)

def load_model(path):
    (model, tfidf) = torch.load(path, device, weights_only = False)
    return model, tfidf

if __name__ == "__main__":
    if len(sys.argv) < 2:
        project = input("project? ")
    else:
        project = sys.argv[1]
        if len(sys.argv) > 2:
            try:
                num_runs = int(sys.argv[2])
            except:
                print(f"expected a number for number of runs (using {num_runs} instead)")
    print(f"preprocessing dataset (project: {project})")
    preprocess_dataset(f"datasets/{project}.csv", f"work/{project}-processed.csv")
    accuracy, precision, recall, f1, roc_auc = 0, 0, 0, 0, 0
    best_accuracy, best_precision, best_recall, best_f1, best_roc_auc = 0, 0, 0, 0, 0
    best_model, best_tfidf = None, None
    for run in range(num_runs):
        print(f"run {run + 1} of {num_runs}")
        run_model, run_tfidf, run_accuracy, run_precision, run_recall, run_f1, run_roc_auc = train_model(f"work/{project}-processed.csv")
        accuracy += run_accuracy
        precision += run_precision
        recall += run_recall
        f1 += run_f1
        roc_auc += run_roc_auc
        # saving the overall best model for evaluator.py
        if run_accuracy >= best_accuracy and run_precision >= best_precision and run_recall >= best_recall and run_f1 >= best_f1 and run_roc_auc >= best_roc_auc:
            best_model = run_model
            best_tfidf = run_tfidf
            best_accuracy = run_accuracy
            best_precision = run_precision
            best_recall = run_recall
            best_f1 = run_f1
            best_roc_auc = run_roc_auc
    print(f"average of {num_runs} runs:")
    print(f"  accuracy: {accuracy / num_runs}")
    print(f"  precision: {precision / num_runs}")
    print(f"  recall: {recall / num_runs}")
    print(f"  F1 score: {f1 / num_runs}")
    print(f"  ROC-AUC score: {roc_auc / num_runs}")
    print(f"saved best run:")
    print(f"  accuracy: {best_accuracy}")
    print(f"  precision: {best_precision}")
    print(f"  recall: {best_recall}")
    print(f"  F1 score: {best_f1}")
    print(f"  ROC-AUC score: {best_roc_auc}")
    save_model(f"work/{project}-model.pt", best_model, best_tfidf)
    print(f"saved model at work/{project}-model.pt")
