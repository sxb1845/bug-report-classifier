import classifier
import torch
import sys

def transform(text, tfidf):
    text = classifier.remove_html(text)
    text = classifier.remove_emoji(text)
    text = classifier.remove_stopwords(text)
    text = classifier.clean_str(text)
    return torch.Tensor(tfidf.transform([text]).toarray())

def predict(text, model, tfidf):
    x = transform(text, tfidf)
    with torch.no_grad():
        model.eval()
        y = model(x.to(classifier.device)).squeeze().cpu()
        return y.numpy() > 0, torch.sigmoid(y)

def show_prediction(text, model, tfidf):
    prediction, probability = predict(text, model, tfidf)
    if prediction:
        print("  ✓ likely performance related")
        print(f"  relation confidence: {probability}")
    else:
        print("  ✗ not performance related")
        print(f"  relation confidence: {probability}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        model_path = input("model path? ")
    else:
        model_path = sys.argv[1]
    model, tfidf = classifier.load_model(model_path)
    print(f"loaded model {model_path}")
    if len(sys.argv) < 3:
        print("reading input")
        while True:
            text = input("> ")
            show_prediction(text, model, tfidf)
    else:
        document_path = sys.argv[2]
        document = open(document_path, "r")
        text = document.read()
        print(f"loaded document {document_path}")
        show_prediction(text, model, tfidf)
