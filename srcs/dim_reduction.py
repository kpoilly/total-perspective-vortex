import mne
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


def main():
    try:
        data_path = "data/processed/data.npz"
        data = np.load(data_path)
        
        X = data['X']
        y = data['y']
        data.close()
        print("Data loaded successfully")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    except FileNotFoundError:
        print(f"Error: File '{data_path}' not found.")
    except Exception as e:
        print(f"Error: {e}")
        
    csp = mne.decoding.CSP(n_components=6, reg=None, log=True, norm_trace=False)
    clf = SVC(kernel='linear')

    csp.fit(X_train, y_train)
    X_train_csp = csp.transform(X_train)
    X_test_csp = csp.transform(X_test)

    clf.fit(X_train_csp, y_train)
    score = cross_val_score(clf, X_test_csp, y_test, cv=5)
    score = np.mean(score)
    print(f"Accuracy: {score}")
        
if __name__ == '__main__':
    main()
