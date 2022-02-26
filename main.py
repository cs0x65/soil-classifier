from classifier.classification import SoilClassifier
from dataset.preparation.ground_truths import GroundTruthBuilder

if __name__ == '__main__':
    GroundTruthBuilder("./dataset/soil-profile-data.csv").build_all()
    SoilClassifier("./dataset/soil-profile-data_with_gt.csv").classify()
