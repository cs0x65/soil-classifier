from classifier.classification import SoilClassifier
from dataset.preparation.ground_truths import GroundTruthBuilder

if __name__ == '__main__':
    builder = GroundTruthBuilder("./dataset/soil-profile-data.csv")
    builder.build_all()
    SoilClassifier(builder.out_csv_file).classify()
