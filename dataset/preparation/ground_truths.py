import csv
from typing import Dict


class GroundTruthBuilder(object):
    PH_GENERIC_CLASS = 'ph_class'
    EC_GENERIC_CLASS = 'ec_class'
    PH_VEG_ROW_CROPS_CLASS = 'ph_veg_and_row_crops'
    PH_FRUITS_NUTS_CLASS = 'ph_fruits_and_nuts'
    OPTIMAL_MACRO_NUTRIENTS_CLASS = 'optimal_macro_nutrients_class'
    OPTIMAL_MICRO_NUTRIENTS_CLASS = 'optimal_micro_nutrients_class'

    # May need to specify what are the tolerable avg distances from the neighbors for them not to be outliers when using
    # kNN algorithm.
    FEATURES_TO_LABELS_DICT = {
        'ph': [PH_GENERIC_CLASS, PH_VEG_ROW_CROPS_CLASS, PH_FRUITS_NUTS_CLASS],
        'ec': [EC_GENERIC_CLASS],
        'av_p,av_k,av_s': [OPTIMAL_MACRO_NUTRIENTS_CLASS],
        'av_zn,av_cu,av_mn': [OPTIMAL_MICRO_NUTRIENTS_CLASS]
    }

    def __init__(self, in_csv_file: str, out_csv_file: str = None):
        self.in_csv_file = in_csv_file.strip()
        self.out_csv_file = out_csv_file.strip() if out_csv_file else f'{self.in_csv_file.split(".csv")[0]}_with_gt.csv'
        self.headers = None
        self.dataset_with_gd = list()

    def build_all(self):
        with open(self.in_csv_file) as csv_file:
            reader = csv.DictReader(csv_file)
            print(f'fieldnames before: {reader.fieldnames}')
            self.headers = reader.fieldnames
            for row in reader:
                print(f'Row no {reader.line_num}: {row}')
                self.build_ph_labels(row)
                self.build_ec_labels(row)
                self.build_optimal_macro_nutrients_binary_label(row)
                self.build_optimal_micro_nutrients_binary_label(row)
                self.dataset_with_gd.append(row)

            self.headers.append(GroundTruthBuilder.PH_GENERIC_CLASS)
            self.headers.append(GroundTruthBuilder.PH_VEG_ROW_CROPS_CLASS)
            self.headers.append(GroundTruthBuilder.PH_FRUITS_NUTS_CLASS)
            self.headers.append(GroundTruthBuilder.EC_GENERIC_CLASS)
            self.headers.append(GroundTruthBuilder.OPTIMAL_MACRO_NUTRIENTS_CLASS)
            self.headers.append(GroundTruthBuilder.OPTIMAL_MICRO_NUTRIENTS_CLASS)

            self._write_to_csv()

    def _write_to_csv(self):
        with open(self.out_csv_file, mode='w') as csv_file:
            print(f'Writing to CSV file: {self.out_csv_file}')
            print(f'Headers: {self.headers}')
            dict_writer = csv.DictWriter(csv_file, fieldnames=self.headers)
            dict_writer.writeheader()
            dict_writer.writerows(self.dataset_with_gd)
            print(f'Finished writing to the CSV file: {self.out_csv_file}')

    @staticmethod
    def build_ph_labels(row: Dict):
        ph = float(row['ph'])

        # Generic class based on ph value
        ph_class = ''
        if ph < 6:
            # plants suffer when ph < 4.8
            ph_class = 'Acidic'
        elif ph > 7.3:
            ph_class = 'Alkaline'
        else:
            ph_class = 'Optimal'
        row[GroundTruthBuilder.PH_GENERIC_CLASS] = ph_class

        # Crop specific ph class
        row[GroundTruthBuilder.PH_VEG_ROW_CROPS_CLASS] = 5.8 <= ph <= 6.5
        row[GroundTruthBuilder.PH_FRUITS_NUTS_CLASS] = 5.5 <= ph <= 5.8

    @staticmethod
    def build_ec_labels(row: Dict):
        ec = float(row['ec'])

        # Generic class based on ec value
        ec_class = ''
        if ec <= 0.5:
            # normal
            ec_class = 'Normal'
        elif 0.5 < ec <= 1.5:
            # marginally high: causes salt injury
            ec_class = 'Marginally High'
        else:
            ec_class = 'Excessive'
        row[GroundTruthBuilder.EC_GENERIC_CLASS] = ec_class

    @staticmethod
    def build_optimal_macro_nutrients_binary_label(row: Dict):
        av_p = float(row['av_p'])
        av_k = float(row['av_k'])
        # TODO: need this cleanup in the build_all itself using pandas missing values construct - that means the csv
        # file shall be 1st updated with the missing values, and do data cleanup of required and then find the ground
        # truths
        try:
            av_s = float(row['av_s'])
        except ValueError:
            av_s = row['av_s'] = 0

        # pattern followed: lowest of medium range to highest of optimal range
        # medium 58.2843 - 78.4596, optimum 80.7013 - 112.085
        av_p_optimal = 58.2843 <= av_p <= 112.085
        # medium 203.995 - 291.421, optimum 293.663 - 392.298
        av_k_optimal = 203.995 <= av_k <= 392.298
        row[GroundTruthBuilder.OPTIMAL_MACRO_NUTRIENTS_CLASS] = av_p_optimal and av_k_optimal and av_s <= 10

    @staticmethod
    def build_optimal_micro_nutrients_binary_label(row: Dict):
        av_zn = float(row['av_zn'])
        try:
            av_cu = float(row['av_cu'])
        except ValueError:
            av_cu = row['av_cu'] = 0

        try:
            av_mn = float(row['av_mn'])
        except ValueError:
            av_mn = row['av_mn'] = 0

        # pattern followed: lowest of medium range to highest of optimal range
        # medium 3.1 - 4.0, optimum 4.0 - 8.0
        av_zn_optimal = 3.1 <= av_zn <= 8.0
        row[GroundTruthBuilder.OPTIMAL_MICRO_NUTRIENTS_CLASS] = av_zn_optimal and av_cu >= 1.0 and av_mn >= 40
