import csv
from typing import Dict


class GroundTruthBuilder(object):
    PH_GENERIC_CLASS = 'ph_class'
    EC_GENERIC_CLASS = 'ec_class'
    PH_VEG_ROW_CROPS_CLASS = 'ph_veg_and_row_crops'
    PH_FRUITS_NUTS_CLASS = 'ph_fruits_and_nuts'

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
                self.dataset_with_gd.append(row)

            self.headers.append(GroundTruthBuilder.PH_GENERIC_CLASS)
            self.headers.append(GroundTruthBuilder.PH_VEG_ROW_CROPS_CLASS)
            self.headers.append(GroundTruthBuilder.PH_FRUITS_NUTS_CLASS)
            self.headers.append(GroundTruthBuilder.EC_GENERIC_CLASS)

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
