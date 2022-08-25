import glob
import os
from glob import glob
from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record
import pandas as pd


def my_summary_iterator(path):
    for r in tf_record.tf_record_iterator(path):
        yield event_pb2.Event.FromString(r)


def main():
    # Get all event* runs from logging_dir subdirectories
    logging_dir = '/home/champib/runs/hmm_ALE'
    files = [y for x in os.walk(logging_dir) for y in glob(os.path.join(x[0], '*/events.*'))]

    for file in files:
        column = "Rewards"
        print(f"Processing: {file}")
        df = pd.concat([pd.DataFrame([value.simple_value], columns=[column]) for event in my_summary_iterator(file) for value in event.summary.value], ignore_index=True)
        df.to_csv(f"../csv_dir/{os.path.basename(file)}.csv")


if __name__ == '__main__':
    main()
