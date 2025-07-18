# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from datasets import load_dataset

dataset = load_dataset(
  'csv', 
  data_files=
  {
    'train': 'raw_data/csts/csts_train.csv',
    'validation': 'raw_data/csts/csts_validation.csv',
    'test': 'raw_data/csts/csts_test.csv'
  }
)

print(dataset)