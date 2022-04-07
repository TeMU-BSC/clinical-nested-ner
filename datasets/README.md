# clinical-nested-ner-data
Module used to transform the raw files to nested NER format.

1. In the raw_data folder, unzip the pharmaconer, and clinical trials zip files and put the dev/test/train folders in the respective directories.

2. Run `pip install -r requirements.txt` to install all dependencies

3. Execute `gen_data_for_clinical_trials.py` and then `gen_data_for_pharmaconer.py`.

4. The files for training the MLC model will be located in the formatted_data/MLC folder.

5. The waiting list corpus files are already located in formatted_data/MLC/wl, while the NUBes preprocessing is still in progress as some decisions on the preprocessing of the corpus are still to be consulted.