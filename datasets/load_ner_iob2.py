# Loading script for the Cantemist NER dataset. 
import datasets
import os
import csv

logger = datasets.logging.get_logger(__name__)

CITATION = ""
DESCRIPTION = ""
SPLITS = ["train", "dev", "test"]
TYPES = ["wl"]
DATASETS_DIR = "/home/ccasimiro/ccasimiro/clinical-nested-ner/datasets"
DATASETS = {"wl": {f"{split}": os.path.join(DATASETS_DIR, f"formatted_data/wl_joint_labeling/{split}.iob2") 
                  for split in SPLITS}}


class NerIOB2Config(datasets.BuilderConfig):
    """BuilderConfig for NER datasets with IOB2 tagging schema."""
    def __init__(self, **kwargs):
        super(NerIOB2Config, self).__init__(**kwargs)

class NerIOB2(datasets.GeneratorBasedBuilder):
    """Cantemist Ner dataset."""
    BUILDER_CONFIGS = [NerIOB2Config(name=type, description=f"{type} NerIOB2 dataset") 
                      for type in TYPES]
        
    def _info(self):
        """Return the features of the current dataset"""
  
        # Extract labels across splits
        labels = []
        for split in SPLITS:
            with open(DATASETS[self.config.name][split]) as fn: 
                for row in csv.reader(fn, delimiter=' ', quoting=csv.QUOTE_NONE):
                    if row:
                        labels.append(row[1])

        return datasets.DatasetInfo(
            description=DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=list(set(labels))
                        )
                    ),
                }
            ),
            supervised_keys=None,
            citation=CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": DATASETS[self.config.name]['train']}),
                datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": DATASETS[self.config.name]['test']}),
                datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": DATASETS[self.config.name]['dev']})]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as fn:
            guid = 0
            tokens = []
            ner_tags = []
            for line in csv.reader(fn, delimiter=' ', quoting=csv.QUOTE_NONE):
                if not line:
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "ner_tags": ner_tags,
                        }
                        guid += 1
                        tokens = []
                        ner_tags = []
                else:
                    tokens.append(line[0])
                    ner_tags.append(line[1].rstrip())
            # last example
            yield guid, {
                "id": str(guid),
                "tokens": tokens,
                "ner_tags": ner_tags,
            }
