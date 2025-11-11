from ..dataset_processor import *
import datasets
import os


class MKQA(Processor):

    def __init__(self, lang, *args, **kwargs):
        dataset_name = f'mkqa_{lang}'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)
        self.lang = lang
        
    def process(self):
        mkqa = datasets.load_dataset('mkqa')
        try:
            kilt_nq = datasets.load_dataset("kilt_tasks", "nq")
            kilt_nq_ids = {s['id']: i for i, s in enumerate(kilt_nq[self.split])}
            mkqa_ids = {s['example_id']: i for i, s in enumerate(mkqa[self.split])}
            overlap_ids = set(mkqa_ids.keys()).intersection(set(kilt_nq_ids.keys()))
            overlap_mkqa = mkqa['train'].select([mkqa_ids[i] for i in overlap_ids])
            overlap_kilt_nq = kilt_nq['train'].select([kilt_nq_ids[i] for i in overlap_ids])
            dataset = overlap_kilt_nq.add_column(
                "content", [sample['queries'][self.lang] for sample in overlap_mkqa]
            )
            dataset = dataset.add_column(
                "label",
                [[a['text'] for a in sample['answers'][self.lang] if a['text'] is not None]
                 for sample in overlap_mkqa]
            )
            dataset = dataset.filter(lambda example: len(example['label']) > 0)
            dataset = dataset.map(
                lambda example: {
                    'ranking_label': [
                        [prov['wikipedia_id'] for prov in el['provenance']]
                        if len(el['answer']) > 0 and len(el['provenance']) > 0 else []
                        for el in example['output']
                    ]
                }
            )
            dataset = dataset.remove_columns(['meta'])
        except RuntimeError as e:
            if 'Dataset scripts are no longer supported' in str(e):
                print("WARNING: 'kilt_tasks' script unsupported in this 'datasets' version. Using MKQA only (no Natural Questions overlap, ranking labels empty). Pin 'datasets<=2.20.0' for original behavior.")
                mkqa_split = mkqa[self.split]
                # Build dataset directly from MKQA examples
                def build_example(sample):
                    answers = [a['text'] for a in sample['answers'][self.lang] if a['text'] is not None]
                    return {
                        'id': sample['example_id'],
                        'content': sample['queries'][self.lang],
                        'label': answers,
                        'ranking_label': []  # no provenance without kilt_nq alignment
                    }
                dataset = mkqa_split.map(build_example)
                dataset = dataset.filter(lambda example: len(example['label']) > 0)
            else:
                raise
        return dataset

class XORQA(Processor):

    def __init__(self, lang, *args, **kwargs):
        dataset_name = f'xor_tydiqa_{lang}'
        self.lang = lang
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        def extend(label, lang):
            if "yes" in label:
                label += {"ru": ["да"], "ko": ["예"], "ja": ["はい"], "fi": ["kyllä", "joo"], "ar": ["نعم", "أجل", "بلى"]}[lang]
            if "no" in label:
                label += {"ru": ["нет"], "ko": ["아니요"], "ja": ["いいえ"], "fi": ["ei"], "ar": ["لا"]}[lang]
            return label
        os.system("wget https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_dev_full_v1_1.jsonl")
        dataset = datasets.load_dataset("json", data_files="xor_dev_full_v1_1.jsonl")["train"] # the file should be already .dev, and train is just default hf label
        dataset = dataset.filter(lambda example: example['lang']==self.lang)
        # discarding empty answers 
        dataset = dataset.map(lambda example: {'label': extend([el for el in example['answers'] if len(el) > 0], self.lang)})
        dataset = dataset.rename_column("question", "content")
        #dataset = dataset.remove_columns(['meta', 'output'])
        os.system("rm xor_dev_full_v1_1.jsonl")
        return dataset

class TydiQA(Processor):

    def __init__(self, langcode="en", language="english", *args, **kwargs):
        dataset_name = f'tydiqa_{langcode}'
        self.language = language
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        dataset =  datasets.load_dataset("google-research-datasets/tydiqa", "secondary_task")[self.split] 
        dataset = dataset.filter(lambda example: example['id'].startswith(self.language))
        dataset = dataset.map(lambda example: {'label': [el for el in example['answers']["text"] if len(el) > 0]})
        dataset = dataset.rename_column("question", "content")
        dataset = dataset.remove_columns(['title', 'context', 'answers'])
        return dataset

