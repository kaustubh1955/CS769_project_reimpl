import os, json, pickle
from collections import defaultdict
from datasets import load_from_disk
from tqdm import tqdm

TREC_FILE = "run.retrieve.top_50.mkqa_en.wiki-100w-multilingual-alllangs.dev.BAAI_bge-m3.trec"
OUTPUT_JSON = "en_top5_with_content.json"
DATASET_BASE = "../datasets"
CACHE_DIR = "../cache"
TOP_K = 5

def read_trec(path, k):
    result = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            qid, _, doc_id, rank, *_ = line.strip().split()
            result[qid].append({"doc_id": doc_id, "rank": int(rank)})
    trimmed = {}
    for qid, lst in result.items():
        trimmed[qid] = sorted(lst, key=lambda x: x["rank"])[:k]
        for i, item in enumerate(trimmed[qid], 1):
            item["rank"] = i
    return trimmed

content_cache = {}
def _cache_path(prefix):
    return os.path.join(CACHE_DIR, f"{prefix}_content.pkl")
def _dataset_folder(prefix):
    return "kilt-100w_full" if prefix.startswith("kilt-100w") else f"{prefix}_train"
def _load_dataset(prefix):
    path = os.path.join(DATASET_BASE, _dataset_folder(prefix))
    if not os.path.exists(path):
        return {}
    ds = load_from_disk(path)
    return {str(r["id"]): r["content"] for r in ds}
def get_content(doc_id):
    prefix, wid = doc_id.rsplit("_", 1)
    if prefix not in content_cache:
        cp = _cache_path(prefix)
        if os.path.exists(cp):
            with open(cp, "rb") as f:
                content_cache[prefix] = pickle.load(f)
        else:
            content_cache[prefix] = _load_dataset(prefix)
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(cp, "wb") as f:
                pickle.dump(content_cache[prefix], f)
    return content_cache[prefix].get(wid, "")

def main():
    topk_data = read_trec(TREC_FILE, TOP_K)
    enriched = {}
    for qid, lst in tqdm(topk_data.items(), desc="Attach content"):
        enriched[qid] = [
            {"doc_id": item["doc_id"], "rank": item["rank"], "content": get_content(item["doc_id"])}
            for item in lst
        ]
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()