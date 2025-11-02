import json
import argparse
from pathlib import Path
from typing import List
import numpy as np
import torch
import clip
from tqdm import tqdm

def read_json_input(path: Path):
    data = json.load(open(path, "r", encoding="utf-8"))
    if isinstance(data, dict):
        return [data]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("JSON должен быть объектом или массивом объектов")

def make_text_from_record(rec: dict) -> str:
    parts = []
    if rec.get("name"):
        parts.append(rec["name"])
    if rec.get("title_current"):
        parts.append(rec["title_current"])
    if rec.get("location"):
        parts.append(rec["location"])
    if rec.get("top_skills"):
        parts.append(", ".join(rec["top_skills"]))
    if rec.get("highlights"):
        parts.append(" | ".join(rec["highlights"]))
    if rec.get("position_pitch"):
        parts.append(rec["position_pitch"])
    if rec.get("cta"):
        parts.append(rec["cta"])
    if rec.get("language"):
        parts.append(f"lang:{rec['language']}")
    return " — ".join(p for p in parts if p)

def batchify(lst: List[str], batch_size: int):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def main():
    parser = argparse.ArgumentParser(description="Vectorize JSON records with OpenAI CLIP (text embeddings)")
    parser.add_argument("input_json", type=str, help="Путь к JSON файлу (объект или массив объектов)")
    parser.add_argument("--outfile", type=str, default="embeddings.npz", help="куда сохранить npz с embeddings")
    parser.add_argument("--json_out", type=str, default="embeddings.json", help="куда сохранить json с embeddings (как list of lists)")
    parser.add_argument("--model", type=str, default="ViT-B/32", help="вариант CLIP (по умолчанию ViT-B/32)")
    parser.add_argument("--batch", type=int, default=64, help="размер батча для токенизации/векторизации")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model, preprocess = clip.load(args.model, device=device)
    model.eval()

    records = read_json_input(Path(args.input_json))
    texts = [make_text_from_record(r) for r in records]

    all_embeddings = []
    with torch.no_grad():
        for batch_texts in tqdm(list(batchify(texts, args.batch)), desc="Batches"):
            tokenized = clip.tokenize(batch_texts, truncate=True).to(device)
            text_features = model.encode_text(tokenized)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            emb_np = text_features.cpu().numpy()
            all_embeddings.append(emb_np)

    all_embeddings = np.vstack(all_embeddings)
    print("Embeddings shape:", all_embeddings.shape)

    np.savez_compressed(args.outfile, embeddings=all_embeddings)
    print("Saved:", args.outfile)

    emb_list = [emb.tolist() for emb in all_embeddings]
    out_json = {
        "records_count": len(records),
        "model": args.model,
        "embeddings": emb_list,
    }
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False)
    print("Saved JSON:", args.json_out)



if __name__ == "__main__":
    main()
