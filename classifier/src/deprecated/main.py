import logging
import os
import torch
import joblib
from __future__ import annotations
from utils import set_determinism, create_embeddings_file
from data import extract_model_number, load_fasta, prepare_for_pca
from predict import predict
from embed import preprocess_df, compute_embeddings, monitor_load_embed_model
from pool import pool_embeddings
from dotenv import load_dotenv

@hydra.main(config_path="./", config_name="config", version_base=None)
def init(cfg:DictConfig) -> None: #noqa: D401
    """
    Main function that performs prediction of protein sequences (stored in FASTA file).
    1) Embeds sequences
    2) Predicts classes
    3) Stores pooled embeddings 
    4) Computes and stores t-SNE coordinates

    Parameters
    -----------
    cfg (DictConfig): The Hydra configuration object
    """
    logger = logging.getLogger(__name__)
    logger.info("Loaded config:\n" + OmegaConf.to_yaml(cfg))

    # 1. Set determinism and environment
    set_determinism(cfg.seed)
    if cfg.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        if cfg.device == "cuda":
            logger.info("CUDA not available, using CPU.")

    # 2. Get model number
    model_no = extract_model_number(cfg.model.path)
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir  # type: ignore[attr-defined]

    # 3. Instantiate model
    checkpoint = cfg.embeddings.model_type
    logger.info(f"Instantiating model: {checkpoint} on {device}")
    model, tokenizer = monitor_load_embed_model(checkpoint, device)

    # 4. Preprocess amino acid sequences
    df = load_fasta(cfg.new_fasta.path)
    preprocessed_df = preprocess_df(df, checkpoint)

    # 5. Compute and pool embeddings
    embed_dict = compute_embeddings(checkpoint, preprocessed_df, model, tokenizer, cfg.embeddings.only_last)
    embed_dict_pooled = pool_embeddings(embed_dict, checkpoint, layers=cfg.pool.layers, strategy=cfg.pool.strategy)

    # 6. Predict
    predictions_df, _ = predict(embed_dict, preprocessed_df, cfg.model.path, model_no, cfg.model.type)

    # 7. Store pooled embeddings
    embeddings_path = os.path.join(output_dir, f"protein_embeddings_{checkpoint}_{"-".join(str(l) for l in cfg.pool.layers)}_{cfg.pool.strategy}.h5")
    create_embeddings_file(embed_dict_pooled, preprocessed_df, str(embeddings_path))
    logger.info(f"Saved pooled embeddings to {embeddings_path}")

    # 8. Store predictions
    predictions_df.to_csv(os.path.join(output_dir, "predictions.tsv"), sep="\t", index=False)
    logger.info(f"Saved predictions to {output_dir}")

    # 9. Load precomputed PCA and t-SNE
    pca_data = joblib.load(cfg.dimred.pca)
    pca_model = pca_data["model"]
    pca_coords = pca_data["coords"]
    pca_ids = pca_data["ids"]

    tsne_data = joblib.load(cfg.dimred.tsne)
    X_tsne = tsne_data["coords"]
    tsne_ids = tsne_data["ids"]

    # 10. Project into embedding
    X_new, ids_new = prepare_for_pca(embed_dict_pooled)
    coords_new = pca_model.transform(X_new)
    X_tsne_new = X_tsne.transform(coords_new)

    # 11. Save new t-SNE coordinates
    tsne_path = os.path.join(output_dir, "tsne_new.joblib")
    joblib.dump(X_tsne_new, tsne_path)
    logger.info(f"Saved new tSNE coordinates to {tsne_path}")


if __name__ == "__main__":
    init()
