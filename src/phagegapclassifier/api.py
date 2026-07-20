"""
API call definitions.
"""

from __future__ import annotations

import logging
from flask import request
from phagegapclassifier import config, app, plm, tokenizer, checkpoint, classifier, label_map, pca, tsne
from data import parse_sequence_data, prepare_for_pca
from predict import predict
from embed import preprocess_df, compute_embeddings
from pool import pool_embeddings
from traceback import print_exc

logger = logging.getLogger(__name__)


@app.get("/health")
def health():
    return "PhageGap Classifier is active.", 200

# ibmidocker -> spock (134....)/predict
"""
@app.post("/predict")
def predict():
    try:
        # Parse sequence data from request.
        df = parse_sequence_data(request.data)
        preprocessed_df = preprocess_df(df, checkpoint)

        # Create embeddings of passed sequence data using `plm`.
        embed_dict = compute_embeddings(checkpoint, preprocessed_df, plm, tokenizer, True)
        embed_dict_pooled = pool_embeddings(embed_dict, checkpoint,
                                            layers=config.get('embed').get('pool_layers', []),
                                            strategy=config.get('embed').get('pool_strategy', 'mean'))

        # Predict classes from embeddings.
        predictions_df, _ = predict(embed_dict, preprocessed_df, classifier, label_map)
        predictions_df.drop( "processed_seq", axis='columns', inplace=True )

        # Load precomputed PCA and t-SNE.
        pca_model = pca["model"]
        pca_coords = pca["coords"]
        pca_ids = pca["ids"]
        X_tsne = tsne["coords"]
        tsne_ids = tsne["ids"]

        # Project into embedding.
        X_new, ids_new = prepare_for_pca(embed_dict_pooled)
        coords_new = pca_model.transform(X_new)
        X_tsne_new = X_tsne.transform(coords_new)

        # Return data...
    except Exception as e :
        print_exc()
        return {"error": str(e)}, 500
"""
