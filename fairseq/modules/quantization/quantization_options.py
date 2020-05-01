# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


def parse_config_yaml(yaml_data):
    # Initialize to default options.
    quantization_options = {
        "n_centroids": {
            "Linear": ["in_features", {"*": 256}],
            "Embedding": ["embedding_dim", {"*": 256}],
        },
        "block_sizes": {
            "Linear": ["fuzzy_name", {"fc": 8, "attn": 4, "emb": 4}],
            "Embedding": ["fuzzy_name", {"emb": 8}],
        },
        "layers_to_quantize": [
            "decoder\\.layers\\.\\d+\\.fc[12]",
            "decoder\\.embed_tokens\\.embeddings\\.[012]\\.[01]",
            "decoder\\.layers\\.\\d+\\.self_attn\\.(k_proj|v_proj|q_proj|out_proj)",
        ],
    }

    if "n_centroids" in yaml_data:
        quantization_options["n_centroids"] = {
            layer: convert_yaml_to_tuple(layer_data)
            for layer, layer_data in yaml_data["n_centroids"].items()
        }
    if "block_sizes" in yaml_data:
        quantization_options["block_sizes"] = {
            layer: convert_yaml_to_tuple(layer_data)
            for layer, layer_data in yaml_data["block_sizes"].items()
        }
    if "layers_to_quantize" in yaml_data:
        quantization_options["layers_to_quantize"] = yaml_data["layers_to_quantize"]

    return quantization_options


def convert_yaml_to_tuple(yaml_dictionary):
    """Converts a yaml dictionary with two keys: `key` and `value` into a two
    argument tuple of those values."""
    return (yaml_dictionary["key"], yaml_dictionary["value"])
