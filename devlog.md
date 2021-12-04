# Dev

## Phase 1 Basic Setup

### Environment

- Uses docker from ThinkMatch master
- Server has docker.io installed, but does not have nvidia-docker
  - Solved by following instructions on nvidia website to install nvidia-docker
  - Now should be able to run torch implementation of ThinkMatch models

## Phase 2 Converting CIE to Paddle Framework

### Parameter Conversion

#### Unexpected Parameters

```sh
RuntimeError: Error(s) in loading state_dict for Net:
    Missing key(s) in state_dict: "gnn_layer_0.gconv.node_fc.weight", "gnn_layer_0.gconv.node_fc.bias", "gnn_layer_0.gconv.node_sfc.weight", "gnn_layer_0.gconv.node_sfc.bias", "gnneight", "gnn_layer_0.gconv.node_sfc.bias", "gnn_layer_0.gconv.edge_fc.weight", "gnn_layer_0.gconv.edge_fc.bias", "gnn_layer_1.gconv.node_gnn_layer_1.gconv.node_sfc.weight", "gnn_layer_fc.weight", "gnn_layer_1.gconv.node_fc.bias", "gnn_layer_1.gconv.node_sfc.weight", "gnn_layer_1.gconv.node_sfc.bias", "gnn_layer_1.gconv.edge_fc.weight", "gnn_layer_1.gconv.edge_fc.bias".                                                                                       _sfc.weight", "gnn_layer_0.gconv1.node_sfc.bias
    Unexpected key(s) in state_dict: "gnn_layer_0.gconv1.node_fc.weight", "gnn_layer_0.gconv1.node_fc.bias", "gnn_layer_0.gconv1.nodefc.bias", "gnn_layer_1.gconv1.node_sfc.weight",_sfc.weight", "gnn_layer_0.gconv1.node_sfc.bias", "gnn_layer_0.gconv1.edge_fc.weight", "gnn_layer_0.gconv1.edge_fc.bias", "gnn_layer_1.gconv1.node_fc.weight", "gnn_layer_1.gconv1.node_fc.bias", "gnn_layer_1.gconv1.node_sfc.weight", "gnn_layer_1.gconv1.node_sfc.bias", "gnn_layer_1.gconv1.edge_fc.weight", "gnn_layer_1.gconv1.edge_fc.bias".
```

- Seems that old implementation of `Siamese_ChannelIndependentConv` has two `ChannelIndependentConv` layers
  - `gconv1` and `gconv2` respectively
- But latest version only have one layer
  - `gconv` layer
  - This causes name mismatch
- Renaming `gconv` to `gconv1` in `Siamese_ChannelIndependentConv` seems to be able to solve the problem
  - Model performance seems to be stable
  - Perhaps just a naming mistake

## Dec.04

- `convert_params` not working
  - Got `KeyErrors`
    - String replacement is buggy
    - Perhaps the script stil has bugs
- `Paddle` model cannot dynamically add layers
  - `setattr` does not work as modules added with `setattr` cannot be tracked by Paddle so they do not exist in state dict
  - Temporary fix is to hard-code a 2-GNN-layer CIE model so that at least the model can be loaded and the evaluation can be run for the pretrained default architecture
