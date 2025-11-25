import tensorflow as tf

ckpt_path = r"E:\Python\MVSNet\pretrained model\model.ckpt-150000"

reader = tf.train.NewCheckpointReader(ckpt_path)
keys = reader.get_variable_to_shape_map().keys()

layernorm = any("LayerNorm" in k for k in keys)
groupnorm = any("group_norm" in k or "/gn/" in k for k in keys)

print("Checkpoint type:")
if layernorm:
    print(" Detected LayerNorm-based R-MVSNet checkpoint (R-MVSNet style).")
elif groupnorm:
    print(" Detected GroupNorm-based MVSNet / old R-MVSNet checkpoint (GroupNorm style).")
else:
    print(" Cannot determine norm type, maybe 3DCNNs-only or different training branch.")

print("\nExample variable names:")
for k in list(keys)[:50]:
    if "conv_gru" in k or "Gates" in k or "Norm" in k:
        print("   ", k)
