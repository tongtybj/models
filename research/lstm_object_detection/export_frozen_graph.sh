#!/usr/bin/env bash

python export_inference_graph.py  --input_type image_tensor \
    --pipeline_config_path ${1} \
    --trained_checkpoint_prefix ${2} \
    --output_directory ${3} \
    --input_shape ${4} \
    --write_inference_graph ${5:-False} \
    --config_override " \
            model{ \
             ssd{ \
              post_processing { \
                batch_non_max_suppression { \
                        score_threshold: 0.0 \
                        iou_threshold: 0.5 \
                } \
               } \
              } \
            }"
