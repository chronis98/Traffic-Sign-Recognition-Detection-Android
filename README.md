# Traffic-Sign-Recognition-Detection-Android
Following [Previous Traffic-Sign-Recognition-Detection project](https://github.com/chronis98/Traffic-Sign-Recognition-Detection) , this is an approach of deploying and using Tensorflow Deep Learning models on android devices on camera road footage.
# Implementation
## Graphs
Creating necessary graphs, specialized for Tensorflow Lite conversion-
*export_tflite_ssd_graph.py --input_type image_tensor --pipeline_config_path training/ssd_mobilenet_v1_coco.config --trained_checkpoint_prefix training/model.ckpt-27412 --output_directory training2  --add_postprocessing_op true* 
