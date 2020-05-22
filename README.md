# Traffic-Sign-Recognition-Detection-Android
Following [Previous Traffic-Sign-Recognition-Detection project](https://github.com/chronis98/Traffic-Sign-Recognition-Detection) , this is an approach of deploying and using Tensorflow Deep Learning models on android devices on camera road footage.
# Implementation
Configuration of [Tensorflow Lite android example source code](https://github.com/chronis98/Traffic-Sign-Recognition-Detection), in order to host and use two custom retrained tensorflow Lite models.Modification on UI was used for displaying classified image.

   tflite model        |  Labels
1. *square.tflite      labels.txt*-Detection
2. *detection.tflite   labels2.txt*-Classification

## Graphs
### Creating necessary graphs, specialized for Tensorflow Lite conversion-

*export_tflite_ssd_graph.py --input_type image_tensor --pipeline_config_path training/ssd_mobilenet_v1_coco.config --trained_checkpoint_prefix training/model.ckpt-27412 --output_directory training2  --add_postprocessing_op true* 
(note that parameter "add_postprocessing_op true" is passed, for another custom op node added to the frozen graph)

### Tensorflow Lite conversion of frozen graphs-

*tflite_convert / --output_file=square.tflite / --graph_def_file=training2/tflite_graph.pb / --output_foprmat=TFLITE / --input_shapes=1,300,300,3 / --input_arrays=image_tensor / --output_arrays=TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3 / --inference_type=QUANTIZED_UINT8 / --default_ranges_min=0 / --default_ranges_max=6 / --mean_values=128 / --std_dev_values=127 / --change_concat_input_ranges=false / --allow_custom_ops*

## Results-
![](https://github.com/chronis98/Traffic-Signs-Recognition-Detection-Android/blob/master/giphy.gif "asd")
<img src="https://github.com/chronis98/Traffic-Signs-Recognition-Detection-Android/blob/master/giphy.gif" width="40" height="40" />
