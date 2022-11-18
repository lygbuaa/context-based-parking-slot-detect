# example
1. the only onnxruntime official linux c++ demo:  https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx/fns_candy_style_transfer 

2. onnxruntime with opencv: https://blog.csdn.net/qq_44747572/article/details/121467657

3. Keypoint is, onnx model accept NCHW, tensorflow model accept NHWC, cv::Mat is interleaved format, BBBGGGRRR, which is also NHWC. The fns_candy_style_transfer use libpng and hwc_to_chw() to do this transform, OpenCV also provide cv::dnn::blobFromImage() to do this.

4. put onnxruntime-linux-x64 into ort directory

5. Snpe_EP is a multi-inputs multi-outputs demo