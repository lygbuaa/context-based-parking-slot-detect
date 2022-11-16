#!/usr/bin/python
# -*- coding:utf-8 -*-

import os, sys, time
import numpy as np
import onnx
import onnxruntime as ort
# import onnxsim
# pip install onnxmltools
from onnxmltools.utils import float16_converter

class OnnxWrapper(object):
    def __init__(self):
        self.print_version()

    def print_version(self):
        print("onnx version: {}".format(onnx.__version__))
        print("onnx runtime version: {}".format(ort.__version__))
        print("onnx runtime: {}".format(ort.get_device()))

    def benchmark(self, ortss, inputs, nwarmup=10, nruns=10000):
        print("Warm up ...")
        for _ in range(nwarmup):
            features = self.run_onnx_model(ortss, inputs)
        
        print("Start timing ...")
        timings = []
        for i in range(1, nruns+1):
            start_time = time.time()
            features = self.run_onnx_model(ortss, inputs)
            end_time = time.time()
            timings.append(end_time - start_time)
            if i%10==0:
                print('Iteration %d/%d, ave batch time %.2f ms'%(i, nruns, np.mean(timings)*1000))
        print('Average batch time: %.2f ms'%(np.mean(timings)*1000))

    def load_onnx_model(self, model_path):
        provider = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.ortss = ort.InferenceSession(model_path, providers=provider)
        print("load model {} onto {}".format(model_path, self.ortss.get_providers()))
        return self.ortss

    #ortss: onnxruntime loaded model, inputs: [input0, input1]
    def run_onnx_model(self, ortss, inputs):
        input_layers = ortss.get_inputs()
        output_layers = ortss.get_outputs()
        # print("input_layers: {}, output_layers: {}".format(input_layers, output_layers))
        input_dict = {}
        output_list = []
        for idx, in_layer in enumerate(input_layers):
            input_dict[in_layer.name] = inputs[idx]
            # print("[{}]- in_layer: {}".format(idx, in_layer))
        # print("input_dict: {}".format(input_dict))

        for idx, out_layer in enumerate(output_layers):
            output_list.append(out_layer.name)
            # print("[{}]- out_layer: {}".format(idx, out_layer))
        # print("output_list: {}".format(output_list))

        outputs = ortss.run(output_list, input_dict)
        # print("outputs: {}".format(outputs))
        return outputs

    # def simplify_onnx_model(self, model_path):
    #     model = onnx.load(model_path)

    #     print("simplify onnx model: {}".format(model_path))
    #     # print('onnx model graph is:\n{}'.format(model.graph))
    #     model_sim, check = onnxsim.simplify(model)
    #     print("onnxsim check: {}".format(check))
    #     new_path = model_path + ".sim"
    #     onnx.save(model_sim, new_path)
    #     print("simplify model saved to: {}".format(new_path))

    # # run onnx with torch.Tensors
    # def run(self, input_tensor_list):
    #     input_np_list = []
    #     device = torch.device('cuda:0')
    #     for idx, input in enumerate(input_tensor_list):
    #         # print("input-[{}]: {}".format(idx, input))
    #         if isinstance(input, torch.Tensor):
    #             input_np = input.cpu().numpy()
    #             print("input-[{}] shape: {}".format(idx, input_np.shape))
    #             input_np_list.append(input_np)
    #         # list of torch.Tensor
    #         elif isinstance(input, list):
    #             for i, item in enumerate(input):
    #                 item_i_np = item.cpu().numpy()
    #                 print("input-[{}-{}] shape: {}".format(idx, i, item_i_np.shape))
    #                 input_np_list.append(item_i_np)
    #         elif isinstance(input, int):
    #             # input_np = np.int64(input)
    #             input_np = np.array(input).astype(np.int64)
    #             print("input-[{}]: {}".format(idx, input_np))
    #             input_np_list.append(input_np)
    #     # print("input_np_list: {}".format(input_np_list))
    #     output_np_list = self.run_onnx_model(self.ortss, input_np_list)
    #     output_tensor_list = []
    #     for idx, output_np in enumerate(output_np_list):
    #         print("make tensor {} from {}".format(idx, output_np))
    #         output_tensor = torch.tensor(output_np, device=device)
    #         output_tensor_list.append(output_tensor)
    #     return output_tensor_list

    #  %input_2[FLOAT, 1x192x64x3]
    def run_pcr(self, model_path):
        model = onnx.load(model_path)
        try:
            onnx.checker.check_model(model)
            # print("pcr graph: {}".format(onnx.helper.printable_graph(model.graph)))
        except Exception as e:
            print("onnx check model error: {}".format(e))
            # return None, None

        ortss = self.load_onnx_model(model_path)
        image = np.random.rand(1, 192, 64, 3).astype(np.float32)
        self.benchmark(ortss, [image], nwarmup=100, nruns=100)
        result = self.run_onnx_model(ortss, [image])
        angle = result[0][0][0]*180.0 - 90.0
        type_probs = result[1][0]
        type = np.argmax(type_probs, axis=0)
        print("pcr output, angle: {}, type: {}".format(angle, type))
        return True

    # inputs: image float32[1,640,640,3], angle float32[1]
    # outpus: boxes, labels, quads, scores
    def run_psd(self, model_path):
        model = onnx.load(model_path)
        try:
            onnx.checker.check_model(model)
            # print("pcr graph: {}".format(onnx.helper.printable_graph(model.graph)))
        except Exception as e:
            print("onnx check model error: {}".format(e))
            # return False

        ortss = self.load_onnx_model(model_path)
        image = np.random.rand(1, 640, 640, 3).astype(np.float32)
        angle = np.random.rand(1).astype(np.float32)
        self.benchmark(ortss, [angle, image], nwarmup=20, nruns=100)
        return True

        for idx in range(10):
            # ortss = self.load_onnx_model(model_path)
            image = np.random.rand(1, 640, 640, 3).astype(np.float32)
            angle = np.random.rand(1).astype(np.float32)
            # self.benchmark(ortss, [angle, image], nwarmup=10, nruns=10)
            result = self.run_onnx_model(ortss, [angle, image])
            boxes = result[0]
            labels = result[1]
            quads = result[2]
            scores = result[3]
            print("psd output, boxes: {}, scores: {}, labels: {}, quads: {}".format(boxes, scores, labels, quads))
        return True

    def test_resnet50(self, model_path):
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        # print(onnx.helper.printable_graph(model.graph))
        ortss = self.load_onnx_model(model_path)
        image = np.random.rand(1, 3, 224, 224).astype(np.float32)
        result = self.run_onnx_model(ortss, [image])

    def fp32_to_fp16(self, model_path):
        model_fp32 = onnx.load(model_path)
        try:
            onnx.checker.check_model(model_fp32)
            # print("pcr graph: {}".format(onnx.helper.printable_graph(model.graph)))
        except Exception as e:
            print("onnx check model error: {}".format(e))
            return None, None
        
        model_fp16 = float16_converter.convert_float_to_float16(model_fp32, keep_io_types=True)
        new_path = model_path + ".fp16"
        onnx.save(model_fp16, new_path)

if __name__ == "__main__":
    resnet50_path = "./resnet50-v1-12/resnet50-v1-12.onnx"
    pcr_path = "./pcr.onnx"
    psd_path = "./psd.fp16.onnx"
    onwp = OnnxWrapper()
    # onwp.fp32_to_fp16(pcr_path)
    # onwp.fp32_to_fp16(psd_path)
    onwp.run_psd(psd_path)
    # onwp.run_pcr(pcr_path)
    # onwp.print_version()
    # onwp.simplify_onnx_model(roi_algo_onnx_path)
    # onwp.test_resnet50(resnet50_path)