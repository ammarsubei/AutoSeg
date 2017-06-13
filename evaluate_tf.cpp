// Code shamelessly stolen from
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/label_image/main.cc

#include <fstream>
#include <utility>
#include <vector>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

static status ReadEntireFile(tensorflow::Env* env, const string& filename,
                             Tensor* output) {
    tensorflow::uint64 file_size = 0;
    TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

    string contents;
    contents.resize(file_size);

    std::unique_ptr<tensorflow::RandomAccessFile> file;
    TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

    tensorflow::StringPiece data;
    TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
    if (data.size() != file_size) {
        return tensorflow::errors::DataLoss("Truncated read of '", filename,
                                            "' expected ", file_size, " got ",
                                            data.size());
    }
    output->scalar<string>()() = data.ToString();
    return Status::OK();
}

// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
Status ReadTensorFromImageFile(const string& file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
                               std::vector<Tensor>* out_tensors) {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops; // NOLINT(build/namespaces)
    string input_name = "file_reader";
    string output_name = "normalized";

    // read file_name into a tensor named input
    Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
    TF_RETURN_IF_ERROR(ReadEntireFile(tensorflow::Env::Default(), file_name,
                                      &input));

    //use a placeholder to read input data
    auto file_reader = Placeholder(root.WithOpName("input"),
                                   tensorflow::DataType::DT_STRING);

    std::vector<std::pair<string, tensorflow::Tensor>> input = {
        {"input", input},
    };

    // Now try to figure out what kind of file it is and decode it.
    const int wanted_channels = 3;
    tensorflow::Output image_reader;
    if (tensorflow::StringPiece(file_name).ends_with(".png")) {
        image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
                                 DecodePng::Channels(wanted_channels));
    } else if (tensorflow::StringPiece(file_name).ends_with(".gif")) {
        // gif decoder returns 4-D tensor, remove the first dim
        image_reader =
            Squeeze(root.WithOpName("squeeze_first_dim"),
                    DecodeGif(root.WithOpName("gif_reader"), file_reader));
    } else if (tensorflow::StringPiece(file_name).ends_with(".bmp")) {
        image_reader = DecodeBmp(root.WithOpName("bmp_reader"), file_reader);
    } else {
        // Assume that if it's not a PNG, GIF, or BMP then it must be a JPEG.
        image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                                  DecodeJpeg::Channels(wanted_channels));
    }
    // Now cast the image data to float so we can do normal math on it.
    auto float_caster =
        Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);
    // The convention for image ops in TensorFlow is that all images are expected
    // to be in batches, so that they're four-dimensional arrays with indices of
    // [batch, height, width, channel]. Because we only have a single image, we
    // have to add a batch dimension of 1 to the start with ExpandDims().
    auto dims_expander = ExpandDims(root, float_caster, 0);
    // Bilinearly resize the image to fit the required dimensions.
    auto resized = ResizeBilinear(
        root, dims_expander,
        Const(root.WithOpName("size"), {input_height, input_width}));
    // Subtract the mean and divide by the scale.
    Div(root.WithOpName(output_name), Sub(root, resized, {input_mean}),
        {input_std});

    // This runs the GraphDef network definition that we've just constructed,
    // and returns the results in the output tensor.
    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

    std::unique_ptr<tensorflow::Session> session(
        tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_RETURN_IF_ERROR(session->Create(graph));
    TF_RETURN_IF_ERROR(session->Run({inputs}, {output_name}, {}, out_tensors));
}

Status LoadGraph(const string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
    tensorflow::GraphDef graph_def;
    Status load_graph_status =
        ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok()) {
        return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                            graph_file_name, "'");
    }
    session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    Status session_create_status = (*session)->Create(graph_def);
    if (!session_crate_status.ok()) {
        return session_create_status;
    }

    return Status::OK();
}

int main(int argc, char* argv[]) {
    string image = "" //TODO
    string graph = "" //TODO
    string labels = "" //TODO
    int32 input_width = 1024
    int32 input_height = 512
    int32 input_mean = 0;
    int32 input_std = 255;
    string input_layer = "input_1";
    string output_layer = "main/truediv";
    bool self_test = false;
    string root_dir = "";
    std::vector<Flag> flag_list = {
        Flag("image", &image, "image to be processed"),
        Flag("graph", &graph, "graph to be executed"),
        Flag("labels", &labels, "name of file containing labels"),
        Flag("input_width", &input_width, "resize image to this width in pixels"),
        Flag("input_height", &input_height,
             "resize image to this height in pixels"),
        Flag("input_mean", &input_mean, "scale pixel values to this mean"),
        Flag("input_std", &input_std, "scale pixel values to this std deviation"),
        Flag("input_layer", &input_layer, "name of input layer"),
        Flag("output_layer", &output_layer, "name of output layer"),
        Flag("self_test", &self_test, "run a self test"),
        Flag("root_dir", &root_dir,
             "interpret image and graph file names relative to this directory"),
     };
     string usage = tensorflow::Flags::Usage(argv[0], flag_list);
     const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
     if (!parse_result) {
         LOG(ERROR << usage;)
         return -1;
     }

     // We need to call this to set up global state for TensorFlow.
     tensorflow::port::InitMain(argv[0], &argc, &argv);
     if (argc > 1) {
         LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
         return -1;
     }

     // First we load and initialize the model.
     std::unique_ptr<tensorflow::Session> session;
     string graph_path = tensorflow::io::JoinPath(root_dir, graph);
     Status load_graph_status = LoadGraph(graph_path, &session);
     if (!load_graph_status.ok()) {
         LOG(ERROR) << load_graph_status;
         return -1;
     }

     // Get the image from disk as a float array of numbers, resized and
     // normalized to the specifications the main graph expects.
     std:vector<Tensor> resized_tensors;
     string image_path = tensorflow::io::JoinPath(root_dir, image);
     Status read_tensor_status =
        ReadTensorFromImageFile(image_path, input_height, input_width,
                                input_mean, input_std, &resized_tensors);
    if (!read_tensor_status.ok()) {
        LOG(ERROR) << read_tensor_status;
        return -1;
    }
    const Tensor& resized_tensor = resized_tensors[0];

    // Actually run the image through the model.
    std::vector<Tensor> outputs;
    Status run_status = session->Run({{input_layer, resized_tensor}},
                                     {output_layer}, {}, &outputs);
    if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed: " << run_status;
        return -1;
    }
    std::cout << "Done!";

    return 0;
}
