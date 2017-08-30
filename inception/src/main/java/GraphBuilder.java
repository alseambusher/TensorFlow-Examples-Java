import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Tensor;

/**
 * Created by alse on 8/30/17.
 */
class GraphBuilder {
    GraphBuilder(Graph g) {
        this.g = g;
    }

    Output div(Output x, Output y) {
        return binaryOp("Div", x, y);
    }

    Output sub(Output x, Output y) {
        return binaryOp("Sub", x, y);
    }

    Output resizeBilinear(Output images, Output size) {
        return binaryOp("ResizeBilinear", images, size);
    }

    Output expandDims(Output input, Output dim) {
        return binaryOp("ExpandDims", input, dim);
    }

    Output cast(Output value, DataType dtype) {
        return g.opBuilder("Cast", "Cast").addInput(value).setAttr("DstT", dtype).build().output(0);
    }

    Output decodeJpeg(Output contents, long channels) {
        return g.opBuilder("DecodeJpeg", "DecodeJpeg")
                .addInput(contents)
                .setAttr("channels", channels)
                .build()
                .output(0);
    }

    Output constant(String name, Object value) {
        try (Tensor t = Tensor.create(value)) {
            return g.opBuilder("Const", name)
                    .setAttr("dtype", t.dataType())
                    .setAttr("value", t)
                    .build()
                    .output(0);
        }
    }

    Output placeholder(String name, DataType dtype) {
        return g.opBuilder("Placeholder", name)
                .setAttr("dtype", dtype)
                .build()
                .output(0);
    }

    private Output binaryOp(String type, Output in1, Output in2) {
        return g.opBuilder(type, type).addInput(in1).addInput(in2).build().output(0);
    }

    private Graph g;
}
